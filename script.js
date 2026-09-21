// script.js — app glue: UI, boot, stage/leader role assignment, chat loop
import { ROOM } from "./config.js";
import { Signaling } from "./signaling.js";
import { RTC, FRAME, pack } from "./rtc.js";
import { CFG } from "./constants.js";
import { assignLayers, Stage, LeaderPipeline } from "./pipeline.js";
import { fetchTensorMap, fetchTensors, bf16BytesToF32Bits, tensorsForLayers } from "./weights.js";

const $ = (id) => document.getElementById(id);
let stageIdx = 0; // set on assign
const tag = () => role === "leader" ? "[leader]" : role === "stage" ? `[follower #${stageIdx}]` : "[node]";
const log = (s) => {
  console.log(tag(), s);
  $("log").textContent += s + "\n";
};

const adapter = await navigator.gpu?.requestAdapter();
const device = adapter && (await adapter.requestDevice());
device.onuncapturederror = (e) => log("GPU error: " + e.error?.message);
device.ondeviceuncapturederror = (e) => log("GPU error: " + e.error?.message);
const gpuOn = !!device;
const SOLO = new URLSearchParams(location.search).get("solo") === "1";
const myProfile = { memGB: navigator.deviceMemory || 4, gpu: gpuOn };
const myId = String(Math.floor(100000 + Math.random() * 900000));
const signaling = new Signaling(ROOM, myId, onSignal, () => myProfile);
signaling.debug = (s) => log(s);
const rtc = new RTC(myId, (peer, m) => signaling.sendTo(peer, m));
rtc.onError = (e) => log("RTC error: " + e);
rtc.onChannelState = (id, st) => { renderRoster(); log(`channel ${id}: ${st}`); };

let role = "waiting"; // waiting | leader | stage
let myTask = null;
let stage = null;
const stageReady = {};
let tokenizer = null;


log(`peer ${myId} | gpu: ${gpuOn} | mem: ${myProfile.memGB}GB | room: ${ROOM} | solo: ${SOLO ? "yes" : "no"}`);
if (SOLO) { tryLeader(); }

// -------- UI
function renderRoster() {
  const entries = [[myId, myProfile], ...Object.entries(signaling.peers)
    .filter(([, p]) => p.profile)
    .map(([id, p]) => [id, p.profile])];
  $("roster").textContent = entries.map(([id, p]) =>
    `${id}${id === myId ? " (me)" : ""} | gpu:${!!p.gpu} mem:${p.memGB || "?"}GB ch:${rtc.channels[id]?.readyState ?? "-"}`,
  ).join("\n");
}

// -------- signaling
// DataChannel control messages (assign/reset/stage-ready)
rtc.onControl = (from, msg) => {
  if (msg.t === "assign" && role === "waiting" && !stage) runStage(msg.start, msg.end, msg.idx ?? 1);
  if (msg.t === "stage-ready") { stageReady[from] = true; log(`stage ${from} ready`); }
  if (msg.t === "reset" && stage) stage.reset();
};

function onSignal(msg) {
  if (msg.type === "hello") { renderRoster(); if (signaling.isLeader() && role === "waiting") tryLeader(); return; }
  if (msg.t === "hb") { log(`heartbeat from leader ${msg.from}`); return; }
  if (msg.to === myId && ["offer", "answer", "ice"].includes(msg.type)) rtc.handleSignal(msg.from, msg);
}

// leader heartbeat every 5s over PubNub (tiny)
setInterval(() => { if (role === "leader") signaling.publish({ t: "hb" }); }, 5000);

// -------- leader
async function tryLeader() {
  role = "leader";
  const peers = SOLO ? [] : Object.keys(signaling.peers).filter((id) => signaling.peers[id].profile?.gpu);
  log(`${SOLO ? "solo run: full model in this tab — " : ""}leader elected (${peers.length} stage peer(s): ${peers.join(",") || "none"})`);
  if (SOLO) { renderRoster(); }
  for (const id of peers) rtc.connect(id);

  const ranges = SOLO
    ? [{ id: myId, start: 0, end: CFG.numLayers }]
    : assignLayers([
    { id: myId, profile: myProfile },
    ...peers.map((id) => ({ id, profile: signaling.peers[id].profile })),
  ]);
  const myRange = ranges.find((s) => s.id === myId);
  $("status").textContent = `I'm leader ${myId} — layers [${myRange.start},${myRange.end}) + embed/lm_head`;

  // hand out assignments instantly — followers download in parallel with the leader
  for (const s of ranges) {
    if (s.id !== myId) {
      const idx = ranges.findIndex((x) => x.id === s.id);
      rtc.send(s.id, JSON.stringify({ t: "assign", start: s.start, end: s.end, idx }));
    }
  }

  // leader downloads its OWN layer slice + embed + final norm (parallel with stages)
  const map = await fetchTensorMap();
  const myTensorNames = tensorsForLayers(myRange.start, myRange.end, true);
  const tensors = await fetchTensors(map, myTensorNames, (frac) => {
    $("status").textContent = `I'm leader ${myId} — downloading layers [${myRange.start},${myRange.end}) — ${(frac * 100).toFixed(0)}%`;
  });
  log(`leader weights fetched (${Object.keys(tensors).length} tensors)`);

  // wait for every remote stage to report ready
  await Promise.all(ranges.filter((s) => s.id !== myId).map((s) => new Promise((r) => {
    const iv = setInterval(() => (stageReady[s.id] ? (clearInterval(iv), r()) : 0), 200);
  })));
  log("all stages ready");

  const kctx = await import("./kernels.js").then((m) => m.initKernels(device));
  myTask = new LeaderPipeline(kctx, [myRange.start, myRange.end], ranges, {
    myId,
    sendFrame: (peer, bytes) => rtc.send(peer, bytes),
    onToken: appendToken,
  });
  const myBits = {};
  for (const [n, bytes] of Object.entries(tensors)) {
    console.log(tag(), n, `${bytes.length / 1e6 | 0}MB`);
    myBits[n] = bf16BytesToF32Bits(bytes);
  }
  for (const [n, bits] of Object.entries(myBits)) {
    if (n === "model.embed_tokens.weight" || n === "model.norm.weight") continue;
    myTask.stage.W[n] = kctx.loadWeightBits(bits, n);
  }
  myTask.normScratch = myBits["model.norm.weight"];
  myTask.loadLeaderWeights(
    myBits["model.embed_tokens.weight"],
    myBits["model.norm.weight"],
  );
  await loadTokenizer();
  $("status").textContent = "I'm leader — all stages online, ready";
  console.log("chain:", ranges);
}

// -------- stage
function runStage(start, end, idx) {
  role = "stage";
  stageIdx = idx;
  log(`stage role assigned: layers [${start},${end})`);
  (async () => {
    const map = await fetchTensorMap();
    $("status").textContent = `I'm follower #${idx} — fetching layers [${start},${end})...`;
    const tensors = await fetchTensors(map, tensorsForLayers(start, end, false), (frac) => {
      $("status").textContent = `I'm follower #${idx} — fetching layers [${start},${end}) — ${(frac * 100).toFixed(0)}%`;
    });
  const kctx = await import("./kernels.js").then((m) => m.initKernels(device));
  stage = new Stage(kctx, start, end);
  for (const [n, bytes] of Object.entries(tensors)) {
    console.log(tag(), n, `${bytes.length / 1e6 | 0}MB`);
    stage.W[n] = kctx.loadWeightBits(bf16BytesToF32Bits(bytes), n);
  }
  stage.reset();
    // report readiness to the leader (lowest peer id in the room)
    const leaderId = Object.keys(signaling.peers).sort()[0];
    rtc.send(leaderId, JSON.stringify({ t: "stage-ready" }));
    $("status").textContent = `I'm follower #${idx} — ready (layers [${start},${end}))`;
  })();
}

// -------- DataChannel frames
rtc.onRawData = (from, frame) => {
  if (role === "leader") { myTask?.onFrame(from, frame); return; }
  if (role === "stage" && stage) handleStageFrame(from, frame);
};

async function handleStageFrame(from, frame) {
  const D = CFG.hiddenSize;
  const t0 = performance.now();
  const n = Math.floor(frame.f32.length / D);
  const hBuf = stage.forward(frame.f32, n, frame.posStart);
  const f32 = await stage.k.readF32(hBuf, n * D);
  const out = frame.msgType === FRAME.FINAL_REQ ? f32.subarray((n - 1) * D, n * D) : f32;
  rtc.send(from, pack(FRAME.RESP, frame.seqId, stage.T0, out));
  log(`hop n=${n} pos=${frame.posStart} ${ (performance.now() - t0).toFixed(0) }ms`);
}

// -------- chat
async function loadTokenizer() {
  try {
    const { AutoTokenizer } = await import("https://esm.run/@huggingface/transformers");
    tokenizer = await AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct");
    log("tokenizer ready");
  } catch (e) {
    log("tokenizer failed: " + e.message);
  }
}

const generatedIds = [];
function appendToken(tok) {
  generatedIds.push(tok);
  $("out").textContent = tokenizer
    ? tokenizer.decode(generatedIds, { skip_special_tokens: false })
    : generatedIds.join(",");
}

$("send").onclick = async () => {
  const q = $("in").value.trim();
  if (!q || !myTask) return;
  $("in").value = "";
  $("out").textContent = "";
  generatedIds.length = 0;
  const templated = `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n${q}<|im_end|>\n<|im_start|>assistant\n`;
  const tokens = tokenizer.encode(templated, { add_special_tokens: false });
  for (const s of myTask.stages) if (s.id !== myId) rtc.send(s.id, JSON.stringify({ t: "reset" }));
  myTask?.stage?.reset(); // leader's own Stage (module-level `stage` is follower-only)
  log(`prompt tokens: ${tokens.length}`);
  const t0 = performance.now();
  const nGen = { v: 0 };
  const prev = myTask.onToken;
  myTask.onToken = (tok) => { nGen.v++; prev(tok); };
  try {
    await myTask.generate(tokens, 128);
  } catch (e) {
    log("generation error: " + e.message);
  }
  myTask.onToken = prev;
  const dt = (performance.now() - t0) / 1000;
  log(`done: ${nGen.v} tok in ${dt.toFixed(1)}s => ${(nGen.v / dt).toFixed(2)} tok/s`);
};

