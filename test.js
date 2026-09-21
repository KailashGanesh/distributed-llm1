// test.js — GPU kernel self-test: each kernel vs CPU reference on real weights
import { FILE_URL, CFG } from "./constants.js";
import { fetchTensorMap, fetchTensors, bf16BytesToF32Bits, bf16BitsToF32 } from "./weights.js";
import { initKernels } from "./kernels.js";
import { Stage } from "./pipeline.js";

const out = (s) => { console.log(s); document.getElementById("log").textContent += s + "\n"; };
function mulberry32(seed) {
  return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}

const names = [
  "model.embed_tokens.weight", "model.norm.weight",
  "model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight",
  "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight",
  "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight",
  "model.layers.0.self_attn.q_proj.bias", "model.layers.0.self_attn.k_proj.bias",
  "model.layers.0.self_attn.v_proj.bias",
  "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.down_proj.weight",
];
const uparams = document.getElementById("info");
uparams.textContent = "fetching weights...";
const map = await fetchTensorMap();
const tensors = await fetchTensors(map, names);
const bits = {};
for (const [n, b] of Object.entries(tensors)) { bits[n] = bf16BytesToF32Bits(b); console.log("tensor", n, b.length, "bytes ->", bits[n].length, "u32"); }
uparams.textContent = `weights ok (${Object.keys(bits).length} tensors)`;

const device = (await (await navigator.gpu.requestAdapter()).requestDevice());
device.onuncapturederror = (e) => out("GPU ERR: " + (e.error?.message || e.error?.code || e));
const k = await initKernels(device);
const W0 = "model.layers.0.";
const stats = (a) => { let mn = Infinity, mx = -Infinity, s = 0; for (const v of a) { mn = Math.min(mn, v); mx = Math.max(mx, v); s += v; } return `mean=${(s / a.length).toFixed(5)} min=${mn.toFixed(5)} max=${mx.toFixed(5)}`; };

// deterministic hidden row
const rnd = mulberry32(42);
const x = new Float32Array(CFG.hiddenSize);
for (let i = 0; i < CFG.hiddenSize; i++) x[i] = rnd() * 0.2 - 0.1;

// ---- TEST 1: matmul (n=1, dout=1536, din=1536) vs CPU
{
  const xBuf = k.createF32(CFG.hiddenSize); k.writeF32(xBuf, x);
  const wq = bits[W0 + "self_attn.q_proj.weight"];
  const wBuf = k.loadWeightBits(wq, "q");
  const gpu = await k.readF32(k.matmul(xBuf, wBuf, 1, CFG.hiddenSize, 1536), 1536, "mm1");
  const wF = bf16BitsToF32(wq);
  const cpu = new Float32Array(1536);
  for (let j = 0; j < 1536; j++) { let acc = 0; for (let p = 0; p < 1536; p++) acc += x[p] * wF[j * 1536 + p]; cpu[j] = acc; }
  let diff = 0; for (let i = 0; i < 1536; i++) diff = Math.max(diff, Math.abs(gpu[i] - cpu[i]));
  out(`[mm n=1 1536x1536] maxAbsDiff=${diff.toExponential(2)} — gpu(${stats(gpu)}) cpu(${stats(cpu)}) ${diff < 1e-2 ? "PASS" : "*** FAIL ***"}`);
}

// ---- TEST 2: matmul n=30 (prefill-like) q_proj
{
  const n = 30;
  const X = new Float32Array(n * CFG.hiddenSize);
  const r2 = mulberry32(7);
  for (let i = 0; i < X.length; i++) X[i] = r2() * 0.2 - 0.1;
  const xBuf = k.createF32(X.length); k.writeF32(xBuf, X);
  const wq = bits[W0 + "self_attn.q_proj.weight"];
  const gpu = await k.readF32(k.matmul(xBuf, k.loadWeightBits(wq), n, 1536, 1536), n * 1536, "mm30");
  const wF = bf16BitsToF32(wq);
  let diff = 0;
  for (let t = 0; t < n; t++) for (let j = 0; j < 1536; j++) { let acc = 0; for (let p = 0; p < 1536; p++) acc += X[t * 1536 + p] * wF[j * 1536 + p]; diff = Math.max(diff, Math.abs(gpu[t * 1536 + j] - acc)); }
  out(`[mm n=30] maxAbsDiff=${diff.toExponential(2)} ${diff < 1e-2 ? "PASS" : "*** FAIL ***"}`);
}

// ---- TEST 3: rmsnorm n=1 and n=30
{
  const ln = bits[W0 + "input_layernorm.weight"];
  const gF = bf16BitsToF32(ln);
  for (const n of [1, 30]) {
    const X = new Float32Array(n * CFG.hiddenSize);
    const r3 = mulberry32(11 + n);
    for (let i = 0; i < X.length; i++) X[i] = r3() * 0.3 - 0.15;
    const xBuf = k.createF32(X.length); k.writeF32(xBuf, X);
    const gpu = await k.readF32(k.rmsnorm(xBuf, k.loadWeightBits(ln), n), n * 1536, `rms${n}`);
    let diff = 0;
    for (let t = 0; t < n; t++) {
      let ss = 0; for (let i = 0; i < 1536; i++) ss += X[t * 1536 + i] ** 2;
      const scale = 1 / Math.sqrt(ss / 1536 + 1e-6);
      for (let i = 0; i < 1536; i++) diff = Math.max(diff, Math.abs(gpu[t * 1536 + i] - X[t * 1536 + i] * scale * gF[i]));
    }
    out(`[rmsnorm n=${n}] maxAbsDiff=${diff.toExponential(2)} ${diff < 1e-3 ? "PASS" : "*** FAIL ***"}`);
  }
}

// ---- TEST 4: attention n=1, T0=3 (manually seeded KV)
{
  const n = 1, T0 = 3, HD = 128;
  const rnd4 = mulberry32(99);
  const q = new Float32Array(12 * HD); for (let i = 0; i < q.length; i++) q[i] = rnd4() * 2 - 1;
  const kcache = new Float32Array(4096 * 2 * HD), vcache = new Float32Array(4096 * 2 * HD);
  for (let p = 0; p < T0 + n; p++) for (let hv = 0; hv < 2; hv++) for (let e = 0; e < HD; e++) {
    kcache[p * 2 * HD + hv * HD + e] = rnd4() * 2 - 1;
    vcache[p * 2 * HD + hv * HD + e] = rnd4() * 2 - 1;
  }
  const cpuAttn = (Tend) => {
    const cpu = new Float32Array(12 * HD);
    const scale = 1 / Math.sqrt(HD);
    for (let h = 0; h < 12; h++) {
      const kvh = Math.floor(h / 6);
      const scores = [];
      for (let p = 0; p < Tend; p++) {
        let s = 0; for (let e = 0; e < HD; e++) s += q[h * HD + e] * kcache[p * 2 * HD + kvh * HD + e];
        scores.push(s * scale);
      }
      const mx = Math.max(...scores); let sume = 0; const w = scores.map((s) => { const e = Math.exp(s - mx); sume += e; return e / sume; });
      for (let e = 0; e < HD; e++) { let acc = 0; for (let p = 0; p < Tend; p++) acc += vcache[p * 2 * HD + kvh * HD + e] * w[p]; cpu[h * HD + e] = acc; }
    }
    return cpu;
  };
  const qB = k.createF32(q.length); k.writeF32(qB, q);
  const kCB = k.createF32(kcache.length); k.writeF32(kCB, kcache);
  const vCB = k.createF32(vcache.length); k.writeF32(vCB, vcache);
  {
    const gpu = await (async () => { const b = k.attention(qB, kCB, vCB, T0, n); return await k.readF32(b, 12 * HD, "attn1"); })();
    const cpu = cpuAttn(T0 + n);
    let diff = 0; for (let i = 0; i < cpu.length; i++) diff = Math.max(diff, Math.abs(gpu[i] - cpu[i]));
    out(`[attention n=1 T0=${T0}] maxAbsDiff=${diff.toExponential(2)} — gpu first4=${Array.from(gpu.slice(0, 4)).map((v) => v.toFixed(3))} cpu first4=${Array.from(cpu.slice(0, 4)).map((v) => v.toFixed(3))} ${diff < 1e-2 ? "PASS" : "*** FAIL ***"}`);
  }
  for (const t0v of [0, 1, 5, 20]) {
    if (t0v === T0) continue;
    const gpu = await (async () => { const b = k.attention(qB, kCB, vCB, t0v, n); return await k.readF32(b, 12 * HD, `attn${t0v}`); })();
    const cpu = cpuAttn(t0v + n);
    let diff = 0; for (let i = 0; i < cpu.length; i++) diff = Math.max(diff, Math.abs(gpu[i] - cpu[i]));
    out(`[attention n=1 T0=${t0v}] maxAbsDiff=${diff.toExponential(2)} ${diff < 1e-2 ? "PASS" : "*** FAIL ***"}`);
  }
}

// ---- TEST 5: full single decoder layer n=1 vs CPU (real layer-0 weights)
{
  const D = 1536, stage = new Stage(k, 0, 1);
  stage.W[W0 + "input_layernorm.weight"] = k.loadWeightBits(bits[W0 + "input_layernorm.weight"]);
  stage.W[W0 + "post_attention_layernorm.weight"] = k.loadWeightBits(bits[W0 + "post_attention_layernorm.weight"]);
  for (const nm of Object.keys(bits)) if (nm.startsWith(W0)) stage.W[nm] = k.loadWeightBits(bits[nm]);
  stage.reset();
  stage.T0 = 0;
  if (!stage.W[W0 + "self_attn.q_proj.weight"]) {
    // continue to next test
  } else {
  // n=1, posStart=17
  const POS = 17, HD = 128;
  const h = stage.forward(x, 1, POS);
  const gpu = await k.readF32(h, 1536, "fwd1");
  // CPU reference
  const F = {}; for (const nm of Object.keys(bits)) F[nm] = bf16BitsToF32(bits[nm]);
  const rms = (v, g) => { let ss = 0; for (const e of v) ss += e * e; const s = 1 / Math.sqrt(ss / 1536 + 1e-6); return v.map((e2, i2) => e2 * s * g[i2]); };
  const mt = (v, w, dout, din) => { const o = new Float32Array(dout); for (let j = 0; j < dout; j++) { let a = 0; for (let p2 = 0; p2 < din; p2++) a += v[p2] * w[j * din + p2]; o[j] = a; } return o; };
  const rope = (v, nHeads) => { const o = v.slice(); for (let hh = 0; hh < nHeads; hh++) for (let half = 0; half < 64; half++) {
    const pos = POS, freq = Math.pow(1000000, -2 * half / 128), ang = pos * freq;
    const x1 = v[hh * HD + half], x2 = v[hh * HD + half + 64];
    o[hh * HD + half] = x1 * Math.cos(ang) - x2 * Math.sin(ang); o[hh * HD + half + 64] = x1 * Math.sin(ang) + x2 * Math.cos(ang);
  } return o; };
  let hCPU = Array.from(x);
  const a1 = rms(hCPU, F[W0 + "input_layernorm.weight"]);
  const q = mt(a1, F[W0 + "self_attn.q_proj.weight"], 1536, 1536), kk = mt(a1, F[W0 + "self_attn.k_proj.weight"], 256, 1536), vv = mt(a1, F[W0 + "self_attn.v_proj.weight"], 256, 1536);
  const addB = (v, nm) => { const b = F[W0 + nm]; for (let i2 = 0; i2 < v.length; i2++) v[i2] += b[i2 % b.length]; return v; };
  addB(q, "self_attn.q_proj.bias"); addB(kk, "self_attn.k_proj.bias"); addB(vv, "self_attn.v_proj.bias");
  const qr = rope(q, 12), kr = rope(kk, 2);
  // attention vs only own position
  const att = new Float32Array(1536);
  for (let hh = 0; hh < 12; hh++) { const kvh = Math.floor(hh / 6); let num = 0;
    let s = 0; for (let e = 0; e < HD; e++) s += qr[hh * HD + e] * kr[kvh * HD + e] * 0.08838834764831845;
    const w = 1.0;
    for (let e = 0; e < HD; e++) att[hh * HD + e] = vv[kvh * HD + e] * w;
  }
  const o = mt(att, F[W0 + "self_attn.o_proj.weight"], 1536, 1536);
  const h1 = hCPU.map((v, i2) => v + o[i2]);
  const a2 = rms(h1, F[W0 + "post_attention_layernorm.weight"]);
  const g = mt(a2, F[W0 + "mlp.gate_proj.weight"], 8960, 1536), u = mt(a2, F[W0 + "mlp.up_proj.weight"], 8960, 1536);
  const m = g.map((v, i2) => (v / (1 + Math.exp(-v))) * u[i2]);
  const d = mt(m, F[W0 + "mlp.down_proj.weight"], 1536, 8960);
  const outCPU = h1.map((v, i2) => v + d[i2]);
  let diff = 0; for (let i = 0; i < 1536; i++) diff = Math.max(diff, Math.abs(gpu[i] - outCPU[i]));
  out(`[layer0 forward n=1] maxAbsDiff=${diff.toExponential(2)} — gpu(${stats(gpu)}) cpu(${stats(outCPU)}) ${diff < 0.1 ? "PASS" : "*** FAIL ***"}`);
  }
}

// ---- TEST 6: lm_head chunk (rmsnorm + chunked matmul) — replicate sampleLast
{
  const normG = bits["model.norm.weight"];
  const xBuf = k.createF32(1536); k.writeF32(xBuf, x);
  const nrm = k.rmsnorm(xBuf, k.loadWeightBits(normG), 1);
  const nrmF = await k.readF32(nrm, 1536, "nrm");
  out(`[final rmsnorm out] ${stats(nrmF)} (expect std near 1: g values are O(~1.4), X small => std ~0.03 fine)`);
  const chunk = bits["model.embed_tokens.weight"].slice(0, 8192 * 1536);
  const gpu = await k.readF32(k.matmul(nrm, k.loadWeightBits(chunk), 1, 1536, 8192), 8192, "lhead");
  const cpu = new Float32Array(8192);
  const wF = bf16BitsToF32(chunk), nF = bf16BitsToF32(bits["model.norm.weight"]);
  let ss = 0; for (const e of x) ss += e * e; const scale = 1 / Math.sqrt(ss / 1536 + 1e-6);
  const xnrm = x.map((v, i2) => v * scale * nF[i2]);
  for (let j = 0; j < 8192; j++) { let a = 0; for (let p = 0; p < 1536; p++) a += xnrm[p] * wF[j * 1536 + p]; cpu[j] = a; }
  let diff = 0; for (let i = 0; i < 8192; i++) diff = Math.max(diff, Math.abs(gpu[i] - cpu[i]));
  out(`[lm_head chunk8192] maxAbsDiff=${diff.toExponential(2)} — gpu(${stats(gpu)}) cpu(${stats(cpu)}) ${diff < 0.1 ? "PASS" : "*** FAIL ***"}`);
}
out("done");
