// pipeline.js — Stage forward logic + Leader decoder + roster-based layer split
import { CFG, FRAME } from "./constants.js";
import { bf16BitsToF32 } from "./weights.js";

// Split layers across peers proportionally to capacity score. Ordered by peerId.
export function assignLayers(peers) {
  // peers: [{id, profile:{memGB, gpu}}] — includes me
  const caps = peers.map((p) => ({
    id: p.id,
    cap: (p.profile?.memGB ?? 4) * (p.profile?.gpu ? 1.5 : 0.5),
  }));
  const total = caps.reduce((a, b) => a + b.cap, 0);
  let assigned = 0;
  return caps.map((c, i) => {
    const share =
      i === caps.length - 1
        ? CFG.numLayers - assigned
        : Math.max(1, Math.round((c.cap / total) * CFG.numLayers));
    const range = [assigned, Math.min(CFG.numLayers, assigned + share)];
    assigned = range[1];
    return { id: c.id, start: range[0], end: range[1] };
  });
}

// Runs a contiguous block of decoder layers, resumable with own KV cache.
export class Stage {
  constructor(kctx, start, end) {
    this.k = kctx;
    this.start = start;
    this.end = end;
    this.W = {};   // tensor name -> GPUBuffer (u32 bf16 bits)
    this.kv = {};  // layer -> {k: GPUBuffer, v: GPUBuffer}
    this.T0 = 0;   // rows in own KV cache
    this.reset();
  }

  reset() {
    const rowF = CFG.numKVHeads * CFG.headDim;
    for (let L = this.start; L < this.end; L++) {
      this.kv[L] ??= {
        k: this.k.createF32(CFG.maxPos * rowF),
        v: this.k.createF32(CFG.maxPos * rowF),
      };
    }
    this.T0 = 0;
  }

  forward(hiddenF32, n, posStart) {
    const k = this.k;
    let h = k.createF32(n * CFG.hiddenSize);
    k.writeF32(h, hiddenF32);
    for (let L = this.start; L < this.end; L++) {
      const p = `model.layers.${L}.`;
      // ---- attention
      const x = k.rmsnorm(h, this.W[p + "input_layernorm.weight"], n);
      let q = k.matmul(x, this.W[p + "self_attn.q_proj.weight"], n, CFG.hiddenSize, CFG.numQHeads * CFG.headDim);
      let kk = k.matmul(x, this.W[p + "self_attn.k_proj.weight"], n, CFG.hiddenSize, CFG.numKVHeads * CFG.headDim);
      let vv = k.matmul(x, this.W[p + "self_attn.v_proj.weight"], n, CFG.hiddenSize, CFG.numKVHeads * CFG.headDim);
      if (this.W[p + "self_attn.q_proj.bias"]) q = k.addBias(q, this.W[p + "self_attn.q_proj.bias"], n, CFG.numQHeads * CFG.headDim);
      if (this.W[p + "self_attn.k_proj.bias"]) kk = k.addBias(kk, this.W[p + "self_attn.k_proj.bias"], n, CFG.numKVHeads * CFG.headDim);
      if (this.W[p + "self_attn.v_proj.bias"]) vv = k.addBias(vv, this.W[p + "self_attn.v_proj.bias"], n, CFG.numKVHeads * CFG.headDim);
      k.ropeQ(q, posStart, n);
      k.ropeK(kk, posStart, n);
      this.appendKv(L, kk, vv, n);
      const attn = k.attention(q, this.kv[L].k, this.kv[L].v, this.T0, n);
      const o = k.matmul(attn, this.W[p + "self_attn.o_proj.weight"], n, CFG.numQHeads * CFG.headDim, CFG.hiddenSize);
      const h1 = k.add(h, o, n * CFG.hiddenSize);
      // ---- mlp
      const x2 = k.rmsnorm(h1, this.W[p + "post_attention_layernorm.weight"], n);
      const g = k.matmul(x2, this.W[p + "mlp.gate_proj.weight"], n, CFG.hiddenSize, CFG.intermediateSize);
      const u = k.matmul(x2, this.W[p + "mlp.up_proj.weight"], n, CFG.hiddenSize, CFG.intermediateSize);
      const m = k.siluMul(g, u, n * CFG.intermediateSize);
      const d = k.matmul(m, this.W[p + "mlp.down_proj.weight"], n, CFG.intermediateSize, CFG.hiddenSize);
      h = k.add(h1, d, n * CFG.hiddenSize);
    }
    this.T0 += n;
    return h;
  }

  appendKv(L, kkBuf, vvBuf, n) {
    const rowBytes = CFG.numKVHeads * CFG.headDim * 4;
    const dst = this.T0 * rowBytes;
    const enc = this.k.device.createCommandEncoder();
    enc.copyBufferToBuffer(kkBuf, 0, this.kv[L].k, dst, n * rowBytes);
    enc.copyBufferToBuffer(vvBuf, 0, this.kv[L].v, dst, n * rowBytes);
    this.k.device.queue.submit([enc.finish()]);
  }
}

// Leader orchestrates: embed -> its layers -> hops through chain -> final norm/lm_head
export class LeaderPipeline {
  constructor(kctx, myRange, stages, opts) {
    this.k = kctx;
    this.myId = opts.myId;
    this.stage = new Stage(kctx, myRange[0], myRange[1]);
    this.stages = stages;                       // [{id,start,end}] ordered, leader first
    this.stagesTail = stages.filter((s) => s.id !== opts.myId);
    this.singleNode = this.stagesTail.length === 0 && myRange[1] >= CFG.numLayers;
    this.sendFrame = opts.sendFrame;            // (peerId, Uint8Array frame)
    this.onToken = opts.onToken ?? (() => {});
    this.embedBits = null;   // Uint32Array vocab*D
    this.embedChunkBufs = []; // GPU chunks for lm_head
    this.chunkRows = 8192;
    this.normBuf = null;
    this.pending = new Map(); // seq -> {resolve, reject, timer}
    this.seq = 1;
  }

  loadLeaderWeights(embedBits, normBits) {
    const k = this.k;
    this.embedBits = embedBits;
    this.normBuf = k.loadWeightBits(normBits);
    // chunk lm_head (= tied embed) to respect maxStorageBufferBindingSize
    for (let c = 0; c * this.chunkRows < CFG.vocabSize; c++) {
      const off = c * this.chunkRows * CFG.hiddenSize;
      const rows = Math.min(this.chunkRows, CFG.vocabSize - c * this.chunkRows);
      this.embedChunkBufs[c] = k.loadWeightBits(embedBits.subarray(off, off + rows * CFG.hiddenSize));
    }
  }

  embedTokens(tokens) {
    const D = CFG.hiddenSize;
    const out = new Float32Array(tokens.length * D);
    for (let t = 0; t < tokens.length; t++) {
      out.set(bf16BitsToF32(this.embedBits, tokens[t] * D, D), t * D);
    }
    return out;
  }

  lastTokRow(f32, n) {
    return f32.slice((n - 1) * CFG.hiddenSize, n * CFG.hiddenSize);
  }

  static stats(f32, name = "") {
    let mn = Infinity, mx = -Infinity, s = 0, s2 = 0;
    for (const v of f32) { if (v < mn) mn = v; if (v > mx) mx = v; s += v; s2 += v * v; }
    const n = f32.length || 1;
    console.log(`[leader] stats ${name} n=${f32.length} mean=${(s / n).toFixed(4)} std=${(Math.sqrt(Math.max(s2 / n - (s / n) ** 2, 0))).toFixed(4)} min=${mn.toFixed(4)} max=${mx.toFixed(4)}`);
  }

  async sampleLast(hiddenF32) {
    // final RMSNorm on last token row, then chunked lm_head, greedy argmax
    const k = this.k;
    const t0 = performance.now();
    console.log("[leader] sample: norm +", this.embedChunkBufs.length, "lm_head chunks");
    LeaderPipeline.stats(hiddenF32.slice(0, 1536), "last-hidden");
    const xBuf = k.createF32(CFG.hiddenSize);
    k.writeF32(xBuf, hiddenF32.slice(0, CFG.hiddenSize));
    const nrm = k.rmsnorm(xBuf, this.normBuf, 1);
    let best = 0, bestV = -Infinity;
    const top = [];
    for (let c = 0; c < this.embedChunkBufs.length; c++) {
      const t1 = performance.now();
      const rows = c === this.embedChunkBufs.length - 1
        ? CFG.vocabSize - c * this.chunkRows : this.chunkRows;
      const outBuf = k.matmul(nrm, this.embedChunkBufs[c], 1, CFG.hiddenSize, rows);
      const logits = await k.readF32(outBuf, rows, `lmhead chunk ${c}`);
      console.log(`[leader] chunk ${c} (${rows} rows) in ${(performance.now() - t1).toFixed(0)}ms`);
      for (let i = 0; i < rows; i++) {
        const global = c * this.chunkRows + i;
        const v = logits[i];
        if (v > bestV) { bestV = v; best = global; }
        if (top.length < 8 || v > top[top.length - 1][1]) {
          top.push([global, v]);
          top.sort((a, b) => b[1] - a[1]);
          if (top.length > 8) top.pop();
        }
      }
    }
    console.log("[leader] top tokens:", top.map(([t, v]) => `${t}(${tokenizerWouldPrint(t)}):${v.toFixed(2)}`).join(" "));
    function tokenizerWouldPrint(t) { return t; }
    console.log(`[leader] sample done in ${(performance.now() - t0).toFixed(0)}ms -> token ${best}`);
    return best;
  }

  onFrame(from, frame) {
    console.log(`[leader] frame rx from=${from} msgType=${frame.msgType} seq=${frame.seqId} floats=${frame.f32.length}`);
    const p = this.pending.get(frame.seqId);
    if (p) {
      clearTimeout(p.timer);
      this.pending.delete(frame.seqId);
      p.resolve(frame);
    }
  }

  // forward one block (prefill: many tokens; decode: n=1) through the full chain
  async forward(hiddenF32, n, posStart) {
    console.log(`[leader] forward n=${n} posStart=${posStart} hidden=${hiddenF32?.length}f32`);
      LeaderPipeline.stats(hiddenF32, "embed-input");
    // own layers first (leader owns layers [0, a))
    let cur, curN = n;
    if (this.stage.end > this.stage.start) {
      const h = this.stage.forward(hiddenF32, n, posStart);
      cur = curN > 1
        ? await this.k.readF32(h, n * CFG.hiddenSize)
        : await this.k.readF32(h, CFG.hiddenSize);
      curN = n;
      console.log(`[leader] own layers done T0=${this.stage.T0}`);
      LeaderPipeline.stats(cur.slice((curN - 1) * CFG.hiddenSize, curN * CFG.hiddenSize), "own-out last row");
      if (this.stagesTail.length === 0) return this.lastTokRow(cur, curN);
    } else {
      cur = this.lastTokRow(hiddenF32, n);
      if (this.stagesTail.length === 0) return cur;
      curN = 1;
    }
    for (let i = 0; i < this.stagesTail.length; i++) {
      const isLast = i === this.stagesTail.length - 1;
      let resp;
      if (curN === 1) {
        resp = await this.hop(this.stagesTail[i].id, cur, 1, posStart, isLast);
        if (!isLast) resp.f32 = resp.f32.slice(0, CFG.hiddenSize); // RESP is last row
      } else {
        resp = await this.hop(this.stagesTail[i].id, cur, curN, posStart, isLast);
        if (!isLast) { /* RESP full matrix, forwarded as-is */ }
      }
      // FINAL_REQ reply is already just the last row
      if (isLast) {
        LeaderPipeline.stats(resp.f32, "final-hidden (from follower)");
        return resp.f32;
      }
      cur = resp.f32;
    }
    return Array.isArray(cur) ? cur : cur; // unreachable
  }

  hop(peerId, hiddenF32, n, posStart, isFinal) {
    return new Promise((resolve, reject) => {
      const seq = this.seq++;
      const nFloats = hiddenF32.length;
      console.log(`[leader] hop -> ${peerId} type=${isFinal ? "FINAL" : "HIDDEN"} seq=${seq} n=${n} nFloats=${nFloats} posStart=${posStart}`);
      const head = new Int32Array([isFinal ? FRAME.FINAL_REQ : FRAME.HIDDEN_REQ, seq, posStart, nFloats]);
      const bytes = 16 + nFloats * 4;
      const pay = new Uint8Array(bytes);
      pay.set(new Uint8Array(head.buffer), 0);
      pay.set(new Uint8Array(hiddenF32.buffer, hiddenF32.byteOffset, hiddenF32.byteLength), 16);
      const timer = setTimeout(() => {
        this.pending.delete(seq);
        reject(new Error("hop timeout: " + peerId));
      }, 180000);
      this.pending.set(seq, { resolve, reject, timer });
      this.sendFrame(peerId, pay);
    });
  }

  onFrame(from, frame) {
    console.log(`[leader] frame rx from=${from} msgType=${frame.msgType} seq=${frame.seqId} floats=${frame.f32.length}`);
    const p = this.pending.get(frame.seqId);
    if (p) {
      clearTimeout(p.timer);
      this.pending.delete(frame.seqId);
      p.resolve(frame);
    }
  }

  async generate(tokens, maxNew, opts = {}) {
    const D = CFG.hiddenSize;
    if (this.stage.end > this.stage.start || this.stagesTail.length > 0) {
      // prefill
      let tokRow = await this.forward(this.embedTokens(tokens), tokens.length, 0);
      let next = await this.sampleLast(tokRow);
      this.onToken(next);
      for (let i = 1; i < maxNew; i++) {
        if (next === CFG.eosTokenId) break;
        tokRow = await this.forward(this.embedTokens([next]), 1, tokens.length + i - 1);
        next = await this.sampleLast(tokRow);
        this.onToken(next);
      }
      return next;
    }
    // single-node
    let tokRow = await this.forward(this.embedTokens(tokens), tokens.length, 0);
    let next = await this.sampleLast(tokRow);
    this.onToken(next);
    for (let i = 1; i < maxNew; i++) {
      if (next === CFG.eosTokenId) break;
      tokRow = await this.forward(this.embedTokens([next]), 1, tokens.length + i - 1);
      next = await this.sampleLast(tokRow);
      this.onToken(next);
    }
    return next;
  }
}
