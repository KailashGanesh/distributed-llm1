// kernels.js — WebGPU compute kernels for a Qwen2-style decoder stage
// All weight matrices are stored as u32 storage buffers holding (bf16bits << 16),
// decoded in-shader by bitcast — no shader-f16 requirement.

export async function initKernels(device) {
  device.lost?.then((info) => console.error("[gpu] DEVICE LOST:", info.reason, info.message));
  const cache = {};
  const makePipe = (code, key) =>
    (cache[key] ??= (() => {
      const mod = device.createShaderModule({ code });
      mod.getCompilationInfo?.().then((info) => {
        for (const m of info.messages) {
          if (m.type === "error") console.error(`[shader-error:${key}] line ${m.lineNum}: ${m.message}`);
        }
      });
      return device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
    })());

  const WLOAD = `
    fn wload(buf: ptr<storage, array<u32>, read>, idx: u32) -> f32 {
      return bitcast<f32>((*buf)[idx] << 16u);
    }
  `;

  // out[t, j] = dot(X[t,:], W[j,:])   W row-major [Dout, Din] bf16; X [N, Din] f32
  const MATMUL = (n, din, dout) => `
    ${WLOAD}
    @group(0) @binding(0) var<storage, read> X: array<f32>;
    @group(0) @binding(1) var<storage, read> W: array<u32>;
    @group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) g: vec3<u32>) {
      let j = g.x; let t = g.y;
      if (j >= ${dout}u || t >= ${n}u) { return; }
      var acc = 0.0;
      for (var k = 0u; k < ${din}u; k = k + 4u) {
        acc = acc + X[t * ${din}u + k]       * wload(&W, j * ${din}u + k);
        acc = acc + X[t * ${din}u + k + 1u] * wload(&W, j * ${din}u + k + 1u);
        acc = acc + X[t * ${din}u + k + 2u] * wload(&W, j * ${din}u + k + 2u);
        acc = acc + X[t * ${din}u + k + 3u] * wload(&W, j * ${din}u + k + 3u);
      }
      OUT[t * ${dout}u + j] = acc;
    }
  `;

  const RMSNORM = (d) => `
    @group(0) @binding(0) var<storage, read> X: array<f32>;
    @group(0) @binding(1) var<storage, read> G: array<u32>;   // bf16 weight bits
    @group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) g: vec3<u32>) {
      let t = g.x;
      var ss = 0.0;
      for (var i = 0u; i < ${d}u; i = i + 1u) {
        let v = X[t * ${d}u + i];
        ss = ss + v * v;
      }
      let scale = 1.0 / sqrt(ss / ${d}.0 + 1e-6);
      for (var i = 0u; i < ${d}u; i = i + 1u) {
        OUT[t * ${d}u + i] = X[t * ${d}u + i] * scale * bitcast<f32>(G[i] << 16u);
      }
    }
  `;

  // rope with half-split pairs: thread (t, h, halfIdx); numHeads = heads in THIS tensor
  const ROPE = (numHeads) => `
    @group(0) @binding(0) var<storage, read_write> X: array<f32>;
    @group(0) @binding(1) var<uniform> pos0: u32;
    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) g: vec3<u32>) {
      let t = g.x; let h = g.y; let half = g.z;
      let pos = f32(pos0 + t);
      let rowLen = ${numHeads}u * 128u;
      let i0 = t * rowLen + h * 128u + half;
      let i1 = i0 + 64u;
      let x1 = X[i0]; let x2 = X[i1];
      let freq = exp2(-2.0 * f32(half) / 128.0 * log2(1000000.0));
      let ang = pos * freq;
      let c = cos(ang); let s = sin(ang);
      X[i0] = x1 * c - x2 * s;
      X[i1] = x1 * s + x2 * c;
    }
  `;

  // causal attention over KV cache; q heads share kv head via h / (NQ/KVH)
  const ATTENTION = (nq, kvh, hd) => `
    @group(0) @binding(0) var<storage, read> Q: array<f32>;         // [n, nq*hd]
    @group(0) @binding(1) var<storage, read> KCACHE: array<f32>;    // [maxPos, kvh*hd] f32
    @group(0) @binding(2) var<storage, read> VCACHE: array<f32>;
    @group(0) @binding(3) var<storage, read_write> OUT: array<f32>; // [n, nq*hd]
    @group(0) @binding(4) var<uniform> params: vec2<u32>;           // [T0_totalPrev, n]
    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) g: vec3<u32>) {
      let t = g.x; let h = g.y;
      let T0 = params.x; let n = params.y;
      if (t >= n || h >= ${nq}u) { return; }
      let kvhIdx = h / (${nq}u / ${kvh}u);
      let Tend = T0 + t + 1u;
      var maxs = -3.0e30; var sume = 0.0;
      for (var p = 0u; p < Tend; p = p + 1u) {
        var s = 0.0;
        let kbase = p * ${kvh}u * ${hd}u + kvhIdx * ${hd}u;
        for (var e = 0u; e < ${hd}u; e = e + 1u) {
          s = s + Q[t * ${nq}u * ${hd}u + h * ${hd}u + e] * KCACHE[kbase + e];
        }
        s = s * ${1 / Math.sqrt(hd)}; // 1/sqrt(headDim)
        if (s > maxs) {
          if (sume > 0.0) { sume = sume * exp(maxs - s); }
          maxs = s;
        }
        sume = sume + exp(s - maxs);
      }
      var acc: array<f32, ${hd}>;
      for (var e = 0u; e < ${hd}u; e = e + 1u) { acc[e] = 0.0; }
      for (var p = 0u; p < Tend; p = p + 1u) {
        var s = 0.0;
        let kbase = p * ${kvh}u * ${hd}u + kvhIdx * ${hd}u;
        for (var e = 0u; e < ${hd}u; e = e + 1u) {
          s = s + Q[t * ${nq}u * ${hd}u + h * ${hd}u + e] * KCACHE[kbase + e];
        }
        s = s * ${1 / Math.sqrt(hd)};
        let w = exp(s - maxs) / sume;
        let vbase = p * ${kvh}u * ${hd}u + kvhIdx * ${hd}u;
        for (var e = 0u; e < ${hd}u; e = e + 1u) {
          acc[e] = acc[e] + VCACHE[vbase + e] * w;
        }
      }
      for (var e = 0u; e < ${hd}u; e = e + 1u) {
        OUT[t * ${nq}u * ${hd}u + h * ${hd}u + e] = acc[e];
      }
    }
  `;

  const SILUMUL = (total) => `
    @group(0) @binding(0) var<storage, read> A: array<f32>;
    @group(0) @binding(1) var<storage, read> B: array<f32>;
    @group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) g: vec3<u32>) {
      let i = g.x;
      if (i >= ${total}u) { return; }
      let a = A[i];
      OUT[i] = a / (1.0 + exp(-a)) * B[i];
    }
  `;

  // OUT[i] = A[i] + B[i % dout]   (broadcast row bias, B = bf16 bits)
  const ADDBIAS = (total, dout) => `
    ${WLOAD}
    @group(0) @binding(0) var<storage, read> A: array<f32>;
    @group(0) @binding(1) var<storage, read> B: array<u32>;
    @group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) g: vec3<u32>) {
      let i = g.x;
      if (i >= ${total}u) { return; }
      OUT[i] = A[i] + wload(&B, i % ${dout}u);
    }
  `;

  const ADD = `
    @group(0) @binding(0) var<storage, read> A: array<f32>;
    @group(0) @binding(1) var<storage, read> B: array<f32>;
    @group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) g: vec3<u32>) {
      let i = g.x;
      // caller balances workgroups correctly; reading past end is UB-free in practice
      OUT[i] = A[i] + B[i];
    }
  `;

  const D = 1536, KVH = 2, NQ = 12, HD = 128;

  const rmsPipe = makePipe(RMSNORM(D), "rms");
  const attnPipe = makePipe(ATTENTION(NQ, KVH, HD), "attn");
  const addPipe = makePipe(ADD, "add");

  const ctx = { device, D, KVH, NQ, HD };

  ctx.loadWeightBits = (u32arr, label) => {
    if (!u32arr || !u32arr.byteLength) throw new Error("loadWeightBits: empty weights");
    const buf = device.createBuffer({ size: u32arr.byteLength, label: label ?? "weights", usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, u32arr);
    return buf;
  };

  ctx.createF32 = (numFloats, label) => {
    if (!numFloats || numFloats <= 0) throw new Error(`createF32: invalid size ${numFloats}`);
    return device.createBuffer({ size: numFloats * 4, label, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  };

  ctx.writeF32 = (buf, f32) => device.queue.writeBuffer(buf, 0, f32.buffer, f32.byteOffset, f32.byteLength);

  ctx.readF32 = async (buf, numFloats, label = "") => {
    if (!numFloats || numFloats <= 0) throw new Error(`readF32: invalid size ${numFloats}`);
    const rb = device.createBuffer({ size: numFloats * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, rb, 0, numFloats * 4);
    device.queue.submit([enc.finish()]);
    await Promise.race([
      rb.mapAsync(GPUMapMode.READ),
      new Promise((_, rej) => setTimeout(
        () => rej(new Error(`readF32 timeout (${label}, ${numFloats} floats)` + (device.lost ? " DEVICE LOST" : ""))), 
        15000,
      )),
    ]);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap(); rb.destroy();
    return out;
  };

  const run1 = (pipe, entries, dispatches) => {
    const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(...dispatches); pass.end();
    device.queue.submit([enc.finish()]);
  };

  ctx.matmul = (xBuf, wBuf, n, din, dout) => {
    const out = ctx.createF32(n * dout, `mm_out_${n}x${dout}`);
    const pipe = makePipe(MATMUL(n, din, dout), `mm_${n}_${din}_${dout}`);
    run1(pipe, [
      { binding: 0, resource: { buffer: xBuf } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: out } },
    ], [Math.ceil(dout / 64), n]);
    return out;
  };

  ctx.rmsnorm = (xBuf, gBuf, n) => {
    const out = ctx.createF32(n * D, `rms_out_${n}`);
    run1(rmsPipe, [
      { binding: 0, resource: { buffer: xBuf } },
      { binding: 1, resource: { buffer: gBuf } },
      { binding: 2, resource: { buffer: out } },
    ], [n]);
    return out;
  };

  const ropeWith = (numHeads, pipe) => (xBuf, posStart, n) => {
    if (!n || n <= 0) throw new Error(`rope: invalid n=${n}`);
    const uparams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uparams, 0, new Uint32Array([posStart, 0, 0, 0]));
    run1(pipe, [
      { binding: 0, resource: { buffer: xBuf } },
      { binding: 1, resource: { buffer: uparams } },
    ], [n, numHeads, 64]);
    return xBuf;
  };
  ctx.ropeQ = ropeWith(NQ, makePipe(ROPE(NQ), "ropeq"));
  ctx.ropeK = ropeWith(KVH, makePipe(ROPE(KVH), "ropek"));

  ctx.attention = (qBuf, kCacheBuf, vCacheBuf, T0, n) => {
    const out = ctx.createF32(n * NQ * HD, `attn_out_${n}`);
    const uparams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uparams, 0, new Uint32Array([T0, n, 0, 0]));
    run1(attnPipe, [
      { binding: 0, resource: { buffer: qBuf } },
      { binding: 1, resource: { buffer: kCacheBuf } },
      { binding: 2, resource: { buffer: vCacheBuf } },
      { binding: 3, resource: { buffer: out } },
      { binding: 4, resource: { buffer: uparams } },
    ], [n, NQ]);
    return out;
  };

  ctx.siluMul = (aBuf, bBuf, total) => {
    const out = ctx.createF32(total, `silu_out_${total}`);
    const pipe = makePipe(SILUMUL(total), `silu_${total}`);
    run1(pipe, [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: out } },
    ], [Math.ceil(total / 64)]);
    return out;
  };

  ctx.add = (aBuf, bBuf, numFloats) => {
    const out = ctx.createF32(numFloats, `add_out_${numFloats}`);
    run1(addPipe, [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: out } },
    ], [Math.ceil(numFloats / 64)]);
    return out;
  };

  ctx.addBias = (aBuf, biasBuf, n, dout) => {
    const total = n * dout;
    const out = ctx.createF32(total, `bias_out_${total}`);
    const pipe = makePipe(ADDBIAS(total, dout), `bias_${total}_${dout}`);
    run1(pipe, [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: biasBuf } },
      { binding: 2, resource: { buffer: out } },
    ], [Math.ceil(total / 64)]);
    return out;
  };

  return ctx;
}
