// debug11.js — probe softmax weights inside a clone of the real attention kernel
const out = (s) => { console.log(s); document.getElementById("log").textContent += s + "\n"; };
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
device.onuncapturederror = (e) => out("GPU ERR: " + (e.error?.message || e.error?.code || e));
const NQ = 12, KVH = 2, HD = 128;

function mulberry32(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 14, 61) | 0; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const rnd = mulberry32(5);
const T0 = 3, n = 1;
const q = new Float32Array(NQ * HD); for (let i = 0; i < q.length; i++) q[i] = rnd() * 2 - 1;
const kcache = new Float32Array(4096 * KVH * HD), vcache = new Float32Array(4096 * KVH * HD);
for (let p = 0; p < T0 + n; p++) for (let hv = 0; hv < KVH; hv++) for (let e = 0; e < HD; e++) {
  kcache[p * KVH * HD + hv * HD + e] = rnd() * 2 - 1;
  vcache[p * KVH * HD + hv * HD + e] = rnd() * 2 - 1;
}

// CPU softmax reference for head 0
const scale = 1 / Math.sqrt(HD);
const scores = [];
for (let p = 0; p < T0 + n; p++) {
  let s = 0; for (let e = 0; e < HD; e++) s += q[0 * HD + e] * kcache[p * KVH * HD + 0 * HD + e];
  scores.push(s * scale);
}
const mx = Math.max(...scores);
let sume = 0; const w = scores.map((s) => { const e = Math.exp(s - mx); sume += e; return e; });
const wNorm = w.map((v) => v / sume);
out(`cpu scores: ${scores.map((v) => v.toFixed(3)).join(",")}`);
out(`cpu w:      ${wNorm.map((v) => v.toFixed(3)).join(",")}`);

// GPU: same kernel body, but writes W into first Tend OUT slots and acc for head0 into [8..]
const mod = device.createShaderModule({ code: `
  @group(0) @binding(0) var<storage, read> Q: array<f32>;
  @group(0) @binding(1) var<storage, read> KCACHE: array<f32>;
  @group(0) @binding(2) var<storage, read> VCACHE: array<f32>;
  @group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
  @group(0) @binding(4) var<uniform> params: vec2<u32>;
  @compute @workgroup_size(1)
  fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = g.x; let h = g.y;
    let T0 = params.x; let n = params.y;
    if (t >= n || h >= ${NQ}u) { return; }
    let kvhIdx = h / (${NQ}u / ${KVH}u);
    let Tend = T0 + t + 1u;
    var maxs = -3.0e30; var sume = 0.0;
    for (var p = 0u; p < Tend; p = p + 1u) {
      var s = 0.0;
      let kbase = p * ${KVH}u * ${HD}u + kvhIdx * ${HD}u;
      for (var e = 0u; e < ${HD}u; e = e + 1u) {
        s = s + Q[t * ${NQ}u * ${HD}u + h * ${HD}u + e] * KCACHE[kbase + e];
      }
      s = s * 0.088;
      if (s > maxs) {
        if (sume > 0.0) { sume = sume * exp(maxs - s); }
        maxs = s;
      }
      sume = sume + exp(s - maxs);
      OUT[p] = exp(s - maxs);  // probe: raw weight numerators
    }
    for (var e = 0u; e < ${HD}u; e = e + 1u) {
      OUT[16u + e] = 0.0;
    }
    for (var p = 0u; p < Tend; p = p + 1u) {
      var s = 0.0;
      let kbase = p * ${KVH}u * ${HD}u + kvhIdx * ${HD}u;
      for (var e = 0u; e < ${HD}u; e = e + 1u) {
        s = s + Q[t * ${NQ}u * ${HD}u + h * ${HD}u + e] * KCACHE[kbase + e];
      }
      s = s * 0.088;
      let w = exp(s - maxs) / sume;
      for (var e = 0u; e < ${HD}u; e = e + 1u) {
        OUT[16u + e] = OUT[16u + e] + VCACHE[p * ${KVH}u * ${HD}u + kvhIdx * ${HD}u + e] * w;
      }
    }
  }
`});
const pipe = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
const mkF32 = (a, label) => { const b = device.createBuffer({ size: a.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, a); return b; };
const qB = mkF32(q), kCB = mkF32(kcache), vCB = mkF32(vcache);
const outBuf = device.createBuffer({ size: 144 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const up = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(up, 0, new Uint32Array([T0, n, 0, 0]));
const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
  { binding: 0, resource: { buffer: qB } }, { binding: 1, resource: { buffer: kCB } },
  { binding: 2, resource: { buffer: vCB } }, { binding: 3, resource: { buffer: outBuf } },
  { binding: 4, resource: { buffer: up } }] });
const enc = device.createCommandEncoder();
const pass = enc.beginComputePass();
pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(n, NQ); pass.end();
device.queue.submit([enc.finish()]);
await device.queue.onSubmittedWorkDone();
const rb = device.createBuffer({ size: 144 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const enc2 = device.createCommandEncoder(); enc2.copyBufferToBuffer(outBuf, 0, rb, 0, 144 * 4); device.queue.submit([enc2.finish()]);
await rb.mapAsync(GPUMapMode.READ);
const r = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap();
out(`gpu raw w (unnormalized): ${Array.from(r.slice(0, 8)).map((v) => v.toFixed(4)).join(",")}`);
out(`gpu sume ref (host):      ${w.map((v) => v.toFixed(3)).join(",")} (sum=${sume.toFixed(3)})`);
out(`gpu acc (head0): ${Array.from(r.slice(8, 16)).map((v) => v.toFixed(3)).join(",")}`);
const cpuAcc = new Float32Array(HD);
for (let e = 0; e < HD; e++) { let acc = 0; for (let p = 0; p < T0 + n; p++) acc += vcache[0 * HD + e] * w[p] / sume * (p === 0 ? 1 : 1); }
const refAcc = new Float32Array(HD);
for (let e = 0; e < HD; e++) { let acc = 0; for (let p = 0; p < T0 + n; p++) acc += vcache[p * KVH * HD + e] * w[p] / sume; refAcc[e] = acc; }
out(`cpu acc (head0): ${Array.from(refAcc.slice(0, 8)).map((v) => v.toFixed(3)).join(",")}`);
