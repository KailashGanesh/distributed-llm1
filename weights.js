// weights.js — fetch Qwen2.5-1.5B-Instruct safetensors by HTTP Range, per-tensor
import { FILE_URL } from "./constants.js";

// ---- persistent download cache (survives reloads; every teammate tab benefits) ----
const CACHE_NAME = "dwllm-weights-v3-" + FILE_URL;

async function cacheGet(url) {
  try {
    const c = await caches.open(CACHE_NAME);
    const m = await c.match(url);
    return m ? new Uint8Array(await m.arrayBuffer()) : null;
  } catch { return null; }
}

async function cachePut(url, size, bytes) {
  try {
    const c = await caches.open(CACHE_NAME);
    await c.put(new Request(url), new Response(bytes, {
      headers: { "Content-Type": "application/octet-stream", "X-Size": String(size) },
    }));
  } catch (e) { console.warn("cache put failed", e.message); }
}

// Safetensors header: first 8 bytes = little-endian u64 header length, then JSON
export async function fetchTensorMap(url = FILE_URL) {
  const range = async (start, end) => {
    const resp = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
    return new Uint8Array(await resp.arrayBuffer());
  };
  const lenBytes = await range(0, 7);
  const len = Number(new DataView(lenBytes.buffer).getBigUint64(0, true));
  const headerBytes = await range(8, 8 + len - 1);
  const header = JSON.parse(new TextDecoder().decode(headerBytes));
  delete header.__metadata__;
  const dataStart = 8 + len; // data_offsets are relative to the start of the data section
  for (const t of Object.values(header)) {
    t.data_offsets = [t.data_offsets[0] + dataStart, t.data_offsets[1] + dataStart];
  }
  return header; // name -> {dtype, shape, data_offsets:[start,end]} relative to file start
}

const DBG = (...a) => console.log("[weights]", ...a);

// fetch byte ranges for a list of tensor names, merging contiguous-ish ranges.
// onProgress(float 0..1) called during each merged-range stream download.
export async function fetchTensors(tensorMap, names, onProgress) {
  const sorted = names
    .filter((n) => tensorMap[n])
    .map((n) => ({ n, start: tensorMap[n].data_offsets[0], end: tensorMap[n].data_offsets[1] }))
    .sort((a, b) => a.start - b.start);
  DBG(`fetchTensors: ${names.length} requested -> ${sorted.length} matched, ${sorted.reduce((a, t) => a + (t.end - t.start), 0) / 1e6 | 0} MB total`);

  const totalBytes = sorted.reduce((a, t) => a + (t.end - t.start), 0);
  let doneBytes = 0;

  const chunks = [];
  const MAX_CHUNK = 900_000_000; // keep ArrayBuffer-backed fetches under allocation limits
  for (const t of sorted) {
    const last = chunks[chunks.length - 1];
    if (last && last.end - last.start + (t.end - t.start) < MAX_CHUNK && t.start - last.end < 1_500_000) {
      last.end = t.end;
      last.names.push(t.n);
    } else {
      chunks.push({ start: t.start, end: t.end, names: [t.n] });
    }
  }

  const out = {};
  for (const chunk of chunks) {
    // NOTE: must use a QUERY STRING, not #fragment — fragments are stripped by
    // the Request constructor, which would collapse all chunk keys into one.
    const rangeKey = `${FILE_URL}?__range=${chunk.start}-${chunk.end - 1}`;
    const size = chunk.end - chunk.start;

    // cached → instant (counts as 100% progress for this chunk)
    let bytes = await cacheGet(rangeKey);
    let fromCache = !!bytes;
    if (bytes && bytes.length !== size) {
      DBG(`cache entry BAD ${rangeKey}: got ${bytes.length}/${size} — evicting`);
      bytes = null; fromCache = false;
      caches.open(CACHE_NAME).then((c) => c.delete(rangeKey)).catch(() => {});
    }
    if (fromCache) DBG(`cache hit ${rangeKey} (${size / 1e6 | 0} MB)`);
    if (bytes) {
      doneBytes += size;
      if (onProgress) onProgress(Math.min(doneBytes / totalBytes, 1));
    } else {
      DBG(`fetching chunk ${chunk.start}..${chunk.end} (${size / 1e6 | 0} MB)`);
      const t0 = performance.now();
      const resp = await fetch(FILE_URL, {
        headers: { Range: `bytes=${chunk.start}-${chunk.end - 1}` },
      });
      DBG(`HTTP ${resp.status} for ${rangeKey}`);

      // stream with per-byte progress
      if (resp.body && onProgress) {
        const reader = resp.body.getReader();
        const parts = [];
        let got = 0;
        let last = performance.now();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          parts.push(value);
          got += value.length;
          if (performance.now() - last > 200) {
            last = performance.now();
            onProgress((doneBytes + got) / totalBytes);
          }
        }
        bytes = concatBytes(parts);
        DBG(`streamed ${got} bytes for range ${chunk.start}..${chunk.end} (expect ${size})`);
        onProgress((doneBytes + got) / totalBytes);
      } else {
        bytes = new Uint8Array(await resp.arrayBuffer());
      }
      doneBytes += size;
      if (onProgress) onProgress(Math.min(doneBytes / totalBytes, 1));
      if (bytes.length !== size) throw new Error(`range ${chunk.start}-${chunk.end} incomplete: got ${bytes.length}/${size} bytes`);
      await cachePut(rangeKey, size, bytes);
    }

    for (const n of chunk.names) {
      const off = tensorMap[n].data_offsets[0] - chunk.start;
      const sz = tensorMap[n].data_offsets[1] - tensorMap[n].data_offsets[0];
      const t = bytes.subarray(off, off + sz);
      if (t.length !== sz) throw new Error(`tensor ${n}: expected ${sz} bytes, got ${t.length} (chunk ${chunk.start}..${chunk.end})`);
      out[n] = t;
    }
  }
  DBG("fetchTensors done:", Object.keys(out).length, "tensors");
  return out; // name -> bf16 byte pairs
}

function concatBytes(parts) {
  const len = parts.reduce((a, p) => a + p.length, 0);
  const out = new Uint8Array(len);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

// bf16 bytes -> Uint32Array of RAW bf16 bits (shaders shift <<16 in-shader)
export function bf16BytesToF32Bits(bytes) {
  const n = bytes.length / 2;
  const out = new Uint32Array(n);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < n; i++) out[i] = view.getUint16(i * 2, true);
  return out;
}

// u32 of RAW bf16 bits -> Float32Array
export function bf16BitsToF32(u32, offset = 0, length = u32.length) {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) out[i] = reinterpretBits(u32[offset + i] << 16);
  return out;
}

export function reinterpretBits(bits) {
  ab_u32[0] = bits;
  return ab_f32[0];
}
const ab = new ArrayBuffer(4);
const ab_u32 = new Uint32Array(ab), ab_f32 = new Float32Array(ab);

// tensors needed by a stage owning layers [start, end)
export function tensorsForLayers(start, end, isLeader) {
  const names = [];
  for (let i = start; i < end; i++) {
    const b = `model.layers.${i}.`;
    names.push(
      `${b}input_layernorm.weight`, `${b}post_attention_layernorm.weight`,
      `${b}self_attn.q_proj.weight`, `${b}self_attn.k_proj.weight`,
      `${b}self_attn.v_proj.weight`, `${b}self_attn.o_proj.weight`,
      `${b}self_attn.q_proj.bias`, `${b}self_attn.k_proj.bias`,
      `${b}self_attn.v_proj.bias`,
      `${b}mlp.gate_proj.weight`, `${b}mlp.up_proj.weight`, `${b}mlp.down_proj.weight`,
    );
  }
  if (isLeader) names.push("model.embed_tokens.weight", "model.norm.weight");
  return names;
}
