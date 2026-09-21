#!/usr/bin/env node
// CPU reference for the embed path: verify what embed_tokens rows actually
// contain using the same chunk logic + bf16 decode the browser uses.
const FILE_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors";

async function main() {
  const headerLenB = new Uint8Array(8);
  const r0 = await fetch(FILE_URL, { headers: { Range: "bytes=0-7" } });
  const b = new Uint8Array(await r0.arrayBuffer());
  const dv = new DataView(b.buffer);
  const headerLen = Number(dv.getBigUint64(0, true));
  const headerB = new Uint8Array(
    (await (await fetch(FILE_URL, { headers: { Range: `bytes=8-${8 + headerLen - 1}` } })).arrayBuffer())
  );
  const JSONWS = JSON.parse(new TextDecoder().decode(headerB));

  const dataStart = 8 + headerLen;
  delete JSONWS.__metadata__;
  for (const t of Object.values(JSONWS)) t.data_offsets = [t.data_offsets[0] + dataStart, t.data_offsets[1] + dataStart];

  const shout = JSONWS["model.embed_tokens.weight"];
  console.log("shape", shout.shape, "dtype", shout.dtype, "offsets", ...shout.data_offsets);

  // token ids for "How are you?" from quick tokenizer assumption — instead
  // just dump rows 0..2 + row for token 9707 ("Hello")
  const tokens = [0, 1, 2, 9707];
  const rows = [];
  for (const t of tokens) rows.push(t);

  let lo = Math.min(...rows) * 1536 * 2 + shout.data_offsets[0];
  let hi = (Math.max(...rows) + 1) * 1536 * 2 + shout.data_offsets[0];
  const res = await fetch(FILE_URL, { headers: { Range: `bytes=${lo}-${hi - 1}` } });
  const buf = new Uint8Array(await res.arrayBuffer());

  function bf16(i) {
    const off = i * 2 + 0;
    const bits = buf[off + 1] << 8 | buf[off];
    const f32bits = bits << 16; // bf16 top bits in f32
    return new Float32Array(new Uint32Array([f32bits]).buffer)[0];
  }

  for (const t of tokens) {
    let mn = Infinity, mx = -Infinity, nnz = 0, sum = 0;
    for (let i = 0; i < 1536; i++) {
      const v = bf16(t * 1536 + i + ((shout.data_offsets[0] - lo) >> 1));
      mn = Math.min(mn, v); mx = Math.max(mx, v); sum += v; if (v !== 0) nnz++;
    }
    console.log(`token ${t}: mean=${(sum / 1536).toFixed(5)} min=${mn.toFixed(5)} max=${mx.toFixed(5)} nnz=${nnz}`);
  }
  console.log("sample row 9707 first 6:", Array.from({ length: 6 }).map((_, j) => bf16(9707 * 1536 + j + ((shout.data_offsets[0] - lo) >> 1))));
}
main().catch(e => { console.error(e); process.exit(1); });
