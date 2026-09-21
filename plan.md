# distributed-web-llm

Browser-native, pipeline-parallel LLM inference across multiple browser tabs/devices.
PubNub is used ONLY for peer discovery + WebRTC signaling. All model data (activations)
flows over direct P2P WebRTC DataChannels.

Stack: plain HTML + vanilla JS + WebGPU (WGSL). No CSS, no bundler, no npm.

## Architecture

```
Tab A (Leader)              Tab B (Stage 1)            Tab C (Stage 2)
embed tokens + layers 0-1 → hidden states → layers 2-3 → hidden states → layers 4-5
        ▲                                                              │
        └────────────── last-token hidden (backward DataChannel) ───────┘
        Leader: final RMSNorm + lm_head + sampling → next token
```

- Model: Qwen/Qwen2.5-1.5B-Instruct (28 layers, hidden 1536, bf16 safetensors from HuggingFace).
  Weights are streamed per-tensor over HTTP Range requests — each tab downloads
  ONLY the layer slice it owns (~1.2 GB bf16 → stored on GPU as packed u32 f16-style bits).
- Tokens flow forward: tab embeds → each stage runs its layers → last stage returns
  final hidden → leader runs norm + lm_head + sampling.
- Every stage keeps its own KV cache for the layers it owns — no KV is ever shipped.
- Decode is lock-step: leader embeds token → chain of DataChannel hops → logits back →
  leader immediately embeds sampled token and starts next hop cycle. No broadcast needed.

## Files

- `index.html` — UI: connection panel (peer roster), chat, log
- `script.js` — app glue / UI
- `signaling.js` — PubNub wrapper: join roster, leader election, SDP/ICE relay
- `rtc.js` — DataChannel management, binary frame codec
- `weights.js` — Safetensors HTTP-range reader + layer-slice fetcher + Qwen config
- `kernels.js` — WebGPU (WGSL) kernels: matmul, rmsnorm, rope, GQA attention, silu, add
- `pipeline.js` — roster → layer split; stage forward logic; leader/decoder logic

## PubNub usage (free-tier friendly)

- One channel: `dwllm-<room>`
- `hello` (peer announces id + device profile) → every 10s keepalive
- `roster` (leader announces assignment)
- `offer` / `answer` / `ice` (WebRTC signaling, only during connect)
- After DataChannels open, PubNub is silent.

## Leader election & split

- peerId = random 8-digit. Lowest id = leader.
- Leader estimates per-peer capacity: `navigator.deviceMemory`, GPU adapter info,
  `deviceMaximumStorageBufferBindingSize`. Split ≈ proportional to (deviceMemory × has-WebGPU).
- Leader owns layers [0, a). With 2 peers: split layers evenly; leader additionally
  owns embed + final norm + lm_head.

## Decode protocol (DataChannel binary frames)

Header = 4×Int32 `[msgType, seqId, posStart, nTokens]` then Float32Array payload.

- `T_HIDDEN_FROM箍` (1): payload hidden states [nTokens × 1536] → hop to next stage
- `T_FINAL` (2): last-stage product, back to leader → logits
- Control: JSON strings `{type:'start', ...}` etc.

## Build order

1. [x] plan.md
2. signaling + rtc: two tabs/devices see each other in roster, channels open
3. weights.js: fetch header + tensor map, load a layer range
4. kernels: forward pass runs single-node (all layers local) as correctness baseline
5. pipeline: split version across DataChannels
6. UI polish, tok/s stats

## Run

1. `python3 -m http.server 8000` in this folder
2. Open `http://localhost:8000` in 2+ Chrome tabs (WebGPU required)
3. Put PubNub keys into `signaling.js` PUBKEY/SUBKEY
4. Everyone uses same `?room=` or default room

## Known limitations

- CPU fallback when no WebGPU: NOT implemented. Chrome desktop required.
- Quality/perf: naive decode, ~1-3 tok/s expected. It's a demo.
- If a peer drops mid-generation, generation aborts; recover on next message.
