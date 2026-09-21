export const FILE_URL =
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors";
export const FRAME = {
  HIDDEN_REQ: 1,  // leader -> stage: run layers, reply full hidden matrix
  FINAL_REQ: 2,   // leader -> last stage: run layers, reply ONLY last token row
  RESP: 3,        // stage -> leader
};

// Qwen2.5-1.5B-Instruct
export const CFG = {
  vocabSize: 151936,
  hiddenSize: 1536,
  numLayers: 28,
  numQHeads: 12,
  numKVHeads: 2,
  headDim: 128,
  intermediateSize: 8960,
  ropeTheta: 1000000.0,
  rmsNormEps: 1e-6,
  tieWordEmbeddings: true,
  eosTokenId: 151645,
  maxPos: 4096,
};
