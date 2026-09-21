// rtc.js — WebRTC peer connections + DataChannel, binary frame codec
import { STUN_URLS } from "./config.js";
import { FRAME } from "./constants.js";
export { FRAME };

// Binary frame: 4x Int32 header [msgType, seqId, posStart, nChunks?] + f32 payload
export function pack(msgType, seqId, posStart, f32) {
  const head = new Int32Array([msgType, seqId, posStart, f32.length]);
  const bytes = new Uint8Array(16 + f32.byteLength);
  bytes.set(new Uint8Array(head.buffer), 0);
  bytes.set(new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength), 16);
  return bytes;
}

export function unpack(data) {
  // header = 4x Int32; payload = the rest of the frame as f32
  const nFloats = (data.byteLength - 16) / 4;
  if (nFloats < 0 || (data.byteLength - 16) % 4 !== 0) throw new Error("bad frame size " + data.byteLength);
  const header = new Int32Array(data, 0, 4);
  const f32 = new Float32Array(data, 16, nFloats);
  return { msgType: header[0], seqId: header[1], posStart: header[2], f32 };
}

export class RTC {
  constructor(myId, sendSignal, onDataChannelOpen) {
    this.myId = myId;
    this.sendSignal = sendSignal;
    this.onDataChannelOpen = onDataChannelOpen;
    this.onError = null; // (err) => {}
    this.conns = {}; // peerId -> RTCPeerConnection
    this.channels = {}; // peerId -> DataChannel
    this.queues = {}; // peerId -> [payload], flushed on open
    this.onRawData = null;
    this.onChannelState = null;
  }

  connect(peerId, isRetry = false) {
    if (this.conns[peerId]) return;
    const pc = new RTCPeerConnection({ iceServers: STUN_URLS });
    this.conns[peerId] = pc;
    pc.onconnectionstatechange = () =>
      this.onChannelState?.(peerId, pc.connectionState);
    pc.onicecandidate = (e) => {
      if (e.candidate) this.sendSignal(peerId, { type: "ice", candidate: e.candidate });
    };
    if (isRetry) {
      pc.ondatachannel = (e) => this.bindChannel(peerId, e.channel);
    } else {
      const chan = pc.createDataChannel("dwllm", { ordered: true });
      this.bindChannel(peerId, chan);
    }
    pc.createOffer()
      .then((offer) => pc.setLocalDescription(offer))
      .then(() => this.sendSignal(peerId, { type: "offer", sdp: pc.localDescription.sdp }))
      .catch((e) => this.onError?.("offer-fail: " + e.message));
  }

  async handleSignal(from, msg) {
    try {
      if (msg.type === "offer") {
        if (this.conns[from]) return;
        const pc = new RTCPeerConnection({ iceServers: STUN_URLS });
        this.conns[from] = pc;
        pc.ondatachannel = (e) => this.bindChannel(from, e.channel);
        pc.onicecandidate = (e) => {
          if (e.candidate) this.sendSignal(from, { type: "ice", candidate: e.candidate });
        };
        await pc.setRemoteDescription({ type: "offer", sdp: msg.sdp });
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        this.sendSignal(from, { type: "answer", sdp: pc.localDescription.sdp });
      } else if (msg.type === "answer") {
        const pc = this.conns[from];
        if (!pc) throw new Error("no pc for answer");
        if (pc.signalingState === "stable" || pc.remoteDescription) return;
        await pc.setRemoteDescription({ type: "answer", sdp: msg.sdp });
      } else if (msg.type === "ice") {
        const pc = this.conns[from];
        if (pc) await pc.addIceCandidate(msg.candidate);
      }
    } catch (e) {
      this.onError?.(`${from} signal ${msg.type}: ${e.message || e}`);
    }
  }

  bindChannel(peerId, chan) {
    this.channels[peerId] = chan;
    chan.binaryType = "arraybuffer";
    chan.onopen = () => {
      this.onChannelState?.(peerId, "open");
      this.flush(peerId);
      this.onDataChannelOpen?.(peerId);
    };
    chan.onmessage = (e) =>
      typeof e.data === "string" ? this.onControl?.(peerId, JSON.parse(e.data)) : this.onRawData?.(peerId, unpack(e.data));
    chan.onclose = () => {
      delete this.channels[peerId];
      this.onChannelState?.(peerId, "closed");
      // watchdog: reconnect in 3s
      setTimeout(() => {
        if (!this.channels[peerId]) {
          delete this.conns[peerId];
          this.connect(peerId, true);
        }
      }, 3000);
    };
    this.onChannelState?.(peerId, "connecting");
  }

  queueOrSend(peerId, payload) {
    if (this.channels[peerId]?.readyState === "open") {
      this.channels[peerId].send(payload);
    } else {
      (this.queues[peerId] ??= []).push(payload);
    }
  }

  flush(peerId) {
    const q = this.queues[peerId] ?? [];
    delete this.queues[peerId];
    for (const p of q) this.channels[peerId]?.send(p);
  }

  send(peerId, payload) {
    // strings (control JSON) and binary frames both queue when the channel isn't open
    this.queueOrSend(peerId, payload);
  }

  openPeers() {
    return Object.keys(this.channels).filter(
      (id) => this.channels[id].readyState === "open",
    );
  }
}
