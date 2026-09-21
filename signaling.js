// signaling.js — PubNub: presence roster, leader election, WebRTC SDP/ICE relay
// PubNub carries ONLY tiny JSON. All model traffic goes over WebRTC DataChannels.

export const PUBKEY = "demo"; // PubNub open sandbox keys (no auth; fine for signaling)
export const SUBKEY = "demo";

export function uuid() {
  return "pii_" + Math.random().toString(36).slice(2, 8);
}

export class Signaling {
  constructor(room, peerId, onMsg, getProfile) {
    this.room = room;
    this.myId = peerId;
    this.onMsg = onMsg;
    this.getProfile = getProfile;
    this.channel = "dwllm-" + room;
    this.peers = {}; // id -> {ts, profile}
    this.pn = new PubNub({ publishKey: PUBKEY, subscribeKey: SUBKEY, uuid: peerId });
    this.pn.addListener({
      message: (m) => {
        const msg = m.message;
        if (!msg || (msg.from && msg.from === this.myId)) return;
        // stale peer pruning
        if (msg.type === "hello") {
          this.peers[msg.from] = { ts: Date.now(), profile: msg.profile };
        }
        this.debug?.("[sig] rx " + JSON.stringify(msg).slice(0, 120));
        this.onMsg(msg);
      },
      status: (s) => this.debug?.("[sig] status " + s.category),
    });
    this.pn.subscribe({ channels: [this.channel] });
    this.helloTimer = setInterval(() => this.hello(), 10000);
    setTimeout(() => this.hello(), 300);
  }

  hello() {
    for (const id of Object.keys(this.peers)) {
      if (Date.now() - this.peers[id].ts > 40000) delete this.peers[id];
    }
    this.publish({ type: "hello", profile: this.getProfile?.() ?? null });
  }

  publish(obj) {
    this.pn
      .publish({ channel: this.channel, message: { ...obj, from: this.myId || undefined } })
      .catch?.((e) => this.debug?.("[sig] publish FAIL " + JSON.stringify(e)));
  }

  sendTo(id, obj) {
    this.publish({ ...obj, to: id });
  }

  isLeader() {
    const ids = Object.keys(this.peers)
      .concat([this.myId])
      .sort();
    return ids[0] === this.myId;
  }

  leave() {
    clearInterval(this.helloTimer);
    this.pn.unsubscribe({ channels: [this.channel] });
  }
}
