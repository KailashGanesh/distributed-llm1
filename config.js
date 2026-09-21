// config.js — shared constants
export const STUN_URLS = [{ urls: "stun:stun.l.google.com:19302" }, { urls: "stun:stun.cloudflare.com:3478" }];
export const ROOM = new URLSearchParams(location.search).get("room") || "demo";
