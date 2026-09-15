// Web Worker: hosts the ONNX session + AI engine off the UI thread.
"use strict";
importScripts("vendor/ort.min.js", "engine.js", "ai.js");

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      await initSession(msg.modelPath);
      self.postMessage({ type: "ready" });
    } else if (msg.type === "move") {
      const b = msg.board; // {cells, toMove, last, history}
      const res = await chooseMove(b, msg.sims, (sim) =>
        self.postMessage({ type: "progress", sim, sims: msg.sims }));
      self.postMessage({ type: "move", id: msg.id, move: res.move, info: res.info });
    } else if (msg.type === "eval") {   // used by hint / selftest
      const b = msg.board;
      const ev = await evaluate(b);
      let best = -1, bp = -1;
      for (const [m, pr] of ev.prior) if (pr > bp) { bp = pr; best = m; }
      self.postMessage({ type: "eval", id: msg.id, move: best, value: ev.value });
    }
  } catch (err) {
    self.postMessage({ type: "error", id: msg.id || null, message: String(err && err.message || err) });
  }
};
