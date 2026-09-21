// Xiangqi worker: hosts the ONNX session + rules engine + move chooser.
importScripts("vendor/ort.min.js", "xiangqi-engine.js", "xiangqi-ai.js");

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      const provider = await initXiangqiSession(msg.modelPath);
      self.postMessage({ type: "ready", provider });
    } else if (msg.type === "new") {
      /* stateless: board arrives with each move request */
    } else if (msg.type === "move") {
      const b = { cells: Int8Array.from(msg.board.cells), toMove: msg.board.toMove,
                  last: msg.board.last, history: [] };
      const res = await xiangqiChooseMove(b, { temperature: 0.25 });
      self.postMessage({ type: "move", move: res.move, info: res.info });
    }
  } catch (err) {
    self.postMessage({ type: "error", message: String(err && err.message || err) });
  }
};
