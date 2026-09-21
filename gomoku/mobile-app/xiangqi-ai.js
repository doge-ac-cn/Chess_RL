// Xiangqi AI: ONNX policy-value net (factorized from/to heads) + light search.
"use strict";
/* global ort, XQ */

const XCFG = { captureWeight: 300, hangPenalty: 220, temperature: 0.25, topK: 6 };
let _xsession = null;

async function initXiangqiSession(modelPath) {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = "vendor/";
  let lastErr = null;
  for (const eps of [["webgpu", "wasm"], ["wasm"]]) {
    try {
      _xsession = await ort.InferenceSession.create(modelPath, {
        executionProviders: eps, graphOptimizationLevel: "all",
      });
      return eps[0];
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

function xEncode(b) {
  const x = new Float32Array(15 * 10 * 9);
  for (let i = 0; i < 90; i++) {
    const p = b.cells[i];
    if (p === 0) continue;
    const r = (i / 9) | 0, c = i % 9;
    if (p < 10) x[(p - 1) * 90 + r * 9 + c] = 1;
    else x[(6 + p - 10) * 90 + r * 9 + c] = 1;
  }
  if (b.toMove === XQ.RED) {
    for (let i = 13 * 90; i < 14 * 90; i++) x[i] = 1;
  }
  return x;
}

async function xEvaluate(b) {
  const t = new ort.Tensor("float32", xEncode(b), [1, 15, 10, 9]);
  const out = await _xsession.run({ planes: t });
  return {
    from: Array.from(out.from.data[0]),
    to: Array.from(out.to.data[0]),
    value: out.value.data[0],
  };
}

const MATERIAL = { 1: 100000, 2: 120, 3: 120, 4: 420, 5: 900, 6: 450, 7: 90 };

// net-driven move choice with light material/hausse heuristics
async function xiangqiChooseMove(b, opts = {}) {
  const { onProgress } = opts;
  const side = b.toMove;
  const legal = XQ.legalMoves(b);
  if (!legal.length) throw new Error("no legal moves");
  const ev = await xEvaluate(b);
  onProgress && onProgress(1);

  const scored = legal.map(([f, t]) => {
    let s = ev.from[f] + ev.to[t];                       // policy logit score
    const captured = b.cells[t];
    if (captured !== 0) s += Math.log10(MATERIAL[XQ.TYPE(captured)] + 10); // free material
    // hang check: would the moved piece be capturable next turn?
    const piece = b.cells[f];
    const value = MATERIAL[XQ.TYPE(piece)];
    if (value >= 90) {
      b.cells[t] = piece; b.cells[f] = 0;
      const opp = 3 - side;
      const hanging = XQ.squareAttacked(b, t, opp) && !XQ.squareAttacked(b, t, side);
      b.cells[f] = piece; b.cells[t] = captured;
      if (hanging) s -= Math.log10(value + 10) * 1.4;
    }
    return { mv: [f, t], s };
  });
  scored.sort((a, b2) => b2.s - a.s);
  onProgress && onProgress(2);

  // softmax sample among top-K (temperature gives human-friendly variety)
  const top = scored.slice(0, Math.min(XCFG.topK, scored.length));
  const mx = top[0].s;
  const weights = top.map(o => Math.exp((o.s - mx) / XCFG.temperature));
  const total = weights.reduce((a, w) => a + w, 0);
  let pick = top[0].mv;
  if (opts.temperature !== 0) {
    let acc = Math.random() * total;
    for (let i = 0; i < top.length; i++) {
      acc -= weights[i];
      if (acc <= 0) { pick = top[i].mv; break; }
    }
  }
  return { move: pick, info: { kind: "policy+light", value: ev.value } };
}
