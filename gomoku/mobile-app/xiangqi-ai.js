// Xiangqi AI: ONNX policy-value net (factorized from/to heads) + light search.
"use strict";
/* global ort, XQ */


// ---------------- teacher heuristic (port of xiangqi/heuristic.py) ----------
const XMAT = { 1: 100000, 2: 120, 3: 120, 4: 420, 5: 900, 6: 450, 7: 90 };

function xType(p) { return p < 10 ? p : p - 10; }
function xSide(p) { return p < 10 ? 1 : 2; }

function xPst(b, i, p) {
  const t = xType(p), side = xSide(p);
  const r = (i / 9) | 0, c = i % 9;
  if (t === 7) {
    const crossed = side === 1 ? r <= 4 : r >= 5;
    const depth = side === 1 ? 9 - r : r;
    return crossed ? 30 : 6 * depth;
  }
  if (t === 4 || t === 6) return 6 * (4 - Math.abs(c - 4));
  return 0;
}

function xMoveScore(b, mv, side, netLogit) {
  const [f, t] = mv;
  const piece = b.cells[f];
  let s = b.cells[t] ? Math.log10(XMAT[xType(b.cells[t])] + 10) : 0;
  s += (xPst(b, t, piece) - xPst(b, f, piece)) / 100;
  s += 0.02 * (netLogit || 0);            // net policy as a small tiebreak
  // tempo: fewer opponent replies is better
  const captured = b.cells[t];
  b.cells[t] = b.cells[f]; b.cells[f] = 0;
  const replies = XQ.legalMoves(b).length;
  b.cells[f] = b.cells[t]; b.cells[t] = captured;
  s -= 2 * replies / 100;
  return s;
}

async function xiangqiChooseMove(b, opts = {}) {
  const { onProgress } = opts;
  const side = b.toMove;
  const legal = XQ.legalMoves(b);
  if (!legal.length) throw new Error("no legal moves");
  const ev = await xEvaluate(b);
  onProgress && onProgress(1);

  // heuristic score for every legal move + net logit as tiebreak
  const netFrom = ev.from, netTo = ev.to;
  const scored = legal.map(([f, t]) => {
    const piece = b.cells[f];
    const captured = b.cells[t];
    let s = captured ? Math.log10(XMAT[xType(captured)] + 10) : 0;
    s += (xPst(b, t, piece) - xPst(b, f, piece)) / 100;
    b.cells[t] = piece; b.cells[f] = 0;
    const replies = XQ.legalMoves(b).length;
    b.cells[f] = piece; b.cells[t] = captured;
    s -= 2 * replies / 100;
    s += 0.02 * (netFrom[f] + netTo[t]) / 4;
    return { mv: [f, t], s };
  });
  onProgress && onProgress(2);
  scored.sort((a, b2) => b2.s - a.s);

  const top = scored.slice(0, Math.min(5, scored.length));
  const mx = top[0].s;
  const weights = top.map(o => Math.exp((o.s - mx) / 0.12));
  const total = weights.reduce((a, w) => a + w, 0);
  let pick = top[0].mv;
  if (opts.temperature !== 0) {
    let acc = Math.random() * total;
    for (let i = 0; i < top.length; i++) {
      acc -= weights[i];
      if (acc <= 0) { pick = top[i].mv; break; }
    }
  }
  return { move: pick, info: { kind: "teacher+policy", value: ev.value } };
}
