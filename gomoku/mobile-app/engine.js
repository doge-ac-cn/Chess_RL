// Gomoku rules engine — JS port of gomoku/board.py (free-style rules, >=5 wins).
// Board is a plain object: { cells: Int8Array(225), toMove: 1|2, last: idx|-1 }
"use strict";
const N = 15;
const EMPTY = 0, BLACK = 1, WHITE = 2;
const DIRS = [[0, 1], [1, 0], [1, 1], [1, -1]];

function newBoard() {
  return { cells: new Int8Array(N * N), toMove: BLACK, last: -1, history: [] };
}
function idx(r, c) { return r * N + c; }
function other(p) { return 3 - p; }
function legal(b, r, c) {
  return r >= 0 && r < N && c >= 0 && c < N && b.cells[idx(r, c)] === EMPTY;
}
function play(b, r, c) {
  b.cells[idx(r, c)] = b.toMove;
  b.history.push(idx(r, c));
  b.last = idx(r, c);
  b.toMove = other(b.toMove);
}
function undo(b) {
  const i = b.history.pop();
  b.cells[i] = EMPTY;
  b.toMove = other(b.toMove);
  b.last = b.history.length ? b.history[b.history.length - 1] : -1;
}
function cloneBoard(b) {
  return { cells: b.cells.slice(), toMove: b.toMove, last: b.last, history: b.history.slice() };
}
function isEmpty(b) {
  for (let i = 0; i < N * N; i++) if (b.cells[i] !== EMPTY) return false;
  return true;
}
function winnerAfter(b, i) {
  const p = b.cells[i];
  if (p === EMPTY) return EMPTY;
  const r = (i / N) | 0, c = i % N;
  for (const [dr, dc] of DIRS) {
    let cnt = 1;
    for (const s of [1, -1]) {
      let rr = r + dr * s, cc = c + dc * s;
      while (rr >= 0 && rr < N && cc >= 0 && cc < N && b.cells[idx(rr, cc)] === p) {
        cnt++; rr += dr * s; cc += dc * s;
      }
    }
    if (cnt >= 5) return p;
  }
  return EMPTY;
}
function status(b) {
  if (b.last >= 0) {
    const w = winnerAfter(b, b.last);
    if (w) return w;
  }
  for (let i = 0; i < N * N; i++) if (b.cells[i] === EMPTY) return 0;
  return -1; // draw
}
function candidates(b, radius = 2) {
  const out = [];
  if (isEmpty(b)) return [idx(7, 7)];
  const has = [];
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++)
    if (b.cells[idx(r, c)] !== EMPTY) has.push([r, c]);
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    if (b.cells[idx(r, c)] !== EMPTY) continue;
    for (const [sr, sc] of has) {
      const d = Math.max(Math.abs(sr - r), Math.abs(sc - c));
      if (d <= radius) { out.push(idx(r, c)); break; }
    }
  }
  return out;
}
function fivePoints(b, p) {
  const pts = [];
  for (const i of candidates(b, 1)) {
    const r = (i / N) | 0, c = i % N;
    for (const [dr, dc] of DIRS) {
      let cnt = 1, done = false;
      for (const s of [1, -1]) {
        let rr = r + dr * s, cc = c + dc * s;
        while (rr >= 0 && rr < N && cc >= 0 && cc < N && b.cells[idx(rr, cc)] === p) {
          cnt++; rr += dr * s; cc += dc * s;
        }
      }
      if (cnt >= 5) { pts.push(i); done = true; }
      if (done) break;
    }
  }
  return pts;
}
// If p plays (r,c): the set of empty points that would complete a five
// (completion points of fours created THROUGH this move). >=2 => unstoppable.
function fourCompletions(b, r, c, p) {
  const pts = new Set();
  const opp = other(p);
  b.cells[idx(r, c)] = p;
  for (const [dr, dc] of DIRS) {
    for (let k = -4; k <= 0; k++) {
      const sr = r + k * dr, sc = c + k * dc;
      let mine = 0, empty = -1, ok = true;
      for (let t = 0; t < 5; t++) {
        const rr = sr + t * dr, cc = sc + t * dc;
        if (rr < 0 || rr >= N || cc < 0 || cc >= N) { ok = false; break; }
        const v = b.cells[idx(rr, cc)];
        if (v === opp) { ok = false; break; }
        if (v === p) mine++;
        else empty = idx(rr, cc);
      }
      if (ok && mine === 4 && empty >= 0) pts.add(empty);
    }
  }
  b.cells[idx(r, c)] = EMPTY;
  return pts;
}
// indices of the winning run through the last move (for UI highlight)
function winLine(b) {
  if (b.last < 0) return [];
  const p = b.cells[b.last], r = (b.last / N) | 0, c = b.last % N;
  for (const [dr, dc] of DIRS) {
    const run = [b.last];
    for (const s of [1, -1]) {
      let rr = r + dr * s, cc = c + dc * s;
      while (rr >= 0 && rr < N && cc >= 0 && cc < N && b.cells[idx(rr, cc)] === p) {
        run.push(idx(rr, cc)); rr += dr * s; cc += dc * s;
      }
    }
    if (run.length >= 5) return run;
  }
  return [];
}
// (4,15,15) planes: my stones, opp stones, last move, black-to-move flag
function encode(b) {
  const me = b.toMove, opp = other(me);
  const x = new Float32Array(4 * N * N);
  for (let i = 0; i < N * N; i++) {
    const v = b.cells[i];
    if (v === me) x[i] = 1;
    else if (v === opp) x[N * N + i] = 1;
  }
  if (b.last >= 0) x[2 * N * N + b.last] = 1;
  if (me === BLACK) { for (let i = 3 * N * N; i < 4 * N * N; i++) x[i] = 1; }
  return x;
}
