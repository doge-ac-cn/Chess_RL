// Chinese Chess (Xiangqi) rules engine — JS port of Chess_RL/xiangqi/xiangqi/board.py.
// Board: 9 cols x 10 rows, row 0 = black back rank, row 9 = red back rank.
// Cells: 0 empty; red K1 A2 B3 N4 R5 C6 P7; black 11..17. idx = r*9 + c.
"use strict";
const XQ = (() => {
  const COLS = 9, ROWS = 10;
  const EMPTY = 0, K = 1, A = 2, B = 3, N = 4, R = 5, C = 6, P = 7;
  const RED = 1, BLACK = 2;
  const IDX = (r, c) => r * COLS + c;
  const TYPE = (p) => (p < 10 ? p : p - 10);
  const SIDE = (p) => (p < 10 ? RED : BLACK);
  const other = (s) => (s === RED ? BLACK : RED);
  const PALACE = {
    1: new Set([].concat(...[7, 8, 9].map(r => [3, 4, 5].map(c => IDX(r, c))))),
    2: new Set([].concat(...[0, 1, 2].map(r => [3, 4, 5].map(c => IDX(r, c))))),
  };
  const START = [
    "rnbakabnr", ".........", ".c.....c.", "p.p.p.p.p", ".........",
    ".........", "P.P.P.P.P", ".C.....C.", ".........", "RNBAKABNR",
  ];
  const CHAR = { K: 1, A: 2, B: 3, N: 4, R: 5, C: 6, P: 7,
                 k: 11, a: 12, b: 13, n: 14, r: 15, c: 16, p: 17 };
  const GLYPH = { 1: "帥", 2: "仕", 3: "相", 4: "馬", 5: "車", 6: "炮", 7: "兵",
                  11: "將", 12: "士", 13: "象", 14: "馬", 15: "車", 16: "砲", 17: "卒" };

  function newBoard() {
    const cells = new Int8Array(90);
    START.forEach((row, r) => {
      for (let c = 0; c < COLS; c++) {
        const ch = row[c];
        if (CHAR[ch]) cells[IDX(r, c)] = CHAR[ch];
      }
    });
    return { cells, toMove: RED, last: -1, history: [] };
  }

  function legalMoves(b) {
    const side = b.toMove;
    const out = [];
    for (let f = 0; f < 90; f++) {
      const p = b.cells[f];
      if (p !== EMPTY && SIDE(p) === side) {
        for (const t of pseudoMoves(b, f, p)) {
          const captured = b.cells[t];
          b.cells[t] = b.cells[f]; b.cells[f] = EMPTY;
          const ok = !inCheck(b, side);
          b.cells[f] = b.cells[t]; b.cells[t] = captured;
          if (ok) out.push([f, t]);
        }
      }
    }
    return out;
  }

  function inCheck(b, side) {
    const target = side === RED ? K : 11;
    let k = -1;
    for (let i = 0; i < 90; i++) if (b.cells[i] === target) { k = i; break; }
    if (k < 0) return true;
    const kr = (k / COLS) | 0, kc = k % COLS;
    // flying general
    const ek = side === RED ? 11 : K;
    let ekr = -1, ekc = -1;
    for (let i = 0; i < 90; i++) if (b.cells[i] === ek) { ekr = (i / COLS) | 0; ekc = i % COLS; break; }
    if (ekc === kc) {
      let facing = true;
      for (let r = Math.min(ekr, kr) + 1; r < Math.max(ekr, kr); r++)
        if (b.cells[IDX(r, kc)] !== EMPTY) { facing = false; break; }
      if (facing) return true;
    }
    for (let i = 0; i < 90; i++) {
      const p = b.cells[i];
      if (p !== EMPTY && SIDE(p) !== side && attacks(b, i, k, p)) return true;
    }
    return false;
  }

  function attacks(b, src, dst, piece) {
    const t = TYPE(piece);
    if (t === K) {
      const [sr, sc] = [(src / COLS) | 0, src % COLS];
      const [dr, dc] = [(dst / COLS) | 0, dst % COLS];
      return Math.max(Math.abs(dr - sr), Math.abs(dc - sc)) === 1;
    }
    if (t === N || t === A || t === B || t === P) return pseudoMoves(b, src, piece).includes(dst);
    // R: slide; C: screen capture
    const [sr, sc] = [(src / COLS) | 0, src % COLS];
    const captureOnly = t === C;
    for (const [dr, dc] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      let rr = sr + dr, cc = sc + dc, jumped = false;
      while (rr >= 0 && rr < 10 && cc >= 0 && cc < 9) {
        const i = IDX(rr, cc), v = b.cells[i];
        if (t === R) {
          if (v === EMPTY) { /* continue */ }
          else { if (i === dst) return true; break; }
        } else {
          if (!jumped) {
            if (v !== EMPTY) jumped = true;
          } else if (v !== EMPTY) { if (i === dst) return true; break; }
        }
        rr += dr; cc += dc;
      }
    }
    return false;
  }

  function pseudoMoves(b, src, piece) {
    const sr = (src / COLS) | 0, sc = src % COLS;
    const side = SIDE(piece), t = TYPE(piece);
    const out = [];
    const add = (r, c) => {
      if (r < 0 || r >= 10 || c < 0 || c >= COLS) return;
      const v = b.cells[IDX(r, c)];
      if (v === EMPTY || SIDE(v) !== side) out.push(IDX(r, c));
    };
    if (t === K) {
      for (const [dr, dc] of [[1,0],[-1,0],[0,1],[0,-1]]) {
        const r = sr + dr, c = sc + dc;
        if (PALACE[side].has(IDX(r, c))) add(r, c);
      }
    } else if (t === A) {
      for (const [dr, dc] of [[1,1],[1,-1],[-1,1],[-1,-1]]) {
        const r = sr + dr, c = sc + dc;
        if (PALACE[side].has(IDX(r, c))) add(r, c);
      }
    } else if (t === B) {
      for (const [dr, dc] of [[2,2],[2,-2],[-2,2],[-2,-2]]) {
        const r = sr + dr, c = sc + dc;
        if (r < 0 || r >= 10 || c < 0 || c >= COLS) continue;
        if (side === RED && r < 5) continue;
        if (side === BLACK && r > 4) continue;
        if (b.cells[IDX(sr + dr/2, sc + dc/2)] !== EMPTY) continue;
        add(r, c);
      }
    } else if (t === N) {
      for (const [dr, dc, lr, lc] of [[2,1,1,0],[2,-1,1,0],[-2,1,-1,0],[-2,-1,-1,0],
                                      [1,2,0,1],[-1,2,0,1],[1,-2,0,-1],[-1,-2,0,-1]]) {
        const r = sr + dr, c = sc + dc;
        if (r < 0 || r >= 10 || c < 0 || c >= COLS) continue;
        if (b.cells[IDX(sr + lr, sc + lc)] !== EMPTY) continue;
        add(r, c);
      }
    } else if (t === R || t === C) {
      for (const [dr, dc] of [[1,0],[-1,0],[0,1],[0,-1]]) {
        let rr = sr + dr, cc = sc + dc, jumped = false;
        const dbg = (src === 64 && typeof window !== "undefined");
        while (rr >= 0 && rr < 10 && cc >= 0 && cc < COLS) {
          const v = b.cells[IDX(rr, cc)];
          if (dbg && t === C) console.log("SCAN", rr, cc, "v=", v, "jumped=", jumped);
          if (t === R) {
            if (v === EMPTY) out.push(IDX(rr, cc));
            else { if (SIDE(v) !== side) out.push(IDX(rr, cc)); break; }
          } else {
            if (!jumped) {
              if (v !== EMPTY) jumped = true;
              else out.push(IDX(rr, cc));
            } else if (v !== EMPTY) {
              if (SIDE(v) !== side) out.push(IDX(rr, cc));
              break;
            }
          }
          rr += dr; cc += dc;
        }
      }
    } else if (t === P) {
      const fwd = side === RED ? -1 : 1;
      add(sr + fwd, sc);
      const crossed = side === RED ? sr <= 4 : sr >= 5;
      if (crossed) { add(sr, sc - 1); add(sr, sc + 1); }
    }
    return out;
  }

  function status(b) {
    if (legalMoves(b).length === 0) return other(b.toMove); // mate or stalemate: mover loses
    return 0;
  }

  // is `dst` attacked by any piece of `side`?
  function squareAttacked(b, dst, side) {
    for (let i = 0; i < 90; i++) {
      const p = b.cells[i];
      if (p !== EMPTY && SIDE(p) === side && pseudoMoves(b, i, p).includes(dst)) return true;
    }
    return false;
  }

  return { COLS, ROWS, EMPTY, RED, BLACK, IDX, TYPE, SIDE, other, GLYPH,
           newBoard, legalMoves, inCheck, status, squareAttacked, pseudoMoves };
})();
