// Gomoku AI: ONNX policy-value net + PUCT MCTS + forced moves + fork guard + VCF.
// JS port of gomoku/{mcts,players,vcf}.py. Runs inside a Web Worker.
// Strength features: cross-move tree reuse, prior pruning, white reply book,
// WebGPU acceleration with WASM fallback.
"use strict";
/* global ort, newBoard, cloneBoard, play, undo, status, candidates, fivePoints,
   fourCompletions, winnerAfter, encode, N, EMPTY */

const CFG = {
  cPuct: 1.8,
  priorKeep: 16,        // max children per expansion (top-K by prior)
  priorMin: 0.01,       // ...plus any child with prior >= this
  vcfDepth: 14,
  vcfBudgetNodes: 10000,
  vcfBudgetMs: 2500,
};

let _session = null;
let _provider = "";

async function initSession(modelPath) {
  ort.env.wasm.numThreads = 1;   // no cross-origin isolation on static hosting
  ort.env.wasm.wasmPaths = "vendor/";
  const attempts = [["webgpu", "wasm"], ["wasm"]];
  let lastErr = null;
  for (const eps of attempts) {
    try {
      _session = await ort.InferenceSession.create(modelPath, {
        executionProviders: eps,
        graphOptimizationLevel: "all",
      });
      _provider = eps[0];
      return _provider;
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr;
}

// One net forward: {prior: Map(idx -> p) over candidates, value}
async function evaluate(b, restrict) {
  const x = encode(b);
  const input = new ort.Tensor("float32", x, [1, 4, N, N]);
  const out = await _session.run({ planes: input });
  const logits = out.policy.data;
  const value = out.value.data[0];
  let cand = candidates(b, 2);
  if (restrict) cand = cand.filter(i => restrict.has(i));
  let maxLogit = -Infinity;
  for (const i of cand) if (logits[i] > maxLogit) maxLogit = logits[i];
  let sum = 0;
  const p = new Map();
  for (const i of cand) {
    const e = Math.exp(logits[i] - maxLogit);
    p.set(i, e); sum += e;
  }
  for (const i of cand) p.set(i, p.get(i) / sum);
  return { prior: p, value };
}

// ---------------- PUCT MCTS (full tree, port of mcts.py) ----------------

function makeNode(prior = 0) {
  return { prior, N: 0, W: 0, children: null };
}

function expandChildren(node, prior) {
  const items = [...prior.entries()].sort((a, b) => b[1] - a[1]);
  node.children = new Map();
  let kept = 0;
  for (const [m, pr] of items) {
    if (kept < 8 || pr >= CFG.priorMin || kept < CFG.priorKeep) {
      node.children.set(m, makeNode(pr));
      kept++;
    }
  }
}

async function mcts(board, sims, root, onProgress) {
  const rootNode = root || makeNode();
  if (!rootNode.children) {
    const ev = await evaluate(board, null);
    expandChildren(rootNode, ev.prior);
  }

  for (let sim = 0; sim < sims; sim++) {
    const scratch = cloneBoard(board);
    const path = [];
    let node = rootNode;

    while (node.children) {
      let total = 0;
      for (const ch of node.children.values()) total += ch.N;
      const sqrtTotal = Math.sqrt(total) || 1;
      let bestM = null, bestCh = null, bestV = -1e18;
      for (const [m, ch] of node.children) {
        const q = ch.N ? ch.W / ch.N : 0;
        const u = CFG.cPuct * ch.prior * sqrtTotal / (1 + ch.N);
        if (q + u > bestV) { bestV = q + u; bestM = m; bestCh = ch; }
      }
      play(scratch, (bestM / N) | 0, bestM % N);
      path.push(bestCh);
      node = bestCh;
    }

    const st = status(scratch);
    let value;
    if (st === -1) value = 0;
    else if (st) value = -1;
    else {
      const ev = await evaluate(scratch, null);
      expandChildren(node, ev.prior);
      value = ev.value;
    }

    for (let i = path.length - 1; i >= 0; i--) {
      value = -value;
      path[i].W += value;
      path[i].N += 1;
    }
    if (onProgress && (sim & 15) === 15) onProgress(sim + 1);
  }

  let bestM = null, bestN = -1;
  const visits = [];
  for (const [m, ch] of rootNode.children) {
    visits.push([m, ch.N]);
    if (ch.N > bestN) { bestN = ch.N; bestM = m; }
  }
  visits.sort((a, b) => b[1] - a[1]);
  const ch = rootNode.children.get(bestM);
  return { move: bestM, q: ch.N ? ch.W / ch.N : 0, ranked: visits, root: rootNode };
}

// ---------------- VCF prover (port of vcf.py, sound: never false-positives) --

function vcfSearch(board, attacker, depth, ctx) {
  if (ctx.nodes++ > CFG.vcfBudgetNodes || Date.now() > ctx.deadline) throw { abort: true };
  const me = attacker, opp = 3 - me;

  const my5 = fivePoints(board, me);
  if (my5.length) return [my5[0]];
  const opp5 = fivePoints(board, opp);
  if (opp5.length) {
    if (opp5.length > 1 || depth <= 0) return null;
    const tries = [];
    for (const m of opp5) {
      const comps = fourCompletions(board, (m / N) | 0, m % N, me);
      if (comps.size) tries.push([m, comps]);
    }
    tries.sort((a, b) => b[1].size - a[1].size);
    return vcfTry(board, tries, me, depth, ctx);
  }
  if (depth <= 0) return null;
  const tries = [];
  for (const m of candidates(board, 1)) {
    const comps = fourCompletions(board, (m / N) | 0, m % N, me);
    if (comps.size) tries.push([m, comps]);
  }
  tries.sort((a, b) => b[1].size - a[1].size);
  return vcfTry(board, tries, me, depth, ctx);
}

function vcfTry(board, tries, me, depth, ctx) {
  const opp = 3 - me;
  for (const [m, comps] of tries) {
    play(board, (m / N) | 0, m % N);
    if (comps.size >= 2) { undo(board); return [m]; }
    const block = comps.values().next().value;
    play(board, (block / N) | 0, block % N);
    const refuted = winnerAfter(board, block) === opp;
    let sub = null;
    if (!refuted) {
      try {
        sub = vcfSearch(board, me, depth - 1, ctx);
      } catch (e) {
        undo(board); undo(board); throw e;
      }
    }
    undo(board); undo(board);
    if (sub) return [m].concat(sub);
  }
  return null;
}

function vcfSafe(board, attacker, depth) {
  const ctx = { nodes: 0, deadline: Date.now() + CFG.vcfBudgetMs };
  try {
    return vcfSearch(board, attacker, depth, ctx);
  } catch (e) {
    if (e && e.abort) return null;
    throw e;
  }
}

// ---------------- full move selection (port of players.NetPlayer) -----------

let cachedRoot = null;      // search tree from the previous move
let cachedHistory = [];     // game history the tree was grown on

function reusableRoot(history) {
  // reuse subtree if: cached history is a prefix of current history, and the
  // tree contains the last two moves (our reply + opponent's answer)
  if (!cachedRoot || history.length < 2) return null;
  if (history.length < cachedHistory.length + 2) return null;
  for (let i = 0; i < cachedHistory.length; i++) {
    if (cachedHistory[i] !== history[i]) return null;
  }
  let node = cachedRoot;
  for (let i = cachedHistory.length; i < history.length; i++) {
    if (!node.children) return null;
    const next = node.children.get(history[i]);
    if (!next) return null;
    node = next;
  }
  return node;
}

async function chooseMove(board, sims, onProgress) {
  const p = board.toMove, opp = 3 - p;
  const info = { kind: "mcts" };

  const my5 = fivePoints(board, p);
  if (my5.length) { info.kind = "win-now"; return { move: my5[0], info }; }
  const opp5 = fivePoints(board, opp);
  if (opp5.length) { info.kind = "block-five"; return { move: opp5[0], info }; }

  const line = vcfSafe(board, p, CFG.vcfDepth);
  if (line) {
    info.kind = "vcf-proof"; info.vcfLen = line.length;
    return { move: line[0], info };
  }

  // white reply book: answer black's first stone with a strong adjacent point
  let restrict = null;
  if (board.history.length === 1 && p !== 1) {
    const s = board.history[0];
    const sr = (s / N) | 0, sc = s % N;
    restrict = new Set();
    for (const [dr, dc] of [[0,1],[1,0],[1,1],[1,-1],[0,-1],[-1,0],[-1,-1],[-1,1]]) {
      const rr = sr + dr, cc = sc + dc;
      if (rr >= 0 && rr < N && cc >= 0 && cc < N) restrict.add(rr * N + cc);
    }
  }

  const reuse = reusableRoot(board.history);
  if (reuse) info.treeReuse = true;
  const { move, q, ranked, root } =
    await mcts(board, sims, reuse, onProgress && ((s) => onProgress(s, reuse ? 2 : 1)));

  // grow the cache for the next move
  cachedRoot = root;
  cachedHistory = board.history.slice();

  info.q = q;

  // fork guard: don't allow the opponent an unstoppable four unless our move
  // is itself forcing (creates a four of our own)
  const myComps = fourCompletions(board, (move / N) | 0, move % N, p);
  if (myComps.size === 0) {
    const rankedMoves = ranked.filter(v => v[1] > 0).slice(0, 8).map(v => v[0]);
    let chosen = null;
    for (const m of rankedMoves) {
      if (!allowsFork(board, m, opp)) { chosen = m; break; }
    }
    if (chosen === null) {
      const defenses = new Set();
      for (const w of candidates(board, 1)) {
        const comps = fourCompletions(board, (w / N) | 0, w % N, opp);
        if (comps.size >= 2) { defenses.add(w); for (const c of comps) defenses.add(c); }
      }
      const prior = new Map(ranked);
      for (const w of candidates(board, 2)) if (!prior.has(w)) prior.set(w, 0);
      for (const m of [...defenses].sort((a, b) => (prior.get(b) || 0) - (prior.get(a) || 0))) {
        if (!allowsFork(board, m, opp)) { chosen = m; break; }
      }
    }
    if (chosen !== null && chosen !== move) {
      info.forkGuard = `${move}->${chosen}`;
      move = chosen;
      cachedRoot = null; cachedHistory = [];   // off-policy move: drop tree
    }
  }
  return { move, info };
}

function allowsFork(board, m, opp) {
  play(board, (m / N) | 0, m % N);
  let bad = fivePoints(board, opp).length > 0;
  if (!bad) {
    for (const w of candidates(board, 1)) {
      if (fourCompletions(board, (w / N) | 0, w % N, opp).size >= 2) { bad = true; break; }
    }
  }
  undo(board);
  return bad;
}
