// UI logic: board rendering (canvas), game flow, worker messaging.
"use strict";
const worker = new Worker("worker.js");

let board = null;          // {cells, toMove, last, history} (mirrors worker's)
let humanColor = 1;        // 1 black, 2 white
let sims = 128;
let thinking = false;
let gameOver = false;
let pendingId = 0;
const pending = new Map();
let busyKind = null;

const $ = (id) => document.getElementById(id);
const canvas = $("board");
const ctx2d = canvas.getContext("2d");

// ---------- rendering ----------
let geom = { cell: 0, pad: 0, size: 0 };

function resizeCanvas() {
  const wrap = $("boardWrap");
  const size = Math.min(wrap.clientWidth, window.innerHeight * 0.62);
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = size + "px";
  canvas.style.height = size + "px";
  canvas.width = Math.round(size * dpr);
  canvas.height = Math.round(size * dpr);
  ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
  geom.size = size;
  geom.pad = size / (N + 1);
  geom.cell = (size - 2 * geom.pad) / (N - 1);
  draw();
}

function xy(i) {
  const r = (i / N) | 0, c = i % N;
  return [geom.pad + c * geom.cell, geom.pad + r * geom.cell];
}

let winLineIdx = [];

function draw() {
  if (!board) return;
  const g = geom;
  // wood background
  const grad = ctx2d.createLinearGradient(0, 0, g.size, g.size);
  grad.addColorStop(0, "#caa26a"); grad.addColorStop(1, "#b58e58");
  ctx2d.fillStyle = grad;
  ctx2d.fillRect(0, 0, g.size, g.size);

  ctx2d.strokeStyle = "rgba(60,40,20,.75)";
  ctx2d.lineWidth = 1;
  for (let i = 0; i < N; i++) {
    const p = g.pad + i * g.cell;
    ctx2d.beginPath(); ctx2d.moveTo(g.pad, p); ctx2d.lineTo(g.size - g.pad, p); ctx2d.stroke();
    ctx2d.beginPath(); ctx2d.moveTo(p, g.pad); ctx2d.lineTo(p, g.size - g.pad); ctx2d.stroke();
  }
  // star points
  for (const [r, c] of [[3,3],[3,11],[11,3],[11,11],[7,7]]) {
    const [x, y] = xy(r * N + c);
    ctx2d.beginPath(); ctx2d.arc(x, y, 2.5, 0, 7); ctx2d.fillStyle = "rgba(60,40,20,.8)";
    ctx2d.fill();
  }
  // coordinates
  ctx2d.fillStyle = "rgba(50,32,15,.85)";
  ctx2d.font = `${Math.max(9, g.cell * 0.32)}px sans-serif`;
  ctx2d.textAlign = "center"; ctx2d.textBaseline = "middle";
  for (let c = 0; c < N; c++) {
    const [x] = xy(c);
    ctx2d.fillText(String.fromCharCode(65 + c), x, g.pad / 2);
    ctx2d.fillText(String(c + 1), x, g.size - g.pad / 2);
  }

  // stones
  for (let i = 0; i < N * N; i++) {
    const v = board.cells[i];
    if (!v) continue;
    const [x, y] = xy(i);
    const r = g.cell * 0.44;
    ctx2d.beginPath(); ctx2d.arc(x, y, r, 0, 7);
    if (v === 1) {
      ctx2d.fillStyle = "#111"; ctx2d.fill();
      ctx2d.strokeStyle = "#000"; ctx2d.stroke();
      ctx2d.beginPath(); ctx2d.arc(x - r * 0.3, y - r * 0.3, r * 0.35, 0, 7);
      ctx2d.fillStyle = "rgba(255,255,255,.25)"; ctx2d.fill();
    } else {
      ctx2d.fillStyle = "#f4efe4"; ctx2d.fill();
      ctx2d.strokeStyle = "#b9ac91"; ctx2d.stroke();
    }
  }
  // last move marker
  if (board.last >= 0 && !winLineIdx.length) {
    const [x, y] = xy(board.last);
    ctx2d.beginPath(); ctx2d.arc(x, y, g.cell * 0.16, 0, 7);
    ctx2d.fillStyle = "#e2574c"; ctx2d.fill();
  }
  // win line
  if (winLineIdx.length) {
    ctx2d.strokeStyle = "#e2574c"; ctx2d.lineWidth = Math.max(2, g.cell * 0.12);
    const [x1, y1] = xy(winLineIdx[0]);
    const [x2, y2] = xy(winLineIdx[winLineIdx.length - 1]);
    ctx2d.beginPath(); ctx2d.moveTo(x1, y1); ctx2d.lineTo(x2, y2); ctx2d.stroke();
  }
}

// ---------- helpers ----------
const NAMES = { "win-now": "直接连五", "block-five": "封堵连五", "vcf-proof": "VCF 必胜证明", "mcts": "MCTS 选点" };

function setStatus(text, cls) {
  const el = $("status");
  el.textContent = text;
  el.className = cls || "";
}
function logLine(html) {
  const el = $("log");
  const div = document.createElement("div");
  div.innerHTML = html;
  el.prepend(div);
}
function callWorker(payload, onDone) {
  const id = ++pendingId;
  pending.set(id, onDone);
  worker.postMessage(Object.assign({ id }, payload));
}
function serialize() {
  return { cells: board.cells.slice(), toMove: board.toMove,
           last: board.last, history: board.history.slice() };
}

// ---------- game flow ----------
function newGame() {
  board = { cells: new Int8Array(N * N), toMove: 1, last: -1, history: [] };
  humanColor = parseInt($("selColor").value, 10) === 2 ? 2 : 1;
  sims = parseInt($("selSims").value, 10);
  gameOver = false; thinking = false; winLineIdx = [];
  $("log").innerHTML = "";
  draw();
  if (board.toMove !== humanColor) aiTurn();
  else setStatus("你执" + (humanColor === 1 ? "黑" : "白") + "，请落子");
}

function aiTurn() {
  thinking = true;
  setStatus("AI 思考中…");
  $("progress").textContent = "";
  callWorker({ type: "move", board: serialize(), sims }, (msg) => {
    thinking = false;
    $("progress").textContent = "";
    applyMove(msg.move, "AI", msg.info);
  });
}

function applyMove(move, who, info) {
  play(board, (move / N) | 0, move % N);
  draw();
  const st = status(board);
  if (st === -1) { gameOver = true; setStatus("和棋"); return; }
  if (st) {
    gameOver = true;
    winLineIdx = winLine(board);
    draw();
    const blackWon = st === 1;
    const youWon = (st === humanColor);
    setStatus((blackWon ? "● 黑棋" : "○ 白棋") + " 连五获胜！" + (youWon ? " 你赢了 🎉" : " AI 获胜"), youWon ? "win" : "loss");
    return;
  }
  const extra = info ? `（${NAMES[info.kind] || info.kind}${info.forkGuard ? " · 防叉" : ""}${info.kind === "vcf-proof" ? " · 强制 " + info.vcfLen + " 手" : ""}）` : "";
  if (who === "AI") {
    setStatus("你执" + (humanColor === 1 ? "黑" : "白") + "，请落子");
    logLine(`AI 落子 <span class="k">${coordName(move)}</span> ${extra}`);
  } else {
    logLine(`你落子 <span class="k">${coordName(move)}</span>`);
    aiTurn();
  }
}

function coordName(i) {
  return String.fromCharCode(65 + (i % N)) + (((i / N) | 0) + 1);
}

canvas.addEventListener("pointerdown", (e) => {
  if (!board || thinking || gameOver) return;
  if (board.toMove !== humanColor) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  const c = Math.round((x - geom.pad) / geom.cell);
  const r = Math.round((y - geom.pad) / geom.cell);
  if (r < 0 || r >= N || c < 0 || c >= N) return;
  const i = r * N + c;
  if (board.cells[i] !== 0) return;
  // snap check: tap must be near the intersection
  const [gx, gy] = xy(i);
  if (Math.hypot(x - gx, y - gy) > geom.cell * 0.55) return;
  applyMove(i, "你", null);
});

$("btnNew").addEventListener("click", newGame);
$("selColor").addEventListener("change", newGame);

$("btnUndo").addEventListener("click", () => {
  if (thinking || !board.history.length) return;
  const undoOne = () => {
    board.cells[board.history.pop()] = 0;
    board.toMove = 3 - board.toMove;
    board.last = board.history.length ? board.history[board.history.length - 1] : -1;
  };
  gameOver = false; winLineIdx = []; setStatus("已悔棋");
  undoOne();
  if (board.toMove !== humanColor && board.history.length) undoOne();
  if (board.toMove !== humanColor) { draw(); aiTurn(); } else draw();
});

$("btnHint").addEventListener("click", () => {
  if (thinking || gameOver || board.toMove !== humanColor) return;
  thinking = true; setStatus("计算提示中…");
  callWorker({ type: "eval", board: serialize() }, (msg) => {
    thinking = false;
    setStatus("提示：" + coordName(msg.move) + "（局面评估 " + msg.value.toFixed(2) + "）");
  });
});

$("btnResign").addEventListener("click", () => {
  if (gameOver) return;
  gameOver = true;
  setStatus("你认输了，AI 获胜", "loss");
});

// ---------- worker events ----------
worker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "ready") { setStatus("模型已加载"); newGame(); }
  else if (msg.type === "progress") {
    $("progress").textContent = `搜索 ${msg.sim}/${msg.sims}`;
  } else if (msg.type === "error") {
    thinking = false;
    setStatus("出错：" + msg.message, "loss");
  } else if (pending.has(msg.id)) {
    const cb = pending.get(msg.id);
    pending.delete(msg.id);
    cb(msg);
  }
};

worker.postMessage({ type: "init", modelPath: "models/gomoku_policy_value.onnx" });
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

// automated-test hook (only active with ?debug=1)
if (location.search.includes("debug=1")) {
  window.__gomokuTest = {
    setStones(list) {           // list of [r,c,1|2]
      board.cells.fill(0);
      board.history = [];
      for (const [r, c, v] of list) {
        board.cells[r * N + c] = v;
        board.history.push(r * N + c);
      }
      board.last = board.history.length ? board.history[board.history.length - 1] : -1;
      board.toMove = board.history.length % 2 === 0 ? 1 : 2;
      gameOver = false; winLineIdx = []; thinking = false;
      draw();
      setStatus("你执黑，请落子");
    },
    status: () => $("status").textContent,
  };
}
