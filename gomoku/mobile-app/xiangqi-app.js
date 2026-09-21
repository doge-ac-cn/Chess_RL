// Xiangqi page UI: board rendering (canvas), game flow, worker messaging.
"use strict";
const XCOLS = 9, XROWS = 10;
const xworker = new Worker("xiangqi-worker.js");

let xb = null;              // {cells, toMove, last, history}
let xthinking = false;
let xgameOver = false;
let xpending = null;
let xselFrom = null;

const $ = (id) => document.getElementById(id);
const xcanvas = $("xboard");
const xctx = xcanvas.getContext("2d");
let xgeom = { cell: 0, rowH: 0, pad: 0, size: 0, h: 0 };

function xResize() {
  const wrap = $("xboardWrap");
  const size = Math.min(wrap.clientWidth, window.innerHeight * 0.6);
  const dpr = window.devicePixelRatio || 1;
  const height = size * 1.12;
  xcanvas.style.width = size + "px";
  xcanvas.style.height = height + "px";
  xcanvas.width = Math.round(size * dpr);
  xcanvas.height = Math.round(height * dpr);
  xctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  xgeom.size = size; xgeom.h = height;
  xgeom.pad = size / 11;
  xgeom.cell = (size - 2 * xgeom.pad) / (XCOLS - 1);
  xgeom.rowH = (height - 2 * xgeom.pad) / (XROWS - 1);
  xDraw();
}

function xXY(i) {
  const r = (i / XCOLS) | 0, c = i % XCOLS;
  return [xgeom.pad + c * xgeom.cell, xgeom.pad + r * xgeom.rowH];
}

const GLYPH = { 1: "帥", 2: "仕", 3: "相", 4: "馬", 5: "車", 6: "炮", 7: "兵",
                11: "將", 12: "士", 13: "象", 14: "馬", 15: "車", 16: "砲", 17: "卒" };

function xDraw() {
  if (!xb) return;
  const g = xgeom;
  const grad = xctx.createLinearGradient(0, 0, g.size, g.h);
  grad.addColorStop(0, "#e8c99a"); grad.addColorStop(1, "#d9b783");
  xctx.fillStyle = grad;
  xctx.fillRect(0, 0, g.size, g.h);

  xctx.strokeStyle = "rgba(80,50,20,.8)"; xctx.lineWidth = 1;
  for (let r = 0; r < XROWS; r++) {
    const y = g.pad + r * g.rowH;
    xctx.beginPath(); xctx.moveTo(g.pad, y); xctx.lineTo(g.size - g.pad, y); xctx.stroke();
  }
  for (let c = 0; c < XCOLS; c++) {
    const x = g.pad + c * g.cell;
    xctx.beginPath(); xctx.moveTo(x, g.pad); xctx.lineTo(x, g.pad + 4 * g.rowH); xctx.stroke();
    if (c === 0 || c === XCOLS - 1) {
      xctx.beginPath(); xctx.moveTo(x, g.pad + 4 * g.rowH); xctx.lineTo(x, g.pad + 9 * g.rowH); xctx.stroke();
    } else {
      xctx.beginPath(); xctx.moveTo(x, g.pad + 5 * g.rowH); xctx.lineTo(x, g.pad + 9 * g.rowH); xctx.stroke();
    }
  }
  for (const top of [0, 7]) {
    xctx.beginPath();
    xctx.moveTo(g.pad + 3 * g.cell, g.pad + top * g.rowH);
    xctx.lineTo(g.pad + 5 * g.cell, g.pad + (top + 2) * g.rowH);
    xctx.moveTo(g.pad + 5 * g.cell, g.pad + top * g.rowH);
    xctx.lineTo(g.pad + 3 * g.cell, g.pad + (top + 2) * g.rowH);
    xctx.stroke();
  }
  xctx.fillStyle = "rgba(90,60,25,.8)";
  xctx.font = `${Math.max(10, g.cell * 0.45)}px serif`;
  xctx.textAlign = "center"; xctx.textBaseline = "middle";
  xctx.fillText("楚 河        漢 界", g.size / 2, g.pad + 4.5 * g.rowH);
  xctx.font = `${Math.max(9, g.cell * 0.3)}px sans-serif`;
  for (let c = 0; c < XCOLS; c++) {
    const x = g.pad + c * g.cell;
    xctx.fillText(String(XCOLS - c), x, g.pad / 2 - 2);
    xctx.fillText(String(c + 1), x, g.h - g.pad / 2 + 2);
  }

  for (let i = 0; i < 90; i++) {
    const p = xb.cells[i];
    if (!p) continue;
    const [x, y] = xXY(i);
    const r = Math.min(g.cell, g.rowH) * 0.46;
    xctx.beginPath(); xctx.arc(x, y, r, 0, 7);
    xctx.fillStyle = "#f6f1e6"; xctx.fill();
    xctx.lineWidth = 2;
    xctx.strokeStyle = p < 10 ? "#c0392b" : "#2c3e50";
    xctx.stroke();
    xctx.fillStyle = p < 10 ? "#c0392b" : "#2c3e50";
    xctx.font = `bold ${r * 1.05}px serif`;
    xctx.fillText(GLYPH[p], x, y + r * 0.1);
  }
  if (xselFrom !== null) {
    const [x, y] = xXY(xselFrom);
    xctx.beginPath(); xctx.arc(x, y, Math.min(g.cell, g.rowH) * 0.48, 0, 7);
    xctx.strokeStyle = "#7ec97e"; xctx.lineWidth = 3; xctx.stroke();
  }
  if (xb.last >= 0 && xselFrom === null) {
    const [x, y] = xXY(xb.last);
    xctx.beginPath(); xctx.arc(x, y, Math.min(g.cell, g.rowH) * 0.47, 0, 7);
    xctx.strokeStyle = "#e2574c"; xctx.lineWidth = 2; xctx.stroke();
  }
}

function xSetName(i) {
  return String.fromCharCode(97 + (i % 9)) + (9 - ((i / 9) | 0));
}
function xSetStatus(text, cls) {
  const el = $("xstatus"); el.textContent = text; el.className = cls || "";
}
function xLog(html) {
  const div = document.createElement("div");
  div.innerHTML = html;
  $("xlog").prepend(div);
}
let xpendingDone = null;
function xAskAI() {
  xthinking = true;
  xSetStatus("AI 思考中…");
  xpendingDone = true;
  xworker.postMessage({ type: "move",
    board: { cells: Array.from(xb.cells), toMove: xb.toMove,
             last: xb.last, history: xb.history.slice() } });
}

function xApply(mv) {
  const [f, t] = mv;
  xb.history.push([f, t]);
  xb.cells[t] = xb.cells[f];
  xb.cells[f] = 0;
  xb.toMove = 3 - xb.toMove;
  xb.last = t;
  xDraw();
  if (XQ.status(xb)) {
    xgameOver = true;
    const winner = XQ.status(xb) === 1 ? "红方" : "黑方";
    xSetStatus(`${winner}获胜！` + (XQ.status(xb) === 1 ? " 你赢了 🎉" : " AI 获胜"),
               XQ.status(xb) === 1 ? "win" : "loss");
    return true;
  }
  return false;
}

function xNewGame() {
  xb = XQ.newBoard();
  xselFrom = null; xgameOver = false; xthinking = false;
  $("xlog").innerHTML = "";
  xSetStatus("你执红（先手）：先点选己方棋子，再点目标位置");
  xDraw();
}

xcanvas.addEventListener("pointerdown", (e) => {
  if (!xb || xthinking || xgameOver || xb.toMove !== 1) return;
  const rect = xcanvas.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  const c = Math.round((x - xgeom.pad) / xgeom.cell);
  const r = Math.round((y - xgeom.pad) / xgeom.rowH);
  if (r < 0 || r >= XROWS || c < 0 || c >= XCOLS) return;
  const i = r * XCOLS + c;
  if (xselFrom === null) {
    if (xb.cells[i] >= 1 && xb.cells[i] < 10) { xselFrom = i; xDraw(); }
    return;
  }
  if (i === xselFrom) { xselFrom = null; xDraw(); return; }
  const mv = [xselFrom, i];
  const isLegal = XQ.legalMoves(xb).some(m => m[0] === mv[0] && m[1] === mv[1]);
  if (!isLegal) {
    if (xb.cells[i] >= 1 && xb.cells[i] < 10) { xselFrom = i; xDraw(); }
    else { xselFrom = null; xDraw(); }
    return;
  }
  xselFrom = null;
  if (xApply(mv)) return;
  xLog(`你落子 <span class="k">${xSetName(mv[0])}->${xSetName(mv[1])}</span>`);
  xthinking = true;
  xSetStatus("AI 思考中…");
  xworker.postMessage({ type: "move",
    board: { cells: Array.from(xb.cells), toMove: xb.toMove,
             last: xb.last, history: xb.history.slice() } });
});

$("xbtnNew").addEventListener("click", xNewGame);
$("xbtnUndo").addEventListener("click", () => {
  if (xthinking || !xb || xb.history.length < 2) return;
  xb.history.splice(-2);
  xb.cells = XQ.newBoard().cells;
  for (const [f, t] of xb.history) { xb.cells[t] = xb.cells[f]; xb.cells[f] = 0; }
  xb.toMove = 1;
  xb.last = xb.history.length ? xb.history[xb.history.length - 1] : -1;
  xgameOver = false; xSetStatus("已悔棋"); xDraw();
});

xworker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "ready") { xSetStatus("模型已加载"); xNewGame(); }
  else if (msg.type === "error") { xthinking = false; xSetStatus("出错：" + msg.message, "loss"); }
  else if (msg.type === "move") {
    xthinking = false;
    if (xApply(msg.move)) return;
    xLog(`AI 落子 <span class="k">${xSetName(msg.move[0])}->${xSetName(msg.move[1])}</span> (${msg.info.kind})`);
    xSetStatus("你执红，请落子");
  }
};

xworker.postMessage({ type: "init", modelPath: "models/xiangqi_policy_value.onnx" });
window.addEventListener("resize", xResize);
xResize();
