// Node regression test for the mobile app AI engine (no browser needed):
//   node tests/test_ai_js.mjs
// Guards against the "Assignment to constant variable" crash that occurred
// when the fork guard replaced MCTS's chosen move (line: move = chosen).
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const app = path.join(here, "..", "mobile-app");
const ctx = { console, Date, Math };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(app, "engine.js"), "utf8"), ctx);
vm.runInContext(fs.readFileSync(path.join(app, "ai.js"), "utf8"), ctx);

// mock the network: uniform priors, neutral value (deterministic search)
vm.runInContext(`
  evaluate = async (b, restrict) => {
    let cand = candidates(b, 2);
    if (restrict) cand = cand.filter(i => restrict.has(i));
    const p = new Map();
    for (const i of cand) p.set(i, 1 / cand.length);
    return { prior: p, value: 0 };
  };
`, ctx);

const script = `
  (async () => {
    let guardFired = 0, games = 0, crashes = 0;
    for (let seed = 0; seed < 8; seed++) {
      const b = newBoard();
      const rngState = seed * 2654435761 % 2147483647;
      const rnd = () => (rngState * 48271 % 2147483647) / 2147483647;
      // random-but-legal opening scramble
      for (let ply = 0; ply < 10 + seed * 2 && !status(b); ply++) {
        const cand = candidates(b, 2);
        play(b, cand[(rnd() * cand.length) | 0]);
      }
      if (status(b)) continue;
      games++;
      try {
        const res = await chooseMove(b, 24, null);
        if (res.info.forkGuard) guardFired++;
      } catch (e) {
        crashes++;
        console.log("CRASH:", String(e && e.message || e));
      }
    }
    // phase 2: opponent open three -> fork guard must engage (this is the
    // exact scenario that crashed with "Assignment to constant variable").
    // VCF is stubbed out so the flow reaches MCTS + fork guard.
    vcfSafe = () => null;
    {
      const b = newBoard();
      [[7,6],[0,0],[7,7],[0,1],[7,8],[1,3]].forEach(([r,c]) => play(b, r, c));
      // white to move; black has an open three (7,6)-(7,8)
      games++;
      try {
        const res = await chooseMove(b, 32, null);
        if (res.info.forkGuard) guardFired++;
        console.log("PHASE2 move:", res.move, "kind:", res.info.kind,
                    "forkGuard:", res.info.forkGuard || "(none)");
      } catch (e) {
        crashes++;
        console.log("CRASH(phase2):", String(e && e.message || e));
      }
    }
    // phase 3: force the replacement path — MCTS (stubbed) returns (0,2),
    // which lets white fork through the open three (8,6)-(8,8); the guard must
    // swap it for a safe block. Regression for the const-assignment crash.
    {
      const b = newBoard();
      const stones = [[8,6,2],[8,7,2],[8,8,2],[7,10,2],[0,0,1],[0,1,1],[1,3,1],[7,6,1]];
      for (const [r, c, v] of stones) b.cells[r * N + c] = v;
      b.to_move = 1;
      b.history = stones.map(([r, c]) => r * N + c);
      b.last = b.history[b.history.length - 1];
      mcts = async (board, sims) => ({
        move: 0 * 15 + 2, q: 0, ranked: [[0 * 15 + 2, 32], [8 * 15 + 5, 8]], root: null,
      });
      const res = await chooseMove(b, 8, null);
      const safe = !allowsFork(b, res.move, 2);
      console.log("PHASE3 move:", res.move, "kind:", res.info.kind,
                  "forkGuard:", res.info.forkGuard || "(none)", "safe:", safe);
      if (!safe || !res.info.forkGuard) {
        console.log("FAIL guard did not swap to a safe move");
        process.exitCode = 1;
      }
      games++;
    }
    console.log("STATS", JSON.stringify({ games, guardFired, crashes }));
    if (crashes > 0) process.exitCode = 1;
  })();
`;
await vm.runInContext(script, ctx);
