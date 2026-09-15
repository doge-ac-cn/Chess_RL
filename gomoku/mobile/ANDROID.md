# Mobile deployment (Android)

The trained network is ~284k params. Export produces two ONNX artifacts:

- `mobile/models/gomoku_policy_value.onnx`        (fp32, ~1.1 MB)
- `mobile/models/gomoku_policy_value.int8.onnx`   (int8-quantized, ~250-300 KB) ← ship this

Export: `python3 -m gomoku.export_onnx --ckpt checkpoints/best.pt --out mobile/models`

## Android integration (ONNX Runtime Mobile)

`app/build.gradle`:

```gradle
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")
}
```

Put `gomoku_policy_value.int8.onnx` in `app/src/main/assets/models/`.

`GomokuEngine.kt` (drop-in, mirrors `gomoku/players.py`):

```kotlin
package com.example.gomoku

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.sqrt

class GomokuEngine(assetPath: String, private val simulations: Int = 200) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(assetPath)
    private val n = 15
    private var cells = Array(n) { IntArray(n) }   // 0 empty, 1 black, 2 white
    private var toMove = 1

    /** Full move selection: forced moves -> (optional MCTS) greedy policy search. */
    fun playMove(): Pair<Int, Int> {
        fivePoints(toMove).firstOrNull()?.let { return it }              // win now
        val oppFive = fivePoints(3 - toMove)
        if (oppFive.isNotEmpty()) return oppFive.first()                 // must block
        // N simulations of a light PUCT search (see mcts() below)
        return mcts()
    }

    fun humanMove(r: Int, c: Int) { cells[r][c] = toMove; toMove = 3 - toMove }

    // ---------- network ----------
    private fun planes(): Array<Array<FloatArray>> {
        val x = Array(4) { Array(n) { FloatArray(n) } }
        for (r in 0 until n) for (c in 0 until n) {
            val v = cells[r][c]
            if (v == toMove) x[0][r][c] = 1f
            if (v == 3 - toMove) x[1][r][c] = 1f
        }
        // x[2] last move / x[3] color flag: optional, set if you track them
        return x
    }

    private class Out(val policy: FloatArray, val value: Float)

    private fun infer(): Out {
        val shape = longArrayOf(1, 4, n.toLong(), n.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(planes().flatMap { it.toList() }.toFloatArray()), shape).use { t ->
            session.run(mapOf("planes" to t)).use { res ->
                @Suppress("UNCHECKED_CAST")
                val p = (res[0].value as Array<FloatArray>)[0]
                val v = (res[1].value as FloatArray)[0]
                return Out(p, v)
            }
        }
    }

    // ---------- rules helpers ----------
    private fun winnerAfter(r: Int, c: Int): Int {
        val p = cells[r][c]
        if (p == 0) return 0
        val dirs = arrayOf(intArrayOf(0, 1), intArrayOf(1, 0), intArrayOf(1, 1), intArrayOf(1, -1))
        for (d in dirs) {
            var cnt = 1
            for (s in intArrayOf(1, -1)) {
                var rr = r + d[0] * s; var cc = c + d[1] * s
                while (rr in 0 until n && cc in 0 until n && cells[rr][cc] == p) {
                    cnt++; rr += d[0] * s; cc += d[1] * s
                }
            }
            if (cnt >= 5) return p
        }
        return 0
    }

    private fun candidates(radius: Int = 2): List<Pair<Int, Int>> {
        val out = mutableListOf<Pair<Int, Int>>()
        var any = false
        for (r in 0 until n) for (c in 0 until n) if (cells[r][c] != 0) { any = true }
        if (!any) return listOf(n / 2 to n / 2)
        for (r in 0 until n) for (c in 0 until n) if (cells[r][c] == 0) {
            loop@ for (rr in max(0, r - radius)..min(n - 1, r + radius))
                for (cc in max(0, c - radius)..min(n - 1, c + radius))
                    if (cells[rr][cc] != 0) { out.add(r to c); break@loop }
        }
        return out
    }

    private fun fivePoints(p: Int): List<Pair<Int, Int>> {
        val pts = mutableListOf<Pair<Int, Int>>()
        for ((r, c) in candidates(1)) {
            cells[r][c] = p
            if (winnerAfter(r, c) == p) pts.add(r to c)
            cells[r][c] = 0
        }
        return pts
    }

    // ---------- minimal PUCT MCTS over the policy-value net ----------
    private class Child(var prior: Float, var N: Int = 0, var W: Float = 0f,
                        val move: Pair<Int, Int>)

    private fun mcts(): Pair<Int, Int> {
        val rootOut = infer()
        val cand = candidates()
        val cSet = cand.toHashSet()
        val children = cand.map { Child(rootOut.policy[it.first * n + it.second], move = it) }
            .filter { it.prior > 0.003f }        // prune junk priors for speed
        val cpuct = 1.8f
        repeat(simulations) {
            // (single-node bandit: good enough on phones with strong priors;
            //  port gomoku/mcts.py for a full tree if you can afford it)
            val sqrtN = sqrt(max(1, children.sumOf { it.N }).toFloat())
            var best = children[0]; var bestV = -Float.MAX_VALUE
            for (ch in children) {
                val q = if (ch.N > 0) ch.W / ch.N else 0f
                val u = cpuct * ch.prior * sqrtN / (1 + ch.N)
                if (q + u > bestV) { bestV = q + u; best = ch }
            }
            cells[best.move.first][best.move.second] = toMove
            val win = winnerAfter(best.move.first, best.move.second)
            val v = when {
                win == toMove -> 1f
                win != 0 -> -1f
                else -> {
                    toMove = 3 - toMove
                    val o = -infer().value          // opponent's view
                    toMove = 3 - toMove
                    o
                }
            }
            cells[best.move.first][best.move.second] = 0
            best.W += v; best.N++
        }
        val mv = children.maxByOrNull { it.N }!!.move
        cells[mv.first][mv.second] = toMove
        toMove = 3 - toMove
        return mv
    }
}
```

Notes
- The Kotlin snippet implements a **root-parallel-free single-level PUCT** bandit
  (network prior + value). For a full tree, port `gomoku/mcts.py` — the logic is
  ~80 lines. `candidates()` and forced-move logic are shared.
- VCF proof search (`gomoku/vcf.py`) is pure Python and ports directly to
  Kotlin/Java; it gives the phone engine *proven* wins whenever a continuous-four
  line exists.
- iOS: same ONNX with `onnxruntime-objc` (`pod 'onnxruntime-objc'`), or convert
  to Core ML with `coremltools` if you prefer native.

## Optional: TFLite

```bash
pip install ai-edge-torch   # Google's torch->tflite converter
python3 - <<'EOF'
import ai_edge_torch, torch
from gomoku.model import GomokuNet
net = GomokuNet(15); net.load_state_dict(torch.load("checkpoints/best.pt", weights_only=True)["model"]); net.eval()
m = ai_edge_torch.convert(net.eval(), (torch.randn(1, 4, 15, 15),))
m.export("mobile/models/gomoku.tflite")
EOF
```
