# Chess_RL — 棋类游戏深度强化学习实战

用"**行为克隆热启动 + AlphaZero 式自我对弈强化学习**"系列方法，逐个攻克不同棋类。
每个子项目独立可复现，配通俗教程（面向有深度学习基础、但没做过棋类 RL 的读者）。

| 子项目 | 棋类 | 状态 | 模型大小 | 战绩 |
|---|---|---|---|---|
| [`gomoku/`](gomoku/) | 五子棋 15×15 | ✅ 完成 | 284k 参数 / int8 ONNX 302KB | 执黑对全部内置对手 100% 不败（含规则老师 AI 与旧版自身） |
| `xiangqi/` | 中国象棋 | 🚧 计划中 | - | - |
| `go/` | 围棋 | 🚧 计划中 | - | - |

## 快速开始（以五子棋为例）

```bash
cd gomoku
pip install -r requirements.txt

python3 -m gomoku.play --color black --ckpt checkpoints/best.pt   # 人机对弈
bash scripts/run_training.sh                                      # 完整训练(~2h)
python3 -m gomoku.arena --ckpt checkpoints/best.pt \
    --opponents random heuristic --games 100                      # 强度评估
```

## 学习路线

1. 先读 [`gomoku/TUTORIAL.md`](gomoku/TUTORIAL.md)——从 MDP 概念到手机部署的完整实验教程，
   包含真实踩坑记录（和棋标签、候选网络漂移、活四叉防御）；
2. 对照源码跑通五子棋全流程；
3. 后续象棋/围棋子项目将复用同一框架：规则引擎 → 双头网络 → MCTS → 自我对弈 → 门控晋级。

## 统一框架

```
规则引擎(play/undo/status/candidates)
   → 策略价值网络(双头小 ResNet)
   → MCTS(PUCT + Dirichlet 噪声)
   → 自我对弈 + 经验回放 + 晋级门控
   → 多对手评估 + ONNX int8 手机部署
```
