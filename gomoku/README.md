# Gomoku AI — 小模型五子棋 + AlphaZero 式强化学习

15×15 自由规则五子棋（无禁手，≥5 连珠获胜，可配置棋盘大小）。训练流程：
**行为克隆热启动（和规则型老师 AI 对弈学习）→ AlphaZero 式自我对弈强化学习**，
最终引擎 = 小型策略价值网络（~28 万参数）+ MCTS + 强制着法 + **VCF 必胜证明器**，
模型可导出 int8 ONNX（约 300 KB）部署到手机。

## 目录结构

```
gomoku/
  board.py        棋盘、规则、胜负判定、候选点/威胁点查询、编码与对称增强
  heuristic.py    规则型老师 AI（棋型打分：五/活四/冲四/活三…，攻防加权）
  model.py        策略价值 ResNet（4 残差块 × 48 通道，双头：225 点策略 + 局面价值）
  mcts.py         PUCT MCTS（Dirichlet 根噪声、访问次数策略目标）
  vcf.py          VCF 连续冲四必胜证明器（只走强制着法，证明可靠）
  players.py      玩家封装：随机 / 老师 / NetPlayer(强制着法→VCF→MCTS)
  selfplay.py     多进程自我对弈数据生成
  eval.py         并行对抗赛（任意两种玩家规格对打）
  train.py        训练管线：BC 热启动 + 自我对弈 RL + 晋级门控
  arena.py        强度评估（vs random / heuristic / 其他 checkpoint）
  play.py         终端人机对弈
  export_onnx.py  ONNX 导出 + int8 动态量化 + 数值一致性校验
tests/test_gomoku.py   规则/启发式/VCF 正确性测试（含随机局面证明健全性）
mobile/ANDROID.md      Android(ONNX Runtime) / iOS / TFLite 集成说明 + Kotlin 代码
scripts/run_training.sh  完整训练启动脚本
```

## 快速开始

```bash
pip install -r requirements.txt

# 人机对弈（你执黑，AI 执白，200 次 MCTS 模拟 + VCF 证明器）
python3 -m gomoku.play --color black --ckpt checkpoints/best.pt --sims 200

# 评估：黑棋胜率（100 局，一半执黑一半执白）
python3 -m gomoku.arena --ckpt checkpoints/best.pt --opponents heuristic random --games 100 --sims 128

# 导出手机模型（fp32 + int8 ONNX，含一致性校验）
python3 -m gomoku.export_onnx --ckpt checkpoints/best.pt --out mobile/models
```

## 训练

```bash
# 完整训练（可先用 --hours 0.2 快速试跑）
bash scripts/run_training.sh

# 纯自我对弈 RL（跳过行为克隆）
python3 -m gomoku.train --skip-bc --hours 2 --out checkpoints_pure
```

训练流程：

1. **BC 热启动**：老师 AI（vs 随机对手的速胜对局，全部局面由确定性老师重标注）
   监督训练策略头（CE）与价值头（胜负 MSE）。
2. **自我对弈 RL**：当前网络 MCTS(64 sims + Dirichlet 噪声) 自我对弈 24 局/轮 →
   经验回放（8 对称增强，CE 策略 + MSE 价值）→ 候选网络与卫冕网络双色 12 局晋级赛
   （≥55% 晋级）→ 每轮同时统计黑棋对老师胜率。
3. 终局引擎再加两层**逻辑保证**：能连五必下 / 对方将连五必堵；
   VCF 证明器找到连续冲四路线时，胜利是**强制性**的（证明，非概率）。

## 关于"黑棋必胜"

自由规则五子棋在数学上已证明先手必胜（Allis 1994），但"每盘都赢"的完整证明需要
全局搜索，纯 RL 只能逼近。本项目的工程化定义与实测（最终权重，MCTS 128 sims）：

| 对手 | 总战绩 | **黑棋胜率** | 平均局长 |
|---|---|---|---|
| 随机 AI | 100/100 | **100%** | 30.6 手 |
| 规则老师 AI | 100/100 | **100%** | 45.1 手 |
| 初版 checkpoint（RL 之前） | 63/100 | **100%**（50/50 执黑） | 63.4 手 |

独立复核（确定性协议、20 局）：黑棋 19/20。合计黑棋对老师 69/70 ≈ **98.6%**。
把搜索加深到 256 sims（手机可承受）复测 100 局：**总战绩 100/100（黑白各 50/50）**。
胜局中绝大多数以"win-now"（直接连五）结束；VCF 证明器与强制着法（连五/堵五/防叉）
提供逻辑保证。严格意义上的"每一盘都必胜"需要全局求解器，不在 RL 可达范围——
但面对已内置的全部对手，黑棋实际未尝一败（除确定性协议下的 1 局）。

## 模型大小

| 项目 | 数值 |
|---|---|
| 参数量 | 284,322 |
| fp32 ONNX | 1114 KB |
| int8 ONNX | **302 KB** |
| 数值一致性 | Δpolicy ≈ 1e-5，Δvalue ≈ 1e-6 |
| 训练量 | 62 轮自我对弈迭代（24-32 局/轮 × MCTS 64 sims，~1500 局），11 次晋级 |
| 手机端推理 | 单次前向 ~ms 级；200 sims 的 MCTS 实时对弈无压力 |
