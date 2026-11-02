# drexforge

> 一个零依赖的**直接打分决策模型**参考实现（Drex Max 风格）：用「Gated DeltaNet
> 线性注意力 + 全注意力」混合骨干在**一次前向**里读上下文、给每个候选项打一个
> 概率，**全程不生成任何文本**。纯 Python 标准库，无 numpy/torch/requests。

[![CI](https://github.com/huzjie/drexforge/actions/workflows/ci.yml/badge.svg)](https://github.com/huzjie/drexforge/actions/workflows/ci.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![zero-dependency](https://img.shields.io/badge/deps-zero-green)

---

## 这个模型和普通 LLM 有什么不同

普通语言模型是**自回归生成**：一个 token 一个 token 往外吐，最后再从文字里
扒答案。而决策模型（decision model）是**直接打分**：读入 context 和一组具名
候选项，一次前向就返回每个选项的概率，你直接 `argmax` / 阈值 / 取整。

为什么这值得单独做成一个项目？因为它是**系统里高频小决策**的最优解——

- **Agent 路由**：这个请求该交给哪个工具（search / python_exec / …）？
- **工单分诊**：优先级 critical / high / medium / low？
- **策略护栏**：这条内容违不违规（yes / no）？
- **工具选择**：该用哪个函数（read_file / run_test / …）？

这些决策要的是「快、便宜、可审计、能塞进热循环」，而不是一段话。决策模型
把答案压成一个**概率分布 + 索引**，调用方直接按整数分支。

## 两大核心机制

### 1. 混合骨干：48 DeltaNet + 16 全注意力

长上下文（Drex Max 是 262144 token）下，全注意力是 O(n²d) 的，根本跑不动。
所以骨干大部分层用 **Gated DeltaNet**（线性时间循环层），只在每第 4 层放一层
**全注意力**（精确的两两交互）。

```
S_t = beta_t * S_{t-1} + outer(v_t, k_t)   # 衰减门控写状态（线性时间）
o_t = S_t @ q_t                             # 线性读出
```

- `beta -> 1`：累积记忆（长上下文）；`beta -> 0`：只记当前 token。
- 状态 `S` 是 `d x d` 矩阵，每 token 更新一次，**从不物化 n x n 注意力矩阵**。

参考实现默认 12 层 = 9 DeltaNet + 3 全注意力（保持「每第 4 层全注意力」的节奏），
可一键切到纯 delta / 纯 full 做消融。

### 2. 选项打分头：一次前向出概率，不生成文本

把骨干输出池化成一个向量 `h`，和每个选项的**正交化嵌入**做余弦，再 softmax：

```python
from drexforge.core.scorer import OptionScorer

scorer = OptionScorer(dim=8)
d = scorer.score(h, ["search", "python_exec", "browse", "calculator"], "choice")
# d == {"choice": 0, "probs": [0.93, 0.02, 0.03, 0.02], "confidence": 0.93, ...}
```

三种题型：

| 题型 | 数学 | 决策 |
|---|---|---|
| choice | softmax 选项 logits | argmax |
| yesno | sigmoid("yes" logit)，no 为补集 | `p_yes >= 0.5` |
| ordinal | 有界标量映射到最近整数档 | `round(level)` |

**正交化选项嵌入是关键技巧**：正确答案嵌入与答案方向 cos≈1，每个干扰项 cos
恰好=0，所以 argmax **可证明**取对，margin 干净不靠运气。

## 快速开始

```bash
python -m drexforge doctor     # 冒烟：skill=1.0 时全部决策正确
python -m drexforge bench      # 单调基准 + 骨干消融
python -m drexforge decide route_api --skill 0.9   # 单条数值决策
python -m drexforge serve      # HTTP API on :8431
```

无需安装，全部 import 自标准库。HTTP 示例：

```bash
curl -X POST http://127.0.0.1:8431/v1/decide -d '{"task_id":"route_api","skill":0.9}'
# -> {"decision":0,"decision_label":"search","confidence":0.993,"hit":true,...}
```

## 实测结果（真实跑出来，非手写）

两条维度都可复现（确定性种子），`python -m drexforge bench` 一键重跑。

**Skill 轴**（0.5 随机 → 1.0 专家）：

| skill | accuracy | ordinal_mae | ece | confidence | entropy |
|---|---|---|---|---|---|
| 0.50 | 0.250 | 2.250 | 0.440 | 0.690 | 0.700 |
| 0.60 | 0.583 | 1.500 | 0.174 | 0.683 | 0.768 |
| 0.70 | 0.917 | 0.250 | 0.153 | 0.764 | 0.651 |
| 0.80 | 1.000 | 0.000 | 0.093 | 0.907 | 0.359 |
| 0.90 | 1.000 | 0.000 | 0.024 | 0.976 | 0.134 |
| 1.00 | 1.000 | 0.000 | 0.007 | 0.993 | 0.047 |

准确率 **0.250 → 1.000**、序数 MAE **2.250 → 0**、校准误差 ECE **0.440 → 0.007**、
置信度 **0.690 → 0.993**、熵 **0.700 → 0.047**，全单调。

**骨干消融**（32 token 上下文，12 层，为什么「48 delta + 16 full」）：

| mode | delta 层 | full 层 | 标量运算量 | 方向保持 |
|---|---|---|---|---|
| delta | 12 | 0 | 24576 | 0.9879 |
| hybrid | 9 | 3 | 43008 | 0.8168 |
| full | 0 | 12 | 98304 | 0.5636 |

全注意力比纯 DeltaNet **贵 4 倍**；混合骨干只贵 **1.75 倍**，却保留了全注意力
的两两交互能力。在真实 262144 token 窗口下，这个差距要再放大 ~26 万倍——这
正是决策模型把二次复杂度层压到少数的原因。

## 可复用的工程要点（干货）

- **置信度公式**（`core/model.py`，可直接抄）：`boost=(skill-0.5)*2`，
  `logit=0.5+boost*(evidence-0.5)+(1-|boost|)*(noise-0.5)`。信号与噪声同尺度，
  两端都收敛，弱信号不会被噪声淹没。
- **决策温度**（`core/model.py`）：softmax 前把 logits 乘一个温度 `T`，否则
  `[0,1]` 区间的 logits 即使选对，赢家概率也只有 ~0.47，置信度上不去。加了
  `T=6` 后正确时置信度 ~0.99，ECE 才真正趋零。
- **正交化选项嵌入**（`core/scorer.py`）：对种子嵌入做 Gram-Schmidt，让干扰项
  与答案方向 cos 恰为 0，argmax 可证明正确。
- **残差 + 归一化**（`core/deltanet.py` / `attention.py`）：层输出返回
  `normalize(x + o)` 而非裸 `o`，否则多层的随机变换会把「写进上下文里的答案
  方向」搅散，方向保持（`cos(h, answer_dir)`）直接归零。
- **选一个闭式真值任务**（`world/`）：把答案写进上下文主方向，真值可证伪，
  「skill 越高越对」是测量出来的，不是断言。
- 直接可复用的提示词模板在 `prompts/`，配置配方在 `recipes/`。

## 目录结构

```
drexforge/
  core/      deltanet, attention, backbone, scorer, model, engine
  world/     corpus（答案写进上下文）+ oracle（真值相关性）
  training/  单调 skill 训练环
  serving/   标准库 HTTP API
  utils/     rng, yamlish, linalg（零依赖）
tests/       标准库 unittest（78 个用例全绿）
examples/    decide_routing / guardrail_check / calibration_curve
docs/        architecture / deltanet / option_scoring / calibration / benchmark
prompts/     可复用提示词模板（routing/choice/yesno/ordinal/guardrail/system）
recipes/     yaml 配置配方
deploy/      Dockerfile, docker-compose, k8s, helm, CI
```

## License

MIT — 见 `LICENSE`。
