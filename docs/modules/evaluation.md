# Evaluation：指标与验证

Evaluation 读取预测结果和真值，计算平面、循环 ERP、球面和性能指标，不修改任何运行时状态。

| 文件 | 职责 |
|---|---|
| `evaluation/otb_metrics.py` | 平面 / 循环 IoU、成功曲线、AUC、跟踪丢失率 |
| `evaluation/vot360_metrics.py` | 360VOT benchmark 分数，调用官方 toolkit 的指标代码 |
| `evaluation/loss_rate.py` | 360VOT BBox 结果上的丢失率 |
| `evaluation/bootstrap.py` | 按序列重采样的置信区间 |
| `evaluation/comparison.py` | 两次运行的配对比较、硬回归序列检查 |
| `evaluation/score_analysis.py` | 逐帧置信度与 IoU 的关系：相关系数、AUROC、分位数、阈值表、丢失前后的分数 |
| `evaluation/profiler.py` | 命名代码段耗时统计 |
| `tools/benchmark.py` | 360VOT：批量运行各方法并打分，用法见 [Benchmark 数据集](../benchmark.md#复现命令) |
| `tools/run_airsim360_dataset.py` | AirSim360 单序列：运行跟踪、生成伪真值、输出逐帧 IoU 和汇总 |

## 指标定义

### 平面 IoU 与循环 IoU

- `bboxIoU()`：交集面积 / 并集面积，用于不跨缝的普通图像框；
- `circularBBoxIoU()`：ERP 框可能跨缝，先把每个框拆成最多两个水平段，累加各段的交集，再计算并集。

控制器中的 `OverlapRate`（交集 / 较小框面积）用来判断两个预测是否指向同一目标；评估中的 IoU（交集 / 并集）用来衡量预测与真值的吻合程度。两者的阈值不能直接比较。

### 成功曲线与 AUC

`successCurve()` 在 0 到 1 之间的 21 个 IoU 阈值上统计 `IoU > 阈值` 的帧比例；AUC 是这条曲线的梯形积分。`successRate@0.5` 是其中一个工作点，`meanIoU` 是逐帧平均。

### 跟踪丢失率（AirSim360 诊断用）

这是 `otb_metrics.py` 里给 AirSim360 单序列工具用的旧定义，和下文 360VOT 的丢失率不是一回事。

```text
lostFrameCount   = 可见帧中 circular IoU ≤ 1e-12 的帧数
trackingLossRate = lostFrameCount / 参与评估的可见帧数
```

`1e-12` 只用来吸收浮点误差，不会把“IoU 很小”误判为丢失。

## 评测规则

- 第 0 帧是给定的初始化框，不计入任何指标；
- 目标不可见的帧不进入 IoU 和丢失率的分子或分母，但预测序列与真值序列必须逐帧对齐；
- 只报告 meanIoU 不够。至少同时报告：循环 IoU / AUC、跟踪丢失率、球面 IoU、球面中心误差、每帧前向次数、P50 / P95 / P99 延迟；
- 单条序列上的提升不能作为结论，必须在完整的测试集上比较，并同时给出按序列平均和按帧加权的结果。

## 360VOT 指标

对外报告的分数全部来自 `evaluation/vot360_metrics.py`。它不重新实现指标，而是调用放在 `third_party/vot360_toolkit/` 里的官方代码（MIT 许可），只在外面做输入检查和汇总。这样做是因为官方实现有一些不容易从论文里看出来的行为（见 [Benchmark 数据集](../benchmark.md#官方指标实现的几个特点)），自己重写的指标即使“更正确”，分数也无法和别人的结果比较。

```python
from track360.evaluation.vot360_metrics import evaluateVot360

scores = evaluateVot360(groundTruth, results, "bbox")
scores.success          # S_dual（bbox）或 S_sphere（bfov / rbfov）
scores.precision        # P_dual，只有 bbox 有
scores.anglePrecision   # P_angle
scores.perSequence      # 每条序列的分数
```

- `groundTruth` 来自 `Vot360Sequence.groundTruth()`，`results` 是按序列名组织的结果数组，两者都是官方布局；
- 汇总方式与官方脚本相同：**先算每条序列的分数，再对序列取平均**，所以长序列和短序列的权重一样；
- 结果文件必须覆盖序列的全部帧，长度不一致会直接报错；
- 支持 `bbox`、`bfov`、`rbfov`。`rbbox` 的旋转 IoU 依赖一个需要 CUDA 的外部库，没有接入。

## 丢失率（360VOT）

`evaluation/loss_rate.py` 在 BBox 结果上统计，随 `evaluateVot360()` 一起算出（`scores.lossRate`，以及每条序列的 `lostFrames`、`firstLostFrame`）。它不是官方指标，用来区分“框不准”和“跟丢了”。

- **一帧的 IoU**：和 S<sub>dual</sub> 用的是同一个值，即预测框与真值、与左移一个图像宽度的真值两者 IoU 的较大者；
- **丢失**：IoU < 0.1 连续至少 5 帧，这一段的每一帧（含开头的 5 帧）都算丢失帧。不足 5 帧的短暂下降不算。IoU 回到 0.1 及以上，这次丢失就结束；之后要再连续 5 帧才算下一次；
- **目标缺席的帧**：不算丢失帧，也不打断一段连续的丢失；
- **丢失率**：所有序列的丢失帧数之和 ÷ 所有序列的总帧数之和（含目标缺席的帧）。按帧加权，长序列的权重更大，这一点和按序列平均的 S<sub>dual</sub> 不同。

阈值 0.1 和 5 帧是代码常量，定下来就不改：改了之后新旧评测记录的丢失率无法比较。

## 置信区间与两次运行的比较

总分是几十条序列的平均，而单条序列的分数起伏很大，两个方法的差可能只是“碰巧选了这些序列”造成的。`evaluation/bootstrap.py` 用按序列重采样估计这个不确定性：从 N 条序列里有放回地抽 N 条，重算一次指标，重复 10,000 次，取中间 95% 的范围。

`evaluation/comparison.py::compareScores()` 比较两次运行（只用两边都有结果的序列）：

- 给出 S<sub>dual</sub>、P<sub>angle</sub>、丢失率各自的值和区间，以及“候选 − 基准”的差和区间；
- 差值用**配对**方式重采样：每次抽样两个方法用同一批序列，这样两个方法共有的起伏会抵消，区间比各自区间相减要窄；
- **差值的区间不包含 0，才认为提升（或下降）是可信的**；
- 随机种子固定，同样的输入得到同样的区间。

## 硬回归序列

[`configs/splits/360vos_tune_hard.txt`](../../configs/splits/360vos_tune_hard.txt) 列出 tune 集里的 5 条序列。每个改动先只跑这 5 条（约 3 分钟），`hardRegressions()` 检查其中有没有 S<sub>dual</sub> 下降超过 0.02 的；有就停下，不再跑全量。命令见 [Benchmark 数据集](../benchmark.md#比较两次运行)。

跟踪结果是确定性的（同一份代码和配置重跑，结果文件逐字节一致），所以 0.02 的门槛不会被运行间的噪声触发。

本文“指标定义”一节的指标（循环 IoU、AirSim360 的跟踪丢失率等）是项目自己的诊断指标，用于开发中分析问题，不用于对外报告。选型理由和评测协议见 [Benchmark 数据集](../benchmark.md)。

## 回归检查

```bash
pytest
ruff check src tests tools
git diff --check
```
