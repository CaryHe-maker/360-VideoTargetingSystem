# Evaluation：指标与验证

Evaluation 读取预测结果和真值，计算平面、循环 ERP、球面和性能指标，不修改任何运行时状态。

| 文件 | 职责 |
|---|---|
| `evaluation/otb_metrics.py` | 平面 / 循环 IoU、成功曲线、AUC、跟踪丢失率 |
| `evaluation/spherical_metrics.py` | 球面中心误差、BFoV 球面 IoU |
| `evaluation/vot360_metrics.py` | 360VOT benchmark 分数，调用官方 toolkit 的指标代码 |
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

### 跟踪丢失率

```text
lostFrameCount   = 可见帧中 circular IoU ≤ 1e-12 的帧数
trackingLossRate = lostFrameCount / 参与评估的可见帧数
```

`1e-12` 只用来吸收浮点误差，不会把“IoU 很小”误判为丢失。

### 球面中心误差

把两个 BFoV 中心转为单位向量，点积裁剪到 [-1, 1] 后取 arccos，得到大圆角距离。在 yaw 跨越 ±180° 时不会出现假的大误差。

### 球面 BFoV IoU

`bfovSphericalIoU()` 在 yaw / pitch 网格上采样，判断每个点是否落在两个 BFoV 内。每个样本按 `cos(pitch)` 加权，补偿 ERP 在两极的过采样；加权交集除以加权并集。采样密度是函数参数，对比实验时必须固定。

## 评测规则

- 第 0 帧是给定的初始化框，不计入任何指标；
- 目标不可见的帧不进入 IoU 和丢失率的分子或分母，但预测序列与真值序列必须逐帧对齐；
- 只报告 meanIoU 不够。至少同时报告：循环 IoU / AUC、跟踪丢失率、球面 IoU、球面中心误差、每帧平均视图数和前向次数、P50 / P95 / P99 延迟；
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

本文前面几节的指标（循环 IoU、跟踪丢失率、`bfovSphericalIoU` 等）是项目自己的诊断指标，用于开发中分析问题，不用于对外报告。选型理由和评测协议见 [Benchmark 数据集](../benchmark.md)。

## 回归检查

```bash
pytest
ruff check src tests tools
git diff --check
```
