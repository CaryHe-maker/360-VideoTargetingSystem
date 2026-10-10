# 配置说明

所有运行参数都来自 YAML 配置文件，由 `core/config.py::loadConfig()` 严格校验。代码中不读取任何环境变量：同一份配置文件加同一个 commit，得到的就是同一组参数。

## 随仓库提供的配置

| 文件 | 用途 | 与默认配置的差别 |
|---|---|---|
| [`configs/default.yaml`](../configs/default.yaml) | 基线 1：每帧一次前向，结果是参考结果 | — |
| [`configs/loss_handling.yaml`](../configs/loss_handling.yaml) | 基线 2：丢失后重新寻找目标（实验功能） | `lossHandling: true`、`scanBudgetPerFrame: 0.5` |
| [`configs/fast_tf32.yaml`](../configs/fast_tf32.yaml) | 基线 1 的速度档 | `model.precision: tf32` |
| [`configs/loss_handling_tf32.yaml`](../configs/loss_handling_tf32.yaml) | 基线 2 的速度档 | 基线 2 加 `model.precision: tf32` |

两条基线和速度档的数字见 [baselines.md](baselines.md)。命令行和 Python API 用 `--preset default|loss_handling` 和 `--precision fp32|tf32` 选择，四种组合都可以，不必自己写配置文件。

## YAML 加载规则

- 每个配置段的字段集合必须**完全匹配**：多一个字段或少一个字段都会抛出 `ConfigError`，所以拼写错误会立即暴露；
- 标量类型、取值范围和跨字段约束都在加载时检查；
- 相对路径以**配置文件所在目录**为基准解析，与进程当前目录无关；
- 角度在 YAML 中用“度”，加载后在内部统一转换为弧度；
- `schemaVersion` 现在是 `2`。版本 1 的配置文件（带下文“已删除的字段”）不能加载。

## 配置段

| 配置段 | 归属模块 | 主要字段 |
|---|---|---|
| `model` | Backends | `variant`（当前只支持 `artrackv2_b_256`）、`weights`、`precision` |
| `geometry` | Geometry | 局部视图尺寸 256×256、`boundarySamplesPerEdge`（局部框每条边回投的采样点数）、FoV 范围 20°–120° |
| `motion` | Controller | 球面运动估计：Huber 参数、过程噪声、最大角速度、最大尺度变化率 |
| `tracking` | Controller | 运动窗口长度、没有框时输出范围的放大系数、最长预测帧数 |
| `backendTuning` | Controller / Backends | 视图、状态判定和丢失处理的参数，见下一节 |
| `reproducibility` | Runtime | `seed`：随机种子；`deterministic`：是否启用确定性 cuDNN |
| `visualization` | Visualization | 是否输出中间视图、输出目录、输出哪些阶段 |

### model.precision

| 取值 | 说明 |
|---|---|
| `fp32` | 基准。同一台机器上结果逐字节可复现 |
| `tf32` | 打开显卡上的 TensorFloat-32 矩阵乘法。RTX 4060 上网络前向从 18.9 ms 到 13.7 ms，整体约快 9%；框的位置相差约 0.001 像素，跟踪结果在末位上有差别，精度与 `fp32` 分不出高低（评测日志 E039）。只在 NVIDIA Ampere 及之后的显卡（RTX 30 系起）上有效，其他设备上等同于 `fp32`。它是进程级的开关，同一进程里的其他模型也受影响 |

半精度（FP16）试过，没有采用：这个模型的一次前向由几百个小运算组成，瓶颈在 CPU 发起运算，半精度在完整流程里不比 `tf32` 快，数值偏差更大。

## backendTuning

### 视图

| 字段 | 默认值 | 作用 |
|---|---|---|
| `viewHorizontalFovCapDeg` / `viewVerticalFovCapDeg` | `90.0` | 透视搜索视图的视场上限；`null` 表示只受 `geometry.maxFovDeg` 限制。视图是正方形，取两者中较小的 |
| `alignedMinFovDeg` | `2.0` | 搜索视图视场的下限（视图是目标平均尺寸的 4 倍，见 [Controller](modules/controller.md)） |
| `sphericalSearch` | `true` | 大目标的视图改用球面采样（以目标为中心的局部 ERP），见 [Controller](modules/controller.md#大目标球面视图)。`false` 时一律用透视视图 |
| `sphericalSearchFovDeg` | `120.0` | 搜索区域达到这个角度时切换到球面视图 |
| `templateFovScale` | `2.5` | 模板视图视场相对目标角尺寸的倍数（≥ 1） |

### 状态判定

每一帧的框总是作为这一帧的测量接受；状态只表示这个框可不可信。

| 字段 | 默认值 | 作用 |
|---|---|---|
| `stateBackendWeight` / `stateAppearanceWeight` / `stateMotionWeight` | `0.40` / `0.55` / `0.05` | 状态分数里后端分、外观分、运动分的权重。丢失处理关闭时没有外观分，剩下两项按比例归一 |
| `motionOffsetScale` / `motionSizeScale` | `0.5` / `0.1` | 运动分的两个衰减宽度：框离预测位置的距离（以目标尺寸为单位）和尺寸比的对数 |
| `uncertainScore` | `0.42` | 丢失处理关闭时，状态分数低于它的帧报告为 `UNCERTAIN`。这时状态只是报告，不影响跟踪 |
| `lostAfterFrames` | `4` | 连续多少帧不可信后状态变为 `LOST` |

### 丢失处理（实验功能）

`lossHandling: true` 时，状态由相对量规则判定：后端分和对第 0 帧模板的外观相似度，各自和本序列到目前为止可信帧的中位数相比，取两个相对偏离的平均。可疑帧不更新后端的外观记忆；状态为 `LOST` 时开始搜索。做法和依据见 [Controller](modules/controller.md#丢失处理) 和 [baselines.md](baselines.md)。需要 `models/hub/` 下的 DINOv2 权重。

| 字段 | 默认值 | 作用 |
|---|---|---|
| `lossHandling` | `false` | 丢失处理的总开关 |
| `lossActions` | `jump` | 丢失后做什么：`jump` 搜索并跳转；`none` 只判定和记录，跟踪结果与关闭丢失处理时相同（分析用） |
| `relativeGate` | `0.05` | 一帧的分数比中位数低不超过这个比例时，计入可信帧的历史 |
| `relativeEnterDeviation` | `-0.53` | 两个相对偏离的平均低于它时进入不可信 |
| `releaseFrames` | `3` | 进入不可信后，偏离回到门槛的一半以内并保持这么多帧才解除 |
| `scanBudgetPerFrame` / `scanBudgetBurst` | `0.0` / `40.0` | 搜索的前向预算：每帧积攒的次数和最多存的次数；0 表示不限。基线 2 用 0.5，平均每帧 1.16 次前向 |
| `reacquireScore` | `0.70` | 搜索到的框，不带记忆的前向分数达到它才跳转。这是唯一能区分真假候选的信号（E026、E034、E036） |
| `zoomFirstScale` / `zoomLastScale` / `zoomLastAfterFrames` | `2.0` / `4.0` / `20` | 搜索视图相对正常视图的倍数，以及丢失多少帧之后改用较大的那个 |

搜索的流程是固定的：判为丢失的第一帧，在原视图上做一次不带记忆的前向；之后每次取一个放大的视图（以最后可信位置为中心），在它指的位置再取一个正常大小的视图，后者给出的框是候选。

`core/config.py::BackendTuningConfig` 的 dataclass 默认值与 `configs/default.yaml` 相同，有测试保证两者不会不一致。

## 已删除的字段

**发布前的清理（schemaVersion 2，2026-10-11）**。这些选项对应的做法或者已经确定、写进了代码，或者试过没有采用：

| 字段 | 现在的行为 |
|---|---|
| 整个 `scoring` 配置段（分数校准产物） | 没有校准。后端分数原样使用 |
| `geometry.resampler` | 取视图一律用 `cv2.remap`；双精度的参考实现只留给测试 |
| `tracking.candidateMinScore`、`backendTuning.acceptAnyCandidate` | 有框就接受 |
| `backendTuning.holdWeakBox` | 这一帧没有框、且目标面积不小于画面的 10% 时，保持上一帧的框 |
| `backendTuning.predictiveSearch` | 搜索视图的中心和大小取运动模型的预测 |
| `backendTuning.useMotionScore` | 分数只用后端分 |
| `backendTuning.verifierModel` | 外观验证器是 DINOv2 ViT-S/14 |
| `backendTuning.stateRule`、`stateLatch`、`latchReleaseMargin` | 丢失处理开启时用相对量规则，关闭时用融合分且不加锁 |
| `backendTuning.scanMode`、`scanViewsPerFrame` | 搜索是“先放大后细看”；分块扫描已删除 |
| `backendTuning.zoomInPlace`、`zoomCentre`、`zoomMidScale`、`zoomMidAfterFrames` | 第一帧原地重检；放大视图以最后可信位置为中心；倍数只有两级 |
| `backendTuning.reacquireSimilarity`、`reacquireMargin` | 采纳候选只看 `reacquireScore` |
| `backendTuning.samePlaceAction`、`zoomSpread`、`crossScore`、`crossCheck` | 候选落在跟踪器原框上也跳转；没有多视图一致性（E033） |

**更早删除的**（多视图、两轮搜索和融合）：整个 `evaluator` 和 `recovery` 配置段；`tracking` 中的 `scaleClusterTolerance`、`guardYawStepDeg`、`minViewsForCommit`、`sameFrameEscalationEnabled`、`maxAttemptsPerFrame`、`maxViewsPerFrameTotal`、`uncertainFovScale`、`reacquireCooldownFrames`；`backendTuning` 中的 `directMode`、`singleRound`、`adaptiveViewCount`、`singleView`、`fourViewFovCapDeg`、`allowSingleViewTemplate`、`fusionSourceMinConfidence`、`fusionOverlap`、`fusionBoxMode`、`sequenceModel`、`fullViewSearch`、`alignedSearch`、`onlineTemplate`、`templateMinConfidence`。

## 新增参数的流程

1. 在 `core/config.py` 中给对应的 dataclass 加字段和校验；
2. 把字段名加入加载处的允许集合并解析；
3. 更新 `configs/` 下的四个配置文件；
4. 更新 `tests/unit/test_core_config.py` 和本文档。

新参数不允许通过环境变量或代码常量绕过配置文件。
