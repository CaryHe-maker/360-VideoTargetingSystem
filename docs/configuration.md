# 配置说明

所有运行参数都来自 YAML 配置文件（默认 `configs/default.yaml`），由 `core/config.py::loadConfig()` 严格校验。代码中不读取任何环境变量：同一份配置文件加同一个 commit，得到的就是同一组参数。

## YAML 加载规则

- 每个配置段的字段集合必须**完全匹配**：多一个字段或少一个字段都会抛出 `ConfigError`，所以拼写错误会立即暴露；
- 标量类型、取值范围和跨字段约束都在加载时检查；
- 相对路径以**配置文件所在目录**为基准解析，与进程当前目录无关；
- 角度在 YAML 中用“度”，加载后在内部统一转换为弧度。

## 配置段

| 配置段 | 归属模块 | 主要字段 |
|---|---|---|
| `model` | Backends | `variant`（当前只支持 `artrackv2_b_256`）、`weights`、`precision`（当前只支持 `fp32`） |
| `scoring` | Controller | `calibrationArtifact`：可选的分数校准 JSON；`requireCheckpointHashMatch`：校准产物必须与权重的 SHA-256 绑定 |
| `geometry` | Geometry | 局部视图尺寸 256×256、`boundarySamplesPerEdge`（局部框每条边回投的采样点数）、FoV 范围 20°–120°、`resampler`（`opencv`：默认，`cv2.remap` 取色；`cpu`：双精度的参考实现，慢约 17 ms / 帧，像素值最多差 1 级） |
| `motion` | Controller | 球面运动估计：Huber 参数、过程噪声、最大角速度、最大尺度变化率 |
| `tracking` | Controller | `candidateMinScore`（带门槛模式下接受测量的最低分）、运动窗口长度、没有框时输出范围的放大系数 |
| `backendTuning` | Controller / Backends | 针对 ARTrackV2 后端的开关和阈值，见下一节 |
| `reproducibility` | Runtime | `seed`：随机种子；`deterministic`：是否启用确定性 cuDNN |
| `visualization` | Visualization | 是否输出中间视图、输出目录、输出哪些阶段 |

## backendTuning

`backendTuning` 把针对 ARTrackV2 后端的开关和阈值集中在一处。后端的分数是模型对“预测框与真值的 IoU”的估计。`tracking.candidateMinScore` 的取值是早先按帧级用法定的，只在 `acceptAnyCandidate: false` 时起作用，没有重新确定过。

| 字段 | 默认值 | 作用 |
|---|---|---|
| `acceptAnyCandidate` | `true` | 只要这一帧有框就作为测量接受；为 `false` 时分数低于 `candidateMinScore` 的框不被接受 |
| `viewHorizontalFovCapDeg` / `viewVerticalFovCapDeg` | `90.0` | 透视搜索视图的视场上限；`null` 表示只受 `geometry.maxFovDeg` 限制。视图是正方形，取两者中较小的 |
| `alignedMinFovDeg` | `2.0` | 搜索视图视场的下限（视图是目标平均尺寸的 4 倍，见 [Controller](modules/controller.md)） |
| `sphericalSearch` | `true` | 大目标的视图改用球面采样（以目标为中心的局部 ERP），见 [Controller](modules/controller.md#大目标球面视图)。需要 `alignedSearch: true`。`false` 时透视视场封顶在上面的上限，超出部分补黑边 |
| `sphericalSearchFovDeg` | `120.0` | 搜索区域（目标平均角尺寸的 4 倍）达到这个角度时切换到球面视图。360VOT 论文用 90°；本项目在 tune 集上 90° 和 120° 没有可分辨的差别，取 120° 只是为了少偏离透视路径（[评测记录](evaluation-log.md) E011） |
| `predictiveSearch` | `true` | 搜索视图的中心和大小取运动模型对这一帧的预测。`false` 时直接取上一帧提交的结果，和上游跟踪器自己的循环一致。66 条序列上关掉后 S<sub>dual</sub> −0.023 [−0.061, +0.014]，没有采纳（[评测记录](evaluation-log.md) E012） |
| `useMotionScore` | `false` | 用“外观 + 运动”加权得到 SingleScore；为 `false` 时只用外观分。没有校准产物时运动权重为 0，此开关不影响结果 |
| `templateFovScale` | `2.5` | 模板视图视场相对目标角尺寸的倍数（≥ 1） |
| `lossHandling` | `false` | 丢失处理的总开关：可疑帧不更新外观记忆，连续可疑后扫描并跳转，见 [Controller](modules/controller.md#丢失处理)。需要 `models/hub/` 下的外观模型权重 |
| `verifierModel` | `dinov2` | 计算外观相似度的模型：`dinov2`（ViT-S/14）、`dino`（ViT-S/16）或 `resnet18` |
| `stateBackendWeight` / `stateAppearanceWeight` / `stateMotionWeight` | `0.40` / `0.55` / `0.05` | 状态分数里后端分、外观分、运动分的权重，见 [Controller](modules/controller.md#状态机)。没有外观分时另外两个按比例归一。数值来自 E018 |
| `motionOffsetScale` / `motionSizeScale` | `0.5` / `0.1` | 运动分的两个衰减宽度：框离预测位置的距离（以目标尺寸为单位）和尺寸比的对数 |
| `uncertainScore` | `0.42` | 状态分数低于它的帧不可信（`UNCERTAIN`）。外观分只对第 0 帧模板计算；这个值在自由运行的数据上约误判 0.8% 的好帧、抓到 28% 的丢失帧（E019、E020） |
| `lostAfterFrames` | `4` | 连续多少帧不可信后状态变为 `LOST`（丢失处理开启时开始扫描） |
| `scanViewsPerFrame` | `4` | 扫描时每帧多取几个视图，每个多一次前向 |
| `stateRule` | `fused` | 状态怎么判：`fused` 用上面的融合分；`relative` 把后端分和模板相似度各自和本序列可信帧的中位数比，取两个相对偏离的平均（需要 `lossHandling`）。`relativeGate`（0.05）是一帧进入基线允许的最大落差，`relativeEnterDeviation`（−0.53）是判为不可信的门槛。见 [Controller](modules/controller.md#状态机) 和评测记录 E027、E030 |
| `stateLatch` / `latchReleaseMargin` / `releaseFrames` | `false` / `0.20` / `3` | 融合规则下，进入不可信后是否锁住：状态分数要回到 `uncertainScore + latchReleaseMargin` 以上并保持 `releaseFrames` 帧才解除。`releaseFrames` 也是分开规则解除可疑所需的帧数 |
| `lossActions` | `jump` | 丢失后做什么：`none` 只判定和记录；`jump` 搜索并跳转 |
| `scanMode` | `tiles` | 丢失后怎么搜索：`tiles` 用正常大小的视图由近到远分块扫描；`zoom` 先在原视图上做一次无状态前向，之后取一个放大的视图定位、再在它指的位置取正常大小的视图。`zoomCentre`（`trusted` / `current`）、`zoomFirstScale` / `zoomMidScale` / `zoomLastScale`（2 / 0 / 4）和 `zoomMidAfterFrames` / `zoomLastAfterFrames`（10 / 20）决定放大视图的中心和倍数，`zoomInPlace` 是否先原地重检（E029、E030） |
| `samePlaceAction` / `zoomSpread` / `crossScore` / `crossCheck` | `stay` / `1` / `0` / `false` | 试验过的搜索选项，配置文件里可以不写（E033）。`samePlaceAction`：候选落在跟踪器原框上时不跳转（`stay`，原地重检一致时确认）还是照样跳转（`jump`，基线 2 用的）；`zoomSpread`：最后阶段一次取几个半重叠的放大视图（1、2、4）；`crossScore`：两个视图指向同一处的候选从这个分数起采纳（0 关闭）；`crossCheck`：分数在 `crossScore` 和 `reacquireScore` 之间的候选再取一个偏移的正常视图确认。`zoomCentre` 另有 `extrapolated`（从可信帧按运动外推）。后三项和外推中心都没有带来提升 |
| `scanBudgetPerFrame` / `scanBudgetBurst` | `0.0` / `40.0` | 搜索的前向预算：每帧积攒的次数和最多存的次数；0 表示不限（E023） |
| `reacquireSimilarity` / `reacquireMargin` / `reacquireScore` | `0.45` / `0.15` / `0.70` | 扫描候选被采纳的条件：相似度的下限、比当前框高出的幅度、无状态前向分数的下限。分数是区分真假候选的主要信号（E017） |
| `holdWeakBox` | `true` | 测量未被接受且目标面积 ≥ 画面的 10% 时，保持上一帧的框 |

`core/config.py::BackendTuningConfig` 的 dataclass 默认值与 `configs/default.yaml` 相同，有测试保证两者不会不一致。因此不传配置、直接构造 `TrackControllerImpl` / `StateEvaluator` / `TemplatePolicy` / `ViewPlanner` 时，得到的也是默认配置的行为。

## 已删除的字段

多视图、两轮搜索和融合删除后，下面这些字段不再存在，旧配置文件里带着它们会在加载时报“unknown”：

- 整个 `evaluator` 配置段和整个 `recovery` 配置段；
- `tracking` 中的 `scaleClusterTolerance`、`guardYawStepDeg`、`minViewsForCommit`、`sameFrameEscalationEnabled`、`maxAttemptsPerFrame`、`maxViewsPerFrameTotal`、`uncertainFovScale`、`reacquireCooldownFrames`；
- `backendTuning` 中的 `directMode`、`singleRound`、`adaptiveViewCount`、`singleView`、`fourViewFovCapDeg`、`allowSingleViewTemplate`、`fusionSourceMinConfidence`、`fusionOverlap`、`fusionBoxMode`；
- `backendTuning.singleViewHorizontalFovCapDeg` / `singleViewVerticalFovCapDeg` 改名为 `viewHorizontalFovCapDeg` / `viewVerticalFovCapDeg`。

`geometry.maxFovDeg` 不再强制为 120（那是 cubemap 每个面的视场）。测试专用的 `configs/tests/legacy_off.yaml` 也已删除。

## 分数校准产物

提供 `scoring.calibrationArtifact` 时，产物的格式必须是 `track360.score-calibration.v2`，其中的 `thresholds.candidateMinScore` 必须与 YAML 一致，权重文件的 SHA-256 必须匹配。v1 格式的产物（带 `fusionSourceMinConfidence`）不再被接受。

## 新增参数的流程

1. 在 `core/config.py` 中给对应的 dataclass 加字段和校验；
2. 把字段名加入 `_section()` 的允许集合，并在构造处解析；
3. 更新 `configs/default.yaml`；
4. 更新 `tests/unit/test_core_config.py` 和本文档。

新参数不允许通过环境变量或代码常量绕过配置文件。

## 运行记录

`track360 track` 和 `track360 airsim360` 每次运行都会在结果文件旁写一个 `<结果文件名>.run.json`，内容包括：git commit 和工作区是否干净、配置哈希、最终生效的完整配置、权重文件的 SHA-256、Python / PyTorch / CUDA / cuDNN / NumPy / OpenCV 版本、GPU 型号和驱动版本。

配置哈希只覆盖影响结果的参数，不包含因机器而异的路径（权重路径、可视化输出目录），所以同一份配置在不同机器上的哈希相同。配置字段变了，哈希也跟着变：删除多视图之前记录的哈希不能和之后的直接比较。

丢失找回已冻结，两条基线的配置和数字见 [baselines.md](baselines.md)：默认配置是基线 1，`configs/loss_handling.yaml` 是基线 2。
