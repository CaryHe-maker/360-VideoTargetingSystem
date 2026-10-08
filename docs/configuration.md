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
| `geometry` | Geometry | 局部视图尺寸 256×256、`boundarySamplesPerEdge`（局部框每条边回投的采样点数）、FoV 范围 20°–120°、`resampler`（`opencv`：默认，`cv2.remap` 取色；`cpu`：双精度的参考实现，慢约 17 ms / 帧，像素值最多差 1 级；`cuda`） |
| `motion` | Controller | 球面运动估计：Huber 参数、过程噪声、最大角速度、最大尺度变化率 |
| `tracking` | Controller | `candidateMinScore`（带门槛模式下接受测量的最低分）、`stableFramesBeforeUpdate`（stable 模板的更新周期）、运动窗口长度、没有框时输出范围的放大系数 |
| `backendTuning` | Controller / Backends | 针对 ARTrackV2 后端的开关和阈值，见下一节 |
| `reproducibility` | Runtime | `seed`：随机种子；`deterministic`：是否启用确定性 cuDNN |
| `visualization` | Visualization | 是否输出中间视图、输出目录、输出哪些阶段 |

## backendTuning

`backendTuning` 把针对 ARTrackV2 后端的开关和阈值集中在一处。分数的含义取决于 `sequenceModel`：序列级用法下它是模型对“预测框与真值的 IoU”的估计；帧级用法下它集中在 0.5 附近、与 IoU 无关（[评测记录](evaluation-log.md) E008）。`tracking.candidateMinScore` 和 `templateMinConfidence` 的现有取值都是按帧级用法定的，序列级用法下还没有重新确定。

| 字段 | 默认值 | 作用 |
|---|---|---|
| `sequenceModel` | `true` | 按序列级模型运行 ARTrackV2：喂入前 7 帧的轨迹，使用模型自己更新的外观特征，见 [Backends](modules/backends.md#这份权重是序列级模型)。`false` 是 2026-10-07 之前的帧级用法，只用于对照 |
| `acceptAnyCandidate` | `true` | 只要这一帧有框就作为测量接受；为 `false` 时分数低于 `candidateMinScore` 的框不被接受 |
| `viewHorizontalFovCapDeg` / `viewVerticalFovCapDeg` | `90.0` | 透视搜索视图的视场上限；`null` 表示只受 `geometry.maxFovDeg` 限制。`alignedSearch` 下视图是正方形，取两者中较小的 |
| `fullViewSearch` | `false` | 只在 `alignedSearch: false` 时有意义：把整个局部视图缩放后作为搜索区域，跳过 ARTrackV2 自己的 4 倍搜索裁剪 |
| `alignedSearch` | `true` | 搜索区域和 ARTrackV2 的训练裁剪对齐：正方形视图、边长为目标平均尺寸的 4 倍，见 [Controller](modules/controller.md#视图规划)。`false` 是旧的取法，只用于对照。不能和 `fullViewSearch` 同时打开，也不支持 `geometry.resampler: cuda` |
| `alignedMinFovDeg` | `2.0` | `alignedSearch` 下视图视场的下限，代替 `geometry.minFovDeg` |
| `sphericalSearch` | `true` | 大目标的视图改用球面采样（以目标为中心的局部 ERP），见 [Controller](modules/controller.md#大目标球面视图)。需要 `alignedSearch: true`。`false` 时透视视场封顶在上面的上限，超出部分补黑边 |
| `sphericalSearchFovDeg` | `120.0` | 搜索区域（目标平均角尺寸的 4 倍）达到这个角度时切换到球面视图。360VOT 论文用 90°；本项目在 tune 集上 90° 和 120° 没有可分辨的差别，取 120° 只是为了少偏离透视路径（[评测记录](evaluation-log.md) E011） |
| `predictiveSearch` | `true` | 搜索视图的中心和大小取运动模型对这一帧的预测。`false` 时直接取上一帧提交的结果，和上游跟踪器自己的循环一致。66 条序列上关掉后 S<sub>dual</sub> −0.023 [−0.061, +0.014]，没有采纳（[评测记录](evaluation-log.md) E012） |
| `useMotionScore` | `false` | 用“外观 + 运动”加权得到 SingleScore；为 `false` 时只用外观分。没有校准产物时运动权重为 0，此开关不影响结果 |
| `templateFovScale` | `2.5` | 模板视图视场相对目标角尺寸的倍数（≥ 1） |
| `onlineTemplate` | `true` | 保留第 0 帧 anchor 的同时允许更新 recent / stable 模板。只在 `sequenceModel: false` 时起作用 |
| `templateMinConfidence` | `0.515` | 允许更新模板的最低分 |
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
