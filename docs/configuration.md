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
| `geometry` | Geometry | 局部视图尺寸 256×256、`boundarySamplesPerEdge`（局部框每条边回投的采样点数）、FoV 范围 20°–120°、`resampler`（`cpu` / `cuda`） |
| `evaluator` | Controller | `fusionSourceMinConfidence`（参与融合的来源最低分）、`overlapThreshold`、`fusionBoxMode` |
| `motion` | Controller | 球面运动估计：Huber 参数、过程噪声、最大角速度、最大尺度变化率 |
| `tracking` | Controller | `candidateMinScore`、运动窗口长度、每帧最多轮数（2）和视图数上限（12） |
| `recovery` | Controller | 找回目标时的 cubemap / 环形视图布局 |
| `backendTuning` | Controller / Backends | 针对 ARTrackV2 后端的开关和阈值，见下一节 |
| `reproducibility` | Runtime | `seed`：随机种子；`deterministic`：是否启用确定性 cuDNN |
| `visualization` | Visualization | 是否输出中间视图、输出目录、输出哪些阶段 |

## backendTuning

ARTrackV2 的原始分数集中在 0.5 附近，不是校准过的概率，所以 `tracking.candidateMinScore`、`evaluator.fusionSourceMinConfidence` 这类按概率设定的门槛不能直接使用。`backendTuning` 把针对这个后端的调整集中在一处。取值为 `null` 表示“不覆盖”，此时使用右栏说明的回退值。

| 字段 | 默认值 | 作用 |
|---|---|---|
| `acceptAnyCandidate` | `true` | 接受最佳候选作为测量，不再用 `candidateMinScore` 过滤。**为 `true` 时每帧只做一轮搜索** |
| `directMode` | `false` | 不做跨视图融合，直接取最高分候选，且每帧只做一轮 |
| `singleRound` | `false` | 在带门槛的模式（`acceptAnyCandidate: false`）下禁止第二轮 |
| `adaptiveViewCount` | `false` | 按目标角尺寸在单视图和四视图之间切换（目标 ≥ 75° 或占画面 ≥ 20% 时用四视图） |
| `singleView` | `false` | 始终只用一个以预测中心为中心的视图 |
| `singleViewHorizontalFovCapDeg` / `singleViewVerticalFovCapDeg` | `90.0` | 单视图模式的视场上限；`null` 表示只受 `geometry.maxFovDeg` 限制 |
| `fourViewFovCapDeg` | `null` | 四视图模式的视场上限（≥ 30）；`null` 表示使用 `geometry.maxFovDeg` |
| `fullViewSearch` | `false` | 把整个局部视图缩放后作为搜索区域，跳过 ARTrackV2 自己的 4 倍搜索裁剪 |
| `useMotionScore` | `false` | 用“外观 + 运动”加权得到 SingleScore；为 `false` 时只用外观分。没有校准产物时运动权重为 0，此开关不影响结果 |
| `templateFovScale` | `2.5` | 模板视图视场相对目标角尺寸的倍数（≥ 1） |
| `onlineTemplate` | `true` | 保留第 0 帧 anchor 的同时允许更新 recent / stable 模板 |
| `templateMinConfidence` | `0.515` | 允许更新模板的最低分 |
| `allowSingleViewTemplate` | `true` | 只有单个视图支持时也允许更新模板；为 `false` 时要求分数 ≥ 0.84 |
| `holdWeakBox` | `true` | 测量未被接受且目标面积 ≥ 画面的 10% 时，保持上一帧的框 |
| `fusionSourceMinConfidence` | `0.35` | 参与融合的来源最低分；`null` 时回退为 0（`acceptAnyCandidate: true`）或 `evaluator.fusionSourceMinConfidence` |
| `fusionOverlap` | `0.45` | 融合所需的最小重叠率；`null` 时使用代码常量 0.70 |
| `fusionBoxMode` | `null` | 融合框的生成方式；`null` 时使用 `evaluator.fusionBoxMode` |

`core/config.py::BackendTuningConfig` 的 dataclass 默认值与 `configs/default.yaml` 相同，有测试保证两者不会不一致。因此不传配置、直接构造 `TrackControllerImpl` / `StateEvaluator` / `TemplatePolicy` / `RecoveryPlanner` 时，得到的也是默认配置的行为。

### configs/tests/legacy_off.yaml

这份配置把 `backendTuning` 的所有开关设为“关闭”：带门槛的两轮搜索、不更新模板、不保持弱框、使用 `evaluator` 中的融合门槛。它对应迁移前“没有设置任何环境变量”时控制器的行为，**只用于测试**两轮搜索、融合门槛等在默认配置下不会触发的逻辑。

## 关键约束

- `geometry.maxFovDeg` 必须为 120：它同时是 cubemap 每个面的视场和动态视图的上限；
- `tracking.maxAttemptsPerFrame` 固定为 2；`tracking.maxViewsPerFrameTotal` 至少为 12，以容纳找回路径的 6 + 4 个视图；
- 提供 `scoring.calibrationArtifact` 时，产物中的 `candidateMinScore` 和 `fusionSourceMinConfidence` 必须与 YAML 一致，且权重文件的 SHA-256 必须匹配。

## 新增参数的流程

1. 在 `core/config.py` 中给对应的 dataclass 加字段和校验；
2. 把字段名加入 `_section()` 的允许集合，并在构造处解析；
3. 更新 `configs/default.yaml` 和 `configs/tests/legacy_off.yaml`；
4. 更新 `tests/unit/test_core_config.py` 和本文档。

新参数不允许通过环境变量或代码常量绕过配置文件。

## 运行记录

`track360 track` 和 `track360 airsim360` 每次运行都会在结果文件旁写一个 `<结果文件名>.run.json`，内容包括：git commit 和工作区是否干净、配置哈希、最终生效的完整配置、权重文件的 SHA-256、Python / PyTorch / CUDA / cuDNN / NumPy / OpenCV 版本、GPU 型号和驱动版本。

配置哈希只覆盖影响结果的参数，不包含因机器而异的路径（权重路径、可视化输出目录），所以同一份配置在不同机器上的哈希相同。
