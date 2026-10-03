# 配置说明

运行参数来自两个地方：

1. **YAML 配置文件**（默认 `configs/default.yaml`），由 `core/config.py::loadConfig()` 严格校验；
2. **环境变量**：ARTrackV2 后端的一组调参开关，暂时还没有迁入 YAML（见本文最后一节）。

## YAML 加载规则

- 每个配置段的字段集合必须**完全匹配**：多一个字段或少一个字段都会抛出 `ConfigError`，所以拼写错误会立即暴露；
- 标量类型、取值范围和跨字段约束都在加载时检查；
- 相对路径以**配置文件所在目录**为基准解析，与进程当前目录无关；
- 角度在 YAML 中用“度”，加载后在内部统一转换为弧度。

## 配置段

| 配置段 | 归属模块 | 主要字段 |
|---|---|---|
| `model` | Backends | `variant`（当前只支持 `artrackv2_b_256`）、`weights`、`precision` |
| `scoring` | Controller | `calibrationArtifact`：可选的分数校准 JSON；`requireCheckpointHashMatch`：校准产物必须与权重的 SHA-256 绑定 |
| `geometry` | Geometry | 局部视图尺寸 256×256、`boundarySamplesPerEdge`（局部框每条边回投的采样点数）、FoV 范围 20°–120° |
| `decisionGate` | Controller | 兼容字段，当前生产路径不使用 |
| `evaluator` | Controller | `fusionSourceMinConfidence`（参与融合的来源最低分）、`overlapThreshold`、`fusionBoxMode` |
| `motion` | Controller | 球面运动估计：Huber 参数、过程噪声、最大角速度、最大尺度变化率 |
| `tracking` | Controller | `candidateMinScore`、运动窗口长度、每帧最多轮数（2）和视图数上限（12） |
| `recovery` | Controller | 找回目标时的 cubemap / 环形视图布局 |
| `runtime` | Runtime | 队列容量；当前逐帧循环是顺序执行的，这些值只被校验，不影响运行 |
| `visualization` | Visualization | 是否输出中间视图、输出目录、输出哪些阶段 |

## 关键约束

- `geometry.maxFovDeg` 必须为 120：它同时是 cubemap 每个面的视场和动态视图的上限；
- `tracking.maxAttemptsPerFrame` 固定为 2；`tracking.maxViewsPerFrameTotal` 至少为 12，以容纳找回路径的 6 + 4 个视图；
- 提供 `scoring.calibrationArtifact` 时，产物中的 `candidateMinScore` 和 `fusionSourceMinConfidence` 必须与 YAML 一致，且权重文件的 SHA-256 必须匹配。

## 新增参数的流程

1. 在 `core/config.py` 中给对应的 dataclass 加字段和校验；
2. 把字段名加入 `_section()` 的允许集合，并在构造处解析；
3. 更新 `configs/default.yaml`；
4. 更新 `tests/unit/test_core_config.py` 和本文档。

## 环境变量（待迁移）

`runtime/driver.py::buildRuntime()` 在 `model.variant` 为 `artrackv2_b_256` 时，会用 `os.environ.setdefault` 写入下面这些默认值，再由各模块读取。已经在外部设置的值不会被覆盖，因此可以用环境变量临时调参。

| 变量 | 后端默认值 | 读取位置 | 作用 |
|---|---|---|---|
| `TRACK360_ARTRACK_ACCEPT_ANY` | `1` | `controller/state_evaluator.py` | 取消融合来源的最低分门槛 |
| `TRACK360_ARTRACK_ADAPTIVE` | `0` | 视图规划、评估器 | 按目标角尺寸自动选择单视图或四视图 |
| `TRACK360_ARTRACK_SINGLE_ROUND` | `0` | 评估器 | 每帧只做一轮搜索 |
| `TRACK360_ARTRACK_DISABLE_MOTION` | `1` | `runtime/driver.py` | 不使用运动分参与 SingleScore |
| `TRACK360_ARTRACK_SINGLE_FOV_DEG` | `90` | `controller/recovery_planner.py` | 单视图模式的视场上限 |
| `TRACK360_ARTRACK_TEMPLATE_FOV_SCALE` | `2.5` | `controller/track_controller.py` | 模板视图视场相对目标的倍数 |
| `TRACK360_ARTRACK_HOLD_WEAK` | `1` | 跟踪控制器 | 弱候选时保持上一个可信框 |
| `TRACK360_ARTRACK_TEMPLATE_MIN_CONF` | `0.515` | `controller/template_policy.py` | 允许更新 recent 模板的最低分 |
| `TRACK360_ARTRACK_ALLOW_SINGLE_TEMPLATE` | `1` | 模板策略 | 允许只有单个来源时更新模板 |
| `TRACK360_ARTRACK_ONLINE_TEMPLATE` | `1` | 模板策略 | 保留第 0 帧 anchor 的同时启用 recent 模板更新 |
| `TRACK360_ARTRACK_FUSION_SOURCE_MIN` | `0.35` | 评估器 | 融合来源最低分（覆盖 YAML 中的值） |
| `TRACK360_ARTRACK_FUSION_OVERLAP` | `0.45` | 评估器 | 融合所需的最小重叠率（覆盖代码常量 0.70） |
| `TRACK360_GPU_GEOMETRY` | `0` | `runtime/driver.py` | 设为 `1` 时使用 CUDA 几何重采样 |

另外还有几个没有默认值、只用于实验的开关：`TRACK360_ARTRACK_FULL_VIEW`、`TRACK360_ARTRACK_SINGLE_VIEW`、`TRACK360_ARTRACK_SINGLE_HFOV_DEG`、`TRACK360_ARTRACK_SINGLE_VFOV_DEG`、`TRACK360_ARTRACK_FOV_CAP_DEG`、`TRACK360_ARTRACK_DIRECT`、`TRACK360_ARTRACK_FUSION_BOX_MODE`，以及性能统计开关 `TRACK360_PROFILE`。

> **注意**：这些变量只在 `buildRuntime()` 中设置默认值。单元测试直接构造控制器、不经过 `buildRuntime()`，因此测试覆盖的是这些开关都关闭时的行为，与实际运行时的行为不同。把它们迁入 YAML 并让测试使用同一套参数，是 [V2Plan](V2Plan.md) 第二阶段的首要任务。
