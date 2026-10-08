# 更新日志

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。到 V2.0 发布前的工作计划见 [docs/V2Plan.md](docs/V2Plan.md)。

## [未发布]

### 丢失判定、扫描和找回（默认关闭）

#### 新增

- `backendTuning.lossHandling`（默认关闭）：用外观相似度和跟踪器分数判断一帧是否可疑；可疑帧不更新跟踪器的外观记忆；连续可疑后从最后可信的位置由近到远扫描整个球面；候选明显更像目标时跳转过去。
- 外观验证器 `backends/appearance.py`：DINOv2 ViT-S/14 的特征相似度，取“对第 0 帧模板”和“对可信帧滑动平均”中较大的。权重放在 `models/hub/`，不随仓库分发。
- 后端门面的无状态推理 `inferDetached()` 和外观记忆的保存 / 恢复 / 重置。
- 外观探针 `tools/benchmark.py run --probe` 和 `tools/probe_analysis.py`：逐帧记录并比较候选的丢失信号。
- hold-out 的稳定 / 不稳定序列名单（只用于解读结果）。
- 66 条训练序列上 S<sub>dual</sub> 0.504 → 0.511（95% 区间 [−0.004, +0.017]），48 次跳转没有一次从跟得好的状态跳走；收益尚未确认，所以默认关闭。见评测记录 E016、E017。

### 回投与 360VOT 官方工具对齐

#### 变更

- 透视视图里拟合 BFoV 时，垂直方向改用纬度跨度（原来是切平面上的角度），与官方工具的定义一致。
- ERP 框的包络在所有视图里都使用框内部的采样点（原来只有球面视图使用），包含极点的框不再偏窄。
- 66 条训练序列上 S<sub>dual</sub> 0.489 → 0.504（95% 区间 [−0.007, +0.040]），丢失率 0.333 → 0.300。见评测记录 E015。金标准轨迹重新录制。

### 取视图改用 OpenCV 重映射

#### 变更

- `geometry.resampler` 增加 `opencv` 并成为默认值：用单精度算采样坐标，`cv2.remap` 双线性取色。取一个 256×256 的视图从约 19 ms 降到约 2 ms，像素值与参考实现最多差 1 个灰度级。
- tune 集上整条流水线的 P50 / P95 从 64 / 75 ms 降到 35 / 44 ms（15.6 → 27.6 FPS，单独运行）；66 条训练序列上 S<sub>dual</sub> 0.489 → 0.489（95% 区间 [−0.026, +0.030]）。见评测记录 E014。
- 原来的双精度实现保留为 `resampler: cpu`。金标准轨迹重新录制。

### 评测：运行记录和新的回归检查

#### 新增

- `tools/benchmark.py archive`：把一次运行写成记录放进 `docs/runs/`（逐条序列的分数和耗时、配置、commit、环境），并更新总表。到 E012 为止的 28 次运行已经补录。
- `compare --fragile-file` 和 `configs/splits/360vos_tune_fragile.txt`：14 条不稳定的序列作为一组判断，看平均 S<sub>dual</sub> 的差值和 bootstrap 区间。

#### 变更

- 硬回归名单换成 7 条稳定的序列（156、099、160、139、149、082、068），门槛从 0.02 改为 0.03。原来的 5 条里有 3 条在无关的改动下就会摆动 0.1 以上，几乎每个实验都误报。

### 实验：运动预测消融

#### 新增

- `backendTuning.predictiveSearch`（默认开启，行为不变）：关闭后搜索视图跟随上一帧提交的结果，不做位置和尺寸的外推。66 条序列上关闭后 S<sub>dual</sub> 0.489 → 0.465（95% 区间 [−0.061, +0.014]），没有采纳。见评测记录 E012。

### 大目标改用球面采样

#### 新增

- `ViewSpec.projection`：视图的投影方式，透视或球面。球面视图的像素在“以视图中心为 (0, 0) 的经纬度”上均匀分布，最多覆盖整个球面，做法与 360VOT 的扩展 BFoV 相同。
- `backendTuning.sphericalSearch`（默认开启）和 `sphericalSearchFovDeg`（默认 120°）：搜索区域达到这个角度时，搜索视图和模板视图改用球面采样。之前是 90° 的透视视图加补黑边。
- `ViewPlanner.templateView()`；Geometry 的裁剪和回投按视图的投影方式处理，球面视图的 ERP 框额外使用框内部的采样点（处理包含极点的框）。
- 66 条可用训练序列上 S<sub>dual</sub> 0.469 → 0.489（95% 区间 [−0.001, +0.049]），P<sub>angle</sub> +0.023 [+0.003, +0.051]，丢失率 0.368 → 0.323；目标接近 180° 的 010 号序列 0.022 → 0.768。见评测记录 E011。

#### 修复

- 序列级模型的结果依赖序列的运行顺序：坐标嵌入表的 `max_norm` 在首次查表时才生效，同一张表又是输出层。加载权重后一次缩放到位。E010 及更早的序列级数字带有这种噪声。
- 运动模型外推出的目标尺寸超出 BFoV 能表示的范围时，整帧报错。

#### 变更

- 默认配置的金标准轨迹重新录制；之前的行为保留为 `perspective_only` 变体，摘要与此前的 `default` 相同。

### 评测：效率数字进入报告

#### 新增

- 每条序列的运行报告增加前向次数（`forwards`）。
- `tools/benchmark.py` 的 `eval` 和 `compare` 打印效率表：每帧前向次数、P50 / P95 延迟、FPS；`--json` 输出里增加 `efficiency` 字段。延迟只作参考（机器负载不受控），前向次数不受机器影响。

#### 修复

- 续跑时被跳过的序列不再用空报告覆盖它原来的运行报告。

### 清理没有调用方的代码

#### 移除

- `scoreMotionConsistency()`、`calibrateMotionScore()`：基于协方差的运动一致性打分，没有调用方。
- `TrackStateMachine.update()`、`StateUpdate` 以及 `uncertainFrames` / `recoveryFrames` 计数：为旧调用方保留的适配层。状态机只剩 `transition()` 和 `recordScore()`。
- `TrackMode.TERMINATED` 和没有任何地方产生的 `TransitionReason`（`PATIENCE_EXHAUSTED`、`RECOVERY_PROGRESS`、`REACQUIRED`、`RECOVERY_EXHAUSTED`、`END_OF_STREAM`、`EXTERNAL_RESET`）。
- `evaluation/spherical_metrics.py`（`SphericalMetrics`、`bfovSphericalIoU()`）：评测使用官方工具包的球面指标。
- 序列级会话的 `inferBatchWithFovs()`。

输出不变：金标准轨迹的摘要没有变化。

### 按序列级模型运行 ARTrackV2

#### 变更

- 官方的 ARTrackV2-B-256 权重是序列级训练的结果。此前用帧级的模型代码加载，303 组参数里有 106 组被丢掉，推理时不输入轨迹。现在默认按序列级模型运行（`backendTuning.sequenceModel: true`）：权重严格加载，每帧输入前 7 帧的目标框，使用模型自己每帧更新的外观特征。`b0` 基线同样切换。
- 66 条可用训练序列上 S<sub>dual</sub> 0.483 → 0.485（95% 区间 [−0.046, +0.047]），总分不变，逐条序列有升有降；硬回归序列 131 不通过，作为已知问题保留。分数从与 IoU 无关变成随跟踪质量变化（丢失帧对好帧的 AUROC 0.42–0.46 → 0.72）。见评测记录 E010。
- 序列级用法下框架不做模板更新，`onlineTemplate` 只在 `sequenceModel: false` 时起作用。
- 默认配置的金标准轨迹重新录制；帧级用法保留为 `frame_model` 变体，摘要与此前的 `default` 相同。

#### 新增

- `backends/artrack_seq_session.py`：`PyTorchARTrackV2SeqSession` 和 `createArtrackSession()`。
- `ViewSpec.trajectory`：目标在前 7 帧的框，换算成这一帧视图里的像素坐标，由 `ViewPlanner` 从控制器保存的历史 BFoV 生成（`localBoxOfBfov()`）。
- `third_party/artrackv2/` 增加上游的序列级模型代码、外观解码器和配置（同一上游版本，见 `NOTICE`）。

### 对齐搜索区域成为默认

#### 变更

- `backendTuning.alignedSearch` 的默认值改为 `true`。66 条可用训练序列上 S<sub>dual</sub> 0.278 → 0.473（95% 区间 [+0.134, +0.255]），其中没有参与设计的 41 条 hold-out 序列上 0.283 → 0.504（评测记录 E005、E007）。硬回归序列 081 仍不通过，作为已知问题保留。
- 默认配置的金标准轨迹重新录制；旧取法保留为 `legacy_search` 变体，摘要与改默认值之前的 `default` 相同。

#### 新增

- 每条序列的逐帧置信度写入 `score/<方法>/<序列>.txt`。
- `tools/score_analysis.py` 和 `evaluation/score_analysis.py`：逐帧置信度与 IoU 的关系。
- hold-out 集 `configs/splits/360vos_holdout.txt`。

### 实验：对齐搜索区域

#### 新增

- `backendTuning.alignedSearch`（默认关闭）和 `alignedMinFovDeg`：搜索视图和模板视图改成正方形、边长为目标平均尺寸的 4 倍（模板为 `templateFovScale` 倍），视场下限 2°；视图带上预测的目标框（`ViewSpec.priorBox`），后端以它为中心裁搜索区域，视场被上限截住时超出部分补黑边。
- tune 集上 S<sub>dual</sub> 0.271 → 0.421（95% 区间 [+0.056, +0.248]），但硬回归序列 081 不通过，默认值没有改（评测记录 E005）。

### 评测：丢失率、置信区间、硬回归序列

#### 新增

- 丢失率 `evaluation/loss_rate.py`：IoU < 0.1 连续至少 5 帧算丢失，丢失率 = 丢失帧数之和 ÷ 总帧数之和。`tools/benchmark.py eval` 的 BBox 表多一列 `loss_rate`，`scores.json` 里每条序列多了 `frames`、`lostFrames`、`firstLostFrame`。
- 按序列 bootstrap 的置信区间 `evaluation/bootstrap.py`，以及 `tools/benchmark.py compare`：比较两次运行，给出 S<sub>dual</sub>、P<sub>angle</sub>、丢失率的差和 95% 区间。
- 硬回归序列 `configs/splits/360vos_tune_hard.txt`（081、107、131、156、160）和 `compare --hard-file`：任何一条的 S<sub>dual</sub> 下降超过 0.02 时命令以退出码 1 结束。

### 改为单视图跟踪

tune 集上多视图方案的 S<sub>dual</sub> 只有 0.065，单视图是 0.271，延迟还是单视图的 3.5 倍（评测记录 E001）。因此删除多视图，改为和 360VOT 论文的 360 跟踪框架相同的做法：每帧一个透视视图、一次前向、一次提交。

#### 移除

- 四角视图（`ViewSpecType1`）、旋转 cubemap 和 `RecoveryPlanner`；同一帧的第二轮搜索（`MoreViewsRequired` / `FrameCommitted`、帧事务、临时运动预测）；跨视图融合（`Fusor`、`FusionBoxMode`、`FrameAggregate`）。
- 配置段 `evaluator` 和 `recovery`；`tracking` 与 `backendTuning` 中只服务于多视图、两轮搜索和融合的字段；测试配置 `configs/tests/legacy_off.yaml`。完整列表见 [docs/configuration.md](docs/configuration.md#已删除的字段)。
- `TrackerBackend.inferTasks()` 和 `TaskKey` / `RoutedInferenceTask` / `RoutedLocalObservation`：为跨轮次混合 batch 准备的接口，运行时从未使用。
- benchmark 方法 `b2`：它现在就是 `ours`。

#### 变更

- `TrackController` 协议改为 `beginFrame(frame) -> SearchPlan` 和 `consume(plan, observation) -> TrackResult`；`SearchPlan.views` 改为单个 `view`。
- `backendTuning.singleViewHorizontalFovCapDeg` / `singleViewVerticalFovCapDeg` 改名为 `viewHorizontalFovCapDeg` / `viewVerticalFovCapDeg`。
- 分数校准产物的格式升为 `track360.score-calibration.v2`，去掉了 `thresholds.fusionSourceMinConfidence`。
- `geometry.maxFovDeg` 不再强制为 120。
- 结果图的标签去掉了 `rounds=<轮数>`。

#### 回归

- 行为与删除前的单视图路径（`backendTuning.singleView: true`，即基线 `b2`）相同：合成序列的金标准摘要与删除前录制的 `single_view_caps` 逐字节一致。tune 集上的对比见评测记录 E003。
- 默认配置的金标准轨迹重新录制（默认路径从 4 个视图变成 1 个视图，这是预期的行为变化）。

### Phase 2：360VOT 评测打通与基线（进行中）

#### 新增

- 360VOT 数据加载器 `datasets/vot360.py`：序列发现、帧读取、四种真值标注（BBox / rBBox / BFoV / rBFoV）。可以直接读发布时的 zip，不需要解压；`groundTruth()` 返回与官方 toolkit 相同的数组布局，`annotation()` 返回本项目的类型。注册为数据格式 `360vot`。
- 图像序列支持 JPG。
- BFoV 初始化：`track360 track --init-bfov clon,clat,fov_h,fov_v`，控制器和 `runTracking()` 接受 `initialBfov`。
- 360VOT 官方格式的结果写入器 `io/vot360_results.py`：每条序列同时输出 BBox 和 BFoV 两种结果文件。
- 360VOT 评测 `evaluation/vot360_metrics.py`：直接调用官方 toolkit 的指标代码（放在 `third_party/vot360_toolkit/`）。与官方脚本交叉验证：24 条序列、两组结果、两种表示，官方打印的 12 个数字与本项目全部相同。
- 批量运行工具 `tools/benchmark.py`：`run` 按方法批量跟踪（断点续跑、分片、失败隔离、记录 FPS 和延迟），`eval` 统一打分。
- 基线方法 `b0`（ARTrackV2 直接在 ERP 上跟踪）和 `b2`（单个透视视图）。
- 序列信息表解析 `datasets/vots_info.py` 和按挑战属性分层打分（`tools/benchmark.py eval --info`）。
- 360VOS 训练序列的读取，以及从分割掩码拟合 360VOT 格式标注的 `datasets/mask_labels.py`。
- tune 集：`configs/splits/360vos_tune.txt`（25 条，14,923 帧）和排除列表 `configs/splits/360vos_train_excluded.csv`，由 `tools/prepare_tune_set.py` 生成。

#### 确认的事实

- **360VOS 的 170 条训练序列里有 97 条就是 360VOT 的测试序列**，另有 5 条与测试序列剪自同一个源视频。这 102 条都不能用于调参。
- 官方 toolkit 只对 BBox 结果计算 S<sub>dual</sub>；BFoV 结果给出的是 S<sub>sphere</sub>。V2Plan 的精度目标相应改为按 BBox 结果计算。
- 官方 S<sub>dual</sub> 只把真值向左平移一个图像宽度。结果写入器把跨缝框写成负的 `x1`，这是唯一能同时匹配两种跨缝标注写法的位置。
- 官方 S<sub>sphere</sub> 把纬度当作极角传入球面 IoU，几何有疑问，暂不用它下结论。详见 [docs/benchmark.md](docs/benchmark.md#官方指标实现的几个特点)。

#### 变更

- RGB 帧改用 OpenCV 解码。原来自己实现的 PNG 解码器解一帧 3840×1920 的图要约 10 秒，现在约 0.15 秒，解码结果逐像素相同。原解码器只保留给 AirSim360 分割掩码这类标签图使用。

### Phase 1：配置收敛与工程底座

#### 变更

- **所有调参开关迁入 YAML**。原来由 `buildRuntime()` 用 `os.environ.setdefault` 写入、各模块再读取的 `TRACK360_ARTRACK_*` 环境变量，全部改为 `configs/default.yaml` 的 `backendTuning` 配置段；`src/` 中不再读取任何环境变量。默认值与迁移前实际生效的值相同，迁移前后逐帧结果一致（验证方式见下文“回归”）。
- `TRACK360_GPU_GEOMETRY` 改为配置项 `geometry.resampler`（`cpu` / `cuda`）。
- `TRACK360_PROFILE` 改为 `buildRuntime(profile=...)` 参数。
- `runTracking()` 新增必填参数 `useMotionScore`，由 `RuntimeBundle.useMotionScore` 提供。
- 单元测试与运行时使用同一份 `configs/default.yaml`。依赖“带门槛的两轮搜索”行为的 12 个控制器测试改为显式加载 `configs/tests/legacy_off.yaml`。

#### 移除

- 配置段 `decisionGate`、`runtime`（队列容量），以及从未被生产路径使用的 `DecisionGate` 类。
- `model.precision: fp16`：推理一直是 FP32，该选项不生效，现在只接受 `fp32`（FP16 在 Phase 5 实现）。

#### 新增

- 端到端回归测试 `tests/regression/`：合成 ERP 序列（目标跨越经线接缝、短暂消失、出现同色干扰物）加一个按颜色定位的假后端，走真实的 `buildRuntime` / `runTracking` 路径，不需要 GPU 和权重。金标准结果在迁移前录制。
- 可复现性：固定随机种子并启用确定性 cuDNN；每次运行在结果文件旁写入 `<结果名>.run.json`，记录 git commit、工作区是否干净、配置哈希、最终生效的配置、GPU / 驱动和依赖版本。
- GitHub Actions CI（`ruff check`、Python 3.11 / 3.12 上的 `pytest`）和 pre-commit 配置。
- `NOTICE` 中记录了 vendored ARTrackV2 代码对应的上游 commit。

#### 回归

- 合成序列 + 假后端：10 组开关组合 × 2 段序列，迁移前（环境变量）与迁移后（配置）的完整轨迹逐字节一致。
- 真实 ARTrackV2 权重（RTX 4060 Laptop，CUDA）：合成序列 40 帧和一段 3840×1920 真实全景序列的前 59 帧，迁移前后的结果文件逐字节一致；迁移前的代码连续运行两次结果也一致，说明 GPU 推理本身是确定的。

#### 迁移中确认的事实

- **默认配置下每帧只做一轮搜索（4 个视图）**。`acceptAnyCandidate: true` 时评估器不会请求第二轮；之前的文档和注释把默认路径描述为“两轮 8 视图”，与实际不符，已更正。两轮搜索只在 `acceptAnyCandidate: false` 时出现。
- 没有分数校准产物时，运动分的权重为 0，`useMotionScore` 的取值不影响结果。

## V1

面向比赛的版本，结论见 [docs/experiments.md](docs/experiments.md)，代码在提交 `20c42e1` 及更早的历史中。
