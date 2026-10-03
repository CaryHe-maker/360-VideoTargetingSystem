# 更新日志

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。到 V2.0 发布前的工作计划见 [docs/V2Plan.md](docs/V2Plan.md)。

## [未发布]

### Phase 2：360VOT 评测打通与基线（进行中）

#### 新增

- 360VOT 数据加载器 `datasets/vot360.py`：序列发现、帧读取、四种真值标注（BBox / rBBox / BFoV / rBFoV）。可以直接读发布时的 zip，不需要解压；`groundTruth()` 返回与官方 toolkit 相同的数组布局，`annotation()` 返回本项目的类型。注册为数据格式 `360vot`。
- 图像序列支持 JPG。

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
