# Track360 V2 Plan

> 起点：提交 `f04c76d`（比赛解耦与结构重组已完成）；Phase 1 已完成，记录见 [CHANGELOG](../CHANGELOG.md)
> 终点：发布 **Track360 V2.0**：在公开 benchmark **360VOT** 上明显超过已发表的 360 跟踪基线，结果可复现、工程规范，可以作为成熟开源项目发布和写进简历。
>
> 版本约定：**V1** 是面向比赛的版本（已结束）；**V2** 是本次开源重构，本计划全部完成后发布 **V2.0**；后续增强放在 V2.1 及以后（见第 6 节）。
>
> 本文只列**尚未完成**的工作。每完成一项，直接从本文删除，并在 `CHANGELOG.md` 中记录。每一轮评测结果都记录在 [evaluation-log.md](evaluation-log.md)。

---

## 0. 成功标准

### 0.1 精度（360VOT 测试集，BBox 结果，官方 toolkit 计算）

| 对比对象 | 当前已知数值 | V2.0 目标 |
|---|---|---|
| 论文中的最佳 360 框架基线 AiATrack-360 | S<sub>dual</sub> 0.534 / P<sub>dual</sub> 0.506 / P<sub>angle</sub> 0.574 | **S<sub>dual</sub> ≥ 0.56**（至少 +2.5 个点），P<sub>angle</sub> 同步提升 |
| 论文中最好的通用跟踪器 OSTrack（直接在 ERP 上跟踪） | S<sub>dual</sub> 0.447 | 超过 10 个点以上 |
| 本项目 B0：ARTrackV2 直接在 ERP 上跟踪 | Phase 2 测出 | Track360 相对 B0 提升 ≥ 8 个点 |

> 论文基线数值摘自 [360VOTS 论文](https://arxiv.org/abs/2404.13953)，写进 README 前要对照原文表格核实。官方 toolkit 只对 BBox 结果计算 S<sub>dual</sub> / P<sub>dual</sub>，所以对比用 BBox 结果；BFoV 结果的 S<sub>sphere</sub> 在官方实现里几何有疑问，暂不作为目标（见 [benchmark.md](benchmark.md#官方指标实现的几个特点)）。

### 0.2 效率（RTX 4060 Laptop，固定功率模式，CUDA event 计时）

| 指标 | V2.0 目标 |
|---|---|
| 端到端 P50 单帧延迟（含解码，3840×1920） | ≤ 100 ms（≥ 10 FPS） |
| TensorRT FP16 档 | ≥ 20 FPS，S<sub>dual</sub> 下降 ≤ 0.3 个点 |
| 峰值显存 | ≤ 6 GB |

### 0.3 发布标准

- 任何人按 README 操作，都能在 ±0.5 个点内复现主表；
- CI 通过（lint、类型检查、CPU 测试）；有 LICENSE、CITATION、CHANGELOG 和文档站；
- 权重可以通过 `track360 download` 自动下载并校验；
- 有 Demo GIF 和在线演示。

---

## 1. 当前的已知问题（起点）

| 问题 | 影响 | 解决阶段 |
|---|---|---|
| 默认配置每帧只做一轮搜索（4 个视图）：`acceptAnyCandidate: true` 时评估器不请求第二轮。两轮搜索、融合门槛等逻辑只在 `configs/tests/legacy_off.yaml` 下被测试覆盖 | “两轮 8 视图”是否比单轮更好，没有在 ARTrackV2 上验证过 | Phase 4 |
| 读视频文件依赖系统里的 ffmpeg / ffprobe | 没装 ffmpeg 的机器只能跟踪图像序列 | Phase 6 |
| 没有任何公开数据集上的结果 | 无法和 benchmark 对比 | Phase 2 |
| ARTrackV2 调用时 `seq_input=None`，没有使用模型的轨迹提示（trajectory prompt） | 很可能丢掉了 ARTrackV2 的大部分时序优势 | Phase 4 |
| `LOST` / cubemap 找回路径保留了但从不触发 | 目标丢失后只能靠扩大局部搜索 | Phase 4 |
| ARTrackV2 的分数集中在 0.5 附近，状态机和融合门槛依赖这个分数 | 门控不可靠 | Phase 4 |
| 函数和变量用 camelCase，YAML 键也是 camelCase | 不符合 PEP 8 | Phase 3 |
| 权重需要用户手动从官方链接下载 | 上手门槛高，不是成熟的开源项目 | Phase 6 |

---

## 2. 分阶段计划

整体顺序：**先能复现 → 再有基线数字 → 再规范代码 → 再提精度 → 再提速度 → 最后发布**。每个 Phase 单独开分支和 PR，验收通过后合并。

```text
Phase 1 配置收敛与工程底座   ─┐
Phase 2 360VOT 评测打通与基线 ─┼─▶ Phase 3 代码规范化（用基线结果做回归）
                              │
                              └─▶ Phase 4 精度提升 ─▶ Phase 5 效率优化 ─▶ Phase 6 产品化 ─▶ Phase 7 发布 V2.0
```

### Phase 1：配置收敛与工程底座（已完成）

完成的内容和回归方式记录在 [CHANGELOG](../CHANGELOG.md)。只剩一项需要确认：

1. **CI 首次运行**：分支推送后确认 GitHub Actions 在 Linux 上通过。金标准回归是在 Windows 上录制的，如果 Linux 上的 OpenCV 重采样结果有差异导致 `tests/regression` 失败，需要在 CI 环境重新录制或放宽比较方式。

### Phase 2：360VOT 评测打通与基线（约 5 天，最关键）

**目标**：拿到第一个能和论文直接对比的数字。

1. **下载数据**：360VOT 测试集与标注（约 58.5 GB）；360VOS 训练集按需下载。数据放在仓库外，通过参数传入。加载器可以直接读 zip，测试集不需要解压；运行官方评测脚本前需要把各序列的 `label.json` 解压出来。
2. **用官方发布的结果复核数据**：下载官方 toolkit README 里提供的 benchmark 结果文件，用 `tools/benchmark.py eval` 打分，确认论文基线的数值能复现。这一步同时验证下载的数据和标注没有问题。
3. **跑基线**（工具和三种方法都已就绪，命令见 [benchmark.md](benchmark.md#复现命令)）：先在 tune 集上跑，确认流程和结果合理，再在测试集上跑 `b0`、`b2`、`ours` 各一次。
4. **记录结果**：三组结果、按属性分层的结果写入 [evaluation-log.md](evaluation-log.md)（记录 E001 起），复现命令写入 `docs/benchmark.md`。

**验收**：三组结果齐全并记入 [evaluation-log.md](evaluation-log.md)；官方 toolkit 交叉验证通过；在 tune 集上 Ours-v0 至少不差于 B2。如果 Ours-v0 比 B2 差，先进入 Phase 4 的问题排查，再继续。

### Phase 3：代码规范化（约 4 天）

**目标**：在有可靠回归手段的前提下完成全仓库 PEP 8 改名，结果不能有任何变化。

1. 函数、变量、参数改为 snake_case；类名保持 PascalCase；用 ruff 的 `N` 规则检查。
2. YAML 键改为 snake_case；`loadConfig()` 在一个版本周期内兼容旧键，并给出弃用警告。
3. 按模块分批提交（core → geometry → controller → backends → runtime → 其他），每批都跑全量测试。
4. 加入 mypy（先用宽松模式），给公开 API 补全类型注解。
5. **回归**：在 tune 集上重新跑 Ours-v0，结果文件必须与 Phase 2 **逐字节一致**。

**验收**：ruff `N` 规则无报错；tune 集结果逐字节一致；mypy 通过。

### Phase 4：精度提升（约 2–3 周，核心）

**目标**：在 tune 集上把 S<sub>dual</sub> 提升到预期能超过 0.56 的水平，再在测试集上**只跑一次**确认。

工作方法：

1. **先诊断再动手**：在 tune 集上把每一帧的失败归类：没有视图覆盖到目标 / 覆盖到了但后端框错 / 后端框对但融合或状态选错 / 回投误差 / 跨缝或极点。每类统计帧数占比，按占比从高到低处理。
2. **一次只改一个变量**，每项实验都报告 S<sub>dual</sub>、P<sub>angle</sub>、丢失率、每帧前向次数和 P95 延迟。
3. **硬回归门槛**：每项改动先在 tune 集中最容易出问题的 5 条序列上跑，任何一条下降超过 2 个点就停止。
4. **每一轮实验都在 [evaluation-log.md](evaluation-log.md) 追加一条记录**（不论是否采纳）；被采纳的改动进入 README 的消融表。

按预期收益排序的实验清单（详细说明见第 3 节对应模块）：

| 优先级 | 实验 | 模块 | 预期作用 |
|---|---|---|---|
| P0 | 恢复 ARTrackV2 轨迹提示（`seq_input`），把上一帧轨迹映射到每个视图的局部坐标 | Backends | 恢复模型的时序能力，预计收益最大 |
| P0 | 搜索区域与视图尺度对齐：让目标在 256 搜索图中的占比与 ARTrackV2 训练分布一致 | Controller / Backends | 减少尺度失配造成的框误差 |
| P1 | 分数重校准：在 tune 集上拟合 IoU 感知的分数映射，替代 0.5 附近的原始分数 | Controller | 让融合、状态机和模板门控可靠 |
| P1 | 运动分重新启用：在分数校准后重新评估“外观 + 运动”加权 | Controller | 抑制相似物体干扰 |
| P1 | 单轮与两轮搜索对比：默认配置是单轮 4 视图，对比带门槛的两轮 8 视图（依赖分数重校准） | Controller | 确认第二轮是否值得它的前向开销 |
| P1 | 启用 LOST 状态和 cubemap 找回，阈值在 tune 集上确定 | Controller | 目标丢失或出画后能找回 |
| P2 | 视图数自适应：高置信度时单轮 4 视图，低置信度时两轮 8 视图 | Controller | 精度不降的前提下减少前向次数 |
| P2 | 旋转 BFoV（rBFoV）输出：用回投边界点拟合旋转角 | Geometry | 提高极点附近和倾斜目标的 IoU |
| P3 | 第二个后端 OSTrack-B256，验证框架与后端无关 | Backends | 证明方法的通用性 |

**验收**：tune 集 S<sub>dual</sub> 稳定达到目标；360VOT 测试集**只跑最终配置一次**，结果写入主表；消融表中每一行都有数据。

### Phase 5：效率优化（约 1 周）

**目标**：精度不降（S<sub>dual</sub> 下降 ≤ 0.3 个点）的前提下达到第 0.2 节的速度目标。每项单独 A/B。

1. **性能剖析**：用 `RuntimeProfiler` + CUDA event 拆分 decode / crop / backend / project / controller 的耗时，确认瓶颈；
2. **GPU 解码**：使用 NVDEC（如 PyNvVideoCodec 或 decord 的 GPU 解码），4K 视频解码往往是第一个瓶颈；
3. **GPU 几何默认开启**：先确认 GPU 重采样与 CPU 路径在 tune 集上的结果差异可以接受；
4. **FP16 / BF16**：实现 `model.precision: fp16`（autocast），模板特征用半精度缓存；
5. **`torch.compile`**：模板编码和搜索前向分开编译，固定 batch 尺寸（4 / 8）避免重编译；
6. **TensorRT**：导出 ONNX → TensorRT FP16，作为独立的速度档后端（`track360 export`）；
7. **流水线**：解码、几何和推理分到不同的 CUDA stream，下一帧的解码与当前帧推理重叠。

**验收**：Model Zoo 表中至少有 “ARTrackV2 FP32 / FP16 / TensorRT” 三档的速度和精度，效率数据记入 [evaluation-log.md](evaluation-log.md) 第 2.3 节。

### Phase 6：产品化（约 1 周）

1. **Python API**：`Track360Tracker.from_pretrained("artrackv2-b256").track("video.mp4", init_bfov=(...))`，返回逐帧结果对象；
2. **权重分发**：上传到 Hugging Face Hub（或 GitHub Releases，需确认上游许可允许再分发；否则 `hub.py` 直接指向官方链接），`hub.py` 负责下载、SHA-256 校验和缓存，并提供 `track360 download`；
3. **命令行补全**：`track360 eval`、`track360 benchmark`、`track360 export`、`track360 download`；
4. **后端注册表**：`@register_backend("artrackv2")`，配置中按名称选择后端；
5. **可视化**：生成 ERP 全图 + 局部视图并排的结果视频，制作 README 顶部的 Demo GIF；
6. **在线演示**：Hugging Face Space（Gradio），上传全景视频、点选初始目标、返回跟踪视频；
7. **文档站**：mkdocs-material + GitHub Pages，内容来自现有 `docs/`；
8. **Docker**：发布带 CUDA 的镜像，并写好 GPU 运行示例。

**验收**：新用户只用 README 上的三条命令就能跑通 Demo。

### Phase 7：发布 V2.0（约 2 天）

1. README 主表：B0、论文基线、Track360 各档位，加上 success / precision 曲线、按属性的雷达图、速度—精度散点图；
2. 消融表：每个组件的贡献；
3. `CITATION.cff`、致谢（ARTrack、360VOT、OSTrack）；
4. 把版本号从 `2.0.0.dev0` 改为 `2.0.0`，打 `v2.0.0` tag，发布 GitHub Release（附权重链接、CHANGELOG、结果文件压缩包）；
5. 把 [evaluation-log.md](evaluation-log.md) 中最终结果对应的结果文件附到 Release，方便别人用官方 toolkit 直接验证。

---

## 3. 按模块的优化建议

### 3.1 Core / 配置

- **配置分层**：`configs/default.yaml` + `configs/backends/<name>.yaml` + 命令行覆盖（`--set tracking.windowLength=7`），方便做消融。
- **精简类型**：`ProjectedObservation` 字段很多，把诊断字段拆到独立的 `ObservationDiagnostics`，主流程只保留必要字段。

### 3.2 Geometry

- **rBFoV**：现在只拟合无旋转的 BFoV。可以用回投边界点求最小外接旋转区域，得到带 roll 的 BFoV，直接对齐 360VOT 的 rBFoV 标注。
- **GPU 路径成为默认**：补一组在 tune 集上的端到端 A/B，确认差异可以接受后默认开启。
- **批量回投**：现在逐个框回投；可以把一帧所有候选的边界点合并成一个张量一次计算（GPU 上效果更明显）。
- **极点测试**：补充纬度 ±85° 以上的回投和裁剪测试，并在报告中按纬度分层统计误差。
- **抗锯齿**：视图视场很大（120°）时，从 4K ERP 采样到 256×256 会产生混叠；可以先构建 ERP 图像金字塔，按视场选择对应层级采样。

### 3.3 Controller

- **视图规划**
  - 让 ARTrackV2 搜索区域中的目标占比与训练分布一致：目前视图视场是目标的 3 倍，再叠加后端 4 倍的搜索裁剪，需要统计实际占比并调整（P0）。
  - 视图数自适应：根据上一帧的置信度和运动不确定度，在 1 / 4 / 8 个视图之间切换。
  - 启用 cubemap 找回时，只接受分数显著高于背景的候选，并设置冷却帧数，防止跳到相似物体。
- **评估与融合**
  - 融合常量（0.70 重叠率、0.15 奖励、0.03 上限）改为配置项，在 tune 集上用网格搜索确定。
  - 融合时考虑投影质量：靠近视图边缘（`edgeMargin` 小）或包络膨胀大的候选降权。
- **分数**
  - 在 tune 集上收集“原始分数 → 候选与真值的 IoU”，拟合单调校准（isotonic 或 Beta），让分数近似“IoU > 0.5 的概率”。
  - 校准后重新评估运动先验的权重，50/50 加权可能重新有效。
- **运动模型**
  - 现在是常速度模型。可以换成球面上的 Kalman 滤波（状态：方向 + 角速度 + log 尺度），用协方差决定搜索视图的大小。
  - 相机自身旋转补偿：全景相机手持或车载时，背景整体旋转会让目标的世界方向变化很大；可以用光流或特征点估计帧间旋转，先补偿再预测。
- **状态机**
  - 用 tune 集数据拟合 `TRACKING / UNCERTAIN / LOST` 的阈值，替代手工的分位数规则；至少把“第 5 / 第 8 名”这两个分位点改为配置项。
- **模板**
  - recent 模板的更新条件改为“校准后分数 + 连续 N 帧稳定 + 尺度变化小”，保留 anchor 回退。
  - 研究 ARTrackV2 原生的外观提示能否跨帧传递，作为模板更新的替代方案。

### 3.4 Backends

- **轨迹提示（最高优先级）**：官方 ARTrackV2 推理时会把前几帧的框坐标作为 `seq_input` 输入，当前实现传入 `None`。需要对照官方 tracker 代码确认输入格式，再把上一帧的球面轨迹投影到每个视图的局部坐标系，转换成 400-bin 的坐标 token。
- **FP16 生效**：实现 `model.precision: fp16`，模板特征在初始化时缓存为半精度。
- **模板编码缓存**：确认同一帧多个视图共享同一份模板特征，没有重复编码。
- **注册表与多后端**：抽象出 `encode_template / infer_batch / decode` 三步，新增 OSTrack-B256；可选一个轻量后端（如 HiT）作为速度档。

### 3.5 Runtime

- **GPU 解码 + 多 stream 流水线**：解码、重采样和推理重叠执行，取代现在只有解码在后台线程的结构。
- **批量序列评测**：benchmark 工具支持多进程，按序列并行。
- **失败隔离**：单条序列出错时记录错误并继续下一条，最后汇总失败列表。
- **结构化日志**：用 `logging` 输出每条序列的进度、FPS 和错误，取代 `print`。

### 3.6 Datasets / IO

- `vot360.py` 补充属性标签、360VOS 读取和序列去重列表。
- 通用视频读取支持 `--init-bfov`，并提供交互式选择初始框的小工具（OpenCV 窗口画框 → 转为 BFoV）。
- 结果写入器支持三种格式：ERP 框、BFoV、360VOT 官方格式。
- AirSim360 读取器保留为开发数据，在文档中标明它不是 benchmark。

### 3.7 Evaluation

- 与官方 toolkit 的交叉验证测试（见 Phase 2）。
- 按属性分层报告、失败帧自动归类（见 Phase 4 工作方法第 1 步）。
- 自动生成 success / precision 曲线、属性雷达图、速度—精度散点图（`tools/plot_results.py`）。
- 显著性：对主要消融做按序列的 bootstrap，给出置信区间，避免把噪声当作提升。

### 3.8 Visualization

- 结果视频：ERP 全图（含跨缝框）+ 当前帧的局部视图拼接，标注状态和分数。
- 失败帧画廊：按失败类别导出关键帧，便于定位问题和写文档。

### 3.9 测试与 CI

- GPU 测试加 `@pytest.mark.cuda` 标记，CI 只跑 CPU 部分；发布前在本地跑全量。
- benchmark 金标准回归：从 tune 集选 3 条短序列，结果文件一旦变化测试就失败，必须显式更新基准。

### 3.10 文档

- 每个 Phase 完成后同步更新 `docs/` 中对应的模块文档和 README。
- 新增“如何接入新后端”“如何复现 benchmark”两篇教程。
- 把 Phase 4 的失败分析和消融结论整理进 `docs/experiments.md`，作为面试素材。

---

## 4. 实验纪律

1. **测试集只用来报告**：所有参数、开关和模型选择只在 tune 集（[`configs/splits/360vos_tune.txt`](../configs/splits/360vos_tune.txt)，25 条）上决定；测试集在 V2.0 之前最多跑两次（Phase 2 基线、Phase 4 最终配置）。
2. **一次只改一个变量**，结果目录包含配置快照、git commit 和环境信息。
3. **同时看多个指标**：S<sub>dual</sub>、P<sub>angle</sub>、丢失率、每帧前向次数、P95 延迟。只涨平均 IoU 但丢失率变差的方案不采用（V1 阶段多次出现这种情况，见 [experiments.md](experiments.md)）。
4. **硬回归序列早停**：先跑最容易出问题的序列，不通过就不扩大实验。
5. **结论写进文档**：每一轮评测都在 [evaluation-log.md](evaluation-log.md) 追加记录；阶段性的经验总结写入 `docs/experiments.md`。

---

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| 优化后仍然达不到 0.56 | 先保证明显超过 B0 和论文中直接在 ERP 上跟踪的基线；重点放在消融分析和效率上；更强的后端或微调留到 V2.1 |
| 360VOS 与 360VOT 序列重叠造成数据泄漏 | Phase 2 第 2 步强制去重，排除列表提交到仓库 |
| 数据集许可（360VOTS 为 CC BY-NC-SA 4.0） | 只公开代码和结果文件，不分发数据 |
| 4060 Laptop 显存 8 GB | 默认 batch ≤ 8，使用 FP16 |
| PEP 8 大规模改名引入 bug | Phase 3 用逐字节一致的结果回归保护 |

---

## 6. V2.1 及以后

以下内容**不进入 V2.0**，V2.0 发布后再评估：

- **后端微调**：用 360VOS 训练集（与 360VOT 测试集去重后）生成“透视视图 + 局部框”训练对，在 ARTrackV2 上做少量轮次微调，适应全景投影的外观分布；训练集与测试集严格隔离；微调权重受数据集许可（CC BY-NC-SA 4.0）约束，需注明非商业使用；显存不足时使用梯度累积。
- 更强或更轻量的后端（如 SAM2 系、HiT）作为新的精度档 / 速度档。

---

## 7. 简历与面试素材（数字在 Phase 4 / 5 完成后填写，以 evaluation-log.md 中的记录为准）

> **Track360：360° 全景视频单目标跟踪框架**（Python / PyTorch / CUDA / TensorRT）
> - 设计球面多视图跟踪框架：按预测 BFoV 把 ERP 帧重采样为多个透视视图，批量送入 ARTrackV2，经跨缝融合、球面运动预测和自适应状态机输出球面框；在 360VOT 上 S<sub>dual</sub> 达到 __，比论文中的最佳 360 基线 AiATrack-360（0.534）高 __ 个点，比直接在 ERP 上跟踪高 __ 个点。
> - 实现 CUDA 球面重采样、GPU 解码和多 stream 流水线，结合 FP16 / TensorRT，把单帧延迟从 __ ms 降到 __ ms（RTX 4060 Laptop）。
> - 建立可复现的评测体系：与官方 toolkit 数值对齐、tune / test 严格隔离、按属性分层与 bootstrap 置信区间、CI 回归。

面试中可能被追问的问题：

1. 为什么不直接在 ERP 上跟踪？（形变、跨缝、分辨率；B0 与 Track360 的对比数据）
2. 跨缝的框怎么求 IoU 和融合？（循环区间、最小覆盖弧）
3. 多个视图的结果冲突时怎么选？（融合规则、分数校准）
4. ARTrackV2 的轨迹提示在多视图下怎么用？（坐标系转换）
5. 状态机阈值如何确定？
6. 延迟瓶颈在哪里，怎么定位和优化的？
7. 哪些尝试失败了，为什么？
