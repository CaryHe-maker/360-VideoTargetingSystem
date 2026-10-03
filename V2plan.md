# V2 Plan：从比赛提交仓库到可展示的开源 360° 跟踪项目

> 分支：`SystemV2`　|　基线提交：`20c42e1`（ARTrackV2-B-256 后端）
> 目标：去掉比赛专用的耦合，把仓库整理成结构规范、结果可复现、有公开 benchmark 数据的开源项目，用于实习简历和面试讲解。
> 本文件只是计划，写这份计划时没有改动任何代码。

---

## 0. TL;DR

| 维度 | 现在 | V2 目标 |
|---|---|---|
| 定位 | InstaTest 比赛提交镜像 + 内部实验仓库 | **360° 全景视频单目标跟踪框架**：插拔式 SOT 后端 + 球面几何 + 多视图控制器 |
| Benchmark | 私有比赛数据（`E:\NewDownload\train`）、序列数和后端都在变，数字不可复现 | **公开的 [360VOT](https://360vot.hkustvgd.com)（ICCV 2023）** 为主，[360VOTS](https://arxiv.org/abs/2404.13953) 为辅；指标与官方 toolkit 对齐 |
| 对照基线 | 只有内部历史版本互相比较 | ① 原生 ERP 直接跟踪 ② 论文中的 360 框架基线（AiATrack-360 等）③ 本项目，并做逐组件消融 |
| 后端 | 只有 ARTrackV2-B-256，checkpoint 1.6 GB 走 Git LFS | `TrackerBackend` 注册表：ARTrackV2 / OSTrack / 轻量 HiT（可选 SAM2 系）；权重放 HF Hub 或 Releases，按需下载并校验 |
| 性能 | GPU Geometry + Pipeline 在旧 HiT 后端上测过 P50 355.9 → 86.9 ms | 在 ARTrackV2 上重新测：FP16 / `torch.compile` / TensorRT，报告 FPS 和 P50/P95 |
| 工程 | 无 LICENSE / CI；`.vs/` 被跟踪；Python 用 camelCase；很多 placeholder 测试 | LICENSE、CI、pre-commit、PEP 8、类型检查、CPU 可跑的测试、文档站、Demo |

**一句话简历描述（V2 完成后才成立）**：
“在 360VOT 上把通用 SOT 模型（ARTrackV2）扩展到全景视频：通过球面多视图搜索、跨缝融合和运动预测，dual success AUC 比直接在 ERP 上跟踪高 X 个点；通过 GPU 球面重采样、流水线和 FP16/TensorRT 把单帧延迟从 A ms 降到 B ms（RTX 4060 Laptop）。”

---

## 1. 现状诊断

### 1.1 值得保留的核心资产

1. **球面几何层**（`geometry/`）：ERP↔球面↔透视视图转换、BFoV、跨经线（seam）循环区间、最小覆盖区间、GPU 双线性重采样（像素回归 P99 误差为 0）。这是项目的技术亮点。
2. **多视图控制器**（`controller/`）：四角视图（VStype1）、旋转 cubemap 恢复（VStype2）、两轮 Fusor 引导搜索、球面运动估计（Huber + 切平面）、ScoreGroup 自适应阈值状态机、帧事务和一次性原子提交。
3. **协议化分层**（`core/protocols.py`、`core/types.py`）：Controller 和 Geometry 不依赖具体模型类型，更换后端的成本低。
4. **评估工具**：circular ERP IoU、球面 BFoV IoU（cos 纬度加权）、球面中心角误差、success 曲线、tracking loss rate、RuntimeProfiler。
5. **实验方法论**：`docs/PostTrain/` 中多份 A/B 报告（单变量、硬回归序列门槛、宏/微平均、P95）。面试时很有说服力，但需要从过程记录整理成结论。

### 1.2 必须处理的问题

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| P1 | 比赛耦合 | `track.py`、`app/competition.py`、`adapters/competition_adapter.py`、`docs/Competition/`、`Dockerfile`（7 层 scratch 重组、`sm_120` 断言、`DATASET_DIR`/`RESULT_DIR`）、`.dockerignore`、`tests/unit/test_competition_submission.py` | 外人看不懂，也不是通用工具 |
| P2 | 数据不可公开、结果不可复现 | 文档中大量 `E:\NewDownload\train`、`E:\tringData\...`；所有指标基于私有 manifest | 简历数字无法被验证，这是最大短板 |
| P3 | 基线混乱 | `backendBaseline.md` 的 0.256 IoU 是旧 HiT 生产版本；ARTrackV2 没有完整对比数据；各报告的序列集合不同 | 拿不出一个“干净”的提升数字 |
| P4 | 权重分发 | 1.6 GB `.pth.tar` 走 Git LFS | GitHub LFS 免费额度很小，别人 clone 很容易失败；仓库也很臃肿 |
| P5 | 缺少开源仓库的基本文件 | 无 `LICENSE`、`CONTRIBUTING`、`CITATION.cff`、`CHANGELOG`、CI | 不是“规范开源项目” |
| P6 | IDE 文件被跟踪 | `.vs/` 4 个文件 | 一看就不专业 |
| P7 | 代码风格 | 全仓库 Python 用 camelCase 函数和变量（`buildRuntime`、`frameIndex`），YAML 也是 camelCase | 违反 PEP 8；面试官会注意到 |
| P8 | 第三方代码合规 | `vendor/artrackv2` 没有附原仓库 LICENSE 和来源说明 | 开源合规风险 |
| P9 | 测试质量 | `*_placeholder.py` 6 个；不少测试依赖 CUDA 和真实权重 | 无法在 CI 中运行 |
| P10 | 死代码与历史包袱 | `classifier.py`（生产路径不调用）、`speculative_pipeline`（默认关闭）、LOST 路径“保留但不触发”、`onnx_backend.py`/`tensorrt_backend.py` 状态不明、`training/` 依赖私有数据 | 增加阅读成本，主线不清晰 |
| P11 | 文档 | 中文过程文档约 50 篇，混着计划、报告和规范；部分链接指向不存在的文件（如 `viewTypes.md`、`scoreCalibration.md`） | 需要重组 |
| P12 | 命令名 | `run`、`getInstanceID` 作为全局 console script 名过于通用，容易冲突 | 改成统一 CLI |
| P13 | 包名 | `instatarget` 带比赛和品牌色彩 | 建议改名（见 §3） |

---

## 2. 参考的成熟开源项目

| 项目 | 借鉴点 |
|---|---|
| [HuajianUP/360VOT](https://github.com/HuajianUP/360VOT)（ICCV 2023） | **Benchmark 与评测协议**：BFoV / rBFoV 标注、dual success、dual precision、angle precision；论文中的“360 tracking framework”是最直接的对照对象（AiATrack-360 dual success 0.534，原始 AiATrack 0.405） |
| [miv-xjtu/ARTrack](https://github.com/miv-xjtu/artrack)（CVPR 2023 / 2024） | 当前后端的上游；参考它的模型 zoo 表格、权重下载方式和引用格式 |
| [botaoye/OSTrack](https://github.com/botaoye/OSTrack) | 最常用的 one-stream ViT 跟踪器，适合作为第二个后端，用来证明框架与后端无关 |
| [visionml/pytracking](https://github.com/visionml/pytracking) | 跟踪器统一接口、数据集抽象、`analysis` 模块（success / precision 曲线绘制） |
| [got-10k/toolkit](https://github.com/got-10k/toolkit) | 轻量、`pip install` 即可使用的评测 toolkit 风格 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | **工程化标杆**：统一 CLI（`yolo track ...`）、Python API、权重自动下载、导出（ONNX/TensorRT）、文档站、徽章 |
| [facebookresearch/sam2](https://github.com/facebookresearch/sam2) / [yangchris11/samurai](https://github.com/yangchris11/samurai) | 可选的强后端；README 中 demo GIF、benchmark 表的写法 |
| [sunset1995/py360convert](https://github.com/sunset1995/py360convert) | 全景几何库的 API 设计；也可以作为几何正确性的交叉验证对象 |

由此得到的共同规范：**一句话定位 → Demo GIF → 结果表（含基线）→ 安装 → Quick Start（三行能跑）→ Model Zoo → 复现实验 → 架构 → 引用 / 致谢 / License**。

---

## 3. 新定位与命名

- **项目名建议**：`OmniTrack` 或 `Track360`（下文用 `omnitrack`；最终由你定）。GitHub 仓库名可以保留，也可以改名，GitHub 会自动重定向旧地址。
- **定位**：*A modular framework that turns any single-object tracker into a 360° (equirectangular) video tracker via spherical multi-view search.*
- **核心卖点**（面试讲解的三条主线）：
  1. **几何**：ERP 畸变、跨经线 seam、极点；用球面 BFoV 表示目标，在局部透视视图中做推理。
  2. **系统**：插拔式后端 + 控制器（状态机 / 运动预测 / 融合）+ 帧事务；GPU 重采样和解码、推理流水线。
  3. **评估**：在公开 benchmark 上用严格的消融和延迟统计证明每个组件的贡献。

---

## 4. Benchmark 方案

### 4.1 数据集

| 数据集 | 用途 | 说明 |
|---|---|---|
| **360VOT** | 主 benchmark | 120 条 ERP 序列，约 113K 帧，32 类，带 BBox / rBBox / BFoV / rBFoV 真值。先确认下载协议允许用于研究展示（结果可公开，数据不能再分发） |
| **360VOTS** | 补充 | 360VOT 的扩展版本（加入分割），可以用来报告更大规模的结果 |
| AirSim360 / 自采数据 | 只用于开发调试和训练实验 | 不进入 README 的主表；不在仓库中出现私有路径 |

### 4.2 指标（与 360VOT toolkit 对齐）

- **S_dual（AUC）**：主指标，跨缝的 dual success。
- **P_dual / P_angle**：中心像素精度和球面角精度。
- **BFoV spherical IoU**（项目已实现，需要和官方实现对齐数值）。
- **Tracking loss rate**：项目自定义的补充指标。
- **效率**：FPS、P50/P95 单帧延迟、每帧 forward 次数、峰值显存。固定硬件（RTX 4060 Laptop，注明驱动、CUDA 版本和功率模式），用 CUDA event 同步计时。

> 第一步必须写**官方 toolkit 交叉验证测试**：同一份结果文件，本项目的 `eval` 和 360VOT 官方脚本算出的数值误差 < 1e-3。否则表格不可信。

### 4.3 对照组与消融（README 主表的结构）

| ID | 方法 | 说明 |
|---|---|---|
| B0 | ARTrackV2 on raw ERP | 直接在 ERP 帧上跟踪（下采样到合适分辨率），用 seam-aware IoU 评估：**naive baseline** |
| B1 | 论文中的 360 框架基线 | 引用 360VOT 论文数值（AiATrack-360 等），不需要复现 |
| B2 | ARTrackV2 + 单视图 local tracking | 以上一帧 BFoV 为中心生成一个透视视图（≈ 360VOT 框架的思路） |
| A1 | + 四角多视图（VStype1）+ Fusor | 本项目核心 |
| A2 | + 两轮引导搜索 | |
| A3 | + 球面运动预测 | |
| A4 | + 自适应状态机 / cubemap 恢复 | 需要重新启用 LOST 路径并做 A/B |
| **Ours** | 完整系统 | |
| Ours-OSTrack | 换后端 | 证明与后端无关 |

效率消融：CPU geometry → GPU geometry → + decode/infer pipeline → + FP16 → + TensorRT。

按 360VOT 的挑战属性分层报告（快速运动、跨缝、极点区域、小目标、遮挡等），用雷达图展示。

### 4.4 已有的、可以作为“历史数据”参考的内部数字（不能直接放进简历）

- `docs/PostTrain/EffectiveEnhancement.md`：HiT 后端、`seq_0045`、8 views / 2 forwards：P50 **355.9 → 86.9 ms（-75.6%）**，P99 **628.9 → 123.9 ms（-80.3%）**，IoU 0.176 → 0.220。
- `docs/PostTrain/PostTrainingPlan.md`：4 条 validation 序列 mean IoU 0.359、P95 355.8 ms。

它们说明优化方向有效，但后端、数据和序列集合都不同，**必须在 360VOT + ARTrackV2 上重新测量**后才能写进 README 和简历。

---

## 5. 目标仓库结构

```text
omnitrack/
├── README.md（中文）/ LICENSE / CITATION.cff / CHANGELOG.md / CONTRIBUTING.md
├── pyproject.toml              # 依赖分组：core / cuda / trt / dev / docs
├── configs/
│   ├── default.yaml            # snake_case 键
│   └── backends/{artrackv2_b256,ostrack_b256,hit_base}.yaml
├── src/omnitrack/
│   ├── api.py                  # OmniTracker(...).track(video, init_bfov)
│   ├── cli.py                  # omnitrack track | eval | benchmark | export | download
│   ├── core/                   # types, protocols, config, errors
│   ├── geometry/               # projection, seam, bfov, gpu sampler（CPU 参考实现 + CUDA 实现）
│   ├── controller/             # planner, fusion, motion, state machine, transaction
│   ├── backends/               # registry + artrackv2/ ostrack/ hit/ (+ onnx/trt runtime)
│   ├── datasets/               # 360vot.py, 360vots.py, video_folder.py
│   ├── evaluation/             # metrics（对齐官方）、report、plots
│   ├── runtime/                # driver、prefetch、profiler
│   ├── visualization/
│   └── hub.py                  # 权重下载 + sha256 校验 + 缓存
├── third_party/artrackv2/      # 附原 LICENSE 和 NOTICE，记录 commit
├── tools/                      # benchmark.py、ablation.py、plot_results.py、export_trt.py
├── tests/                      # unit（CPU，可在 CI 运行）、gpu（标记 @pytest.mark.cuda）
├── docs/                       # mkdocs：getting-started / architecture / benchmark / api / design-notes
├── assets/                     # demo.gif、架构图、结果曲线图
├── docker/Dockerfile           # 标准单阶段 CUDA 运行镜像
└── .github/workflows/          # ci.yml（ruff + mypy + pytest-cpu）、docs.yml、release.yml
```

---

## 6. 分阶段执行计划

每个阶段单独开 PR，合并到 `SystemV2`，最后再合并回 `main`。

### Phase 1：清理与去比赛化（约 2–3 天）

- [ ] 删除 `.vs/`，在 `.gitignore` 中忽略；精简 `.gitignore`（大量 CMake/C++ 规则无用）。
- [ ] 删除 `track.py`、`app/competition.py`、`adapters/competition_adapter.py`、`docker/partition_image.py`、`docs/Competition/` 和对应测试。保留其中通用的能力：
  - mp4 视频源 + `init` BFoV 读取 → 迁移到 `datasets/video_folder.py`；
  - BFoV 文本 sink（度数、原子写、帧数校验）→ `io/writers.py`，格式与 360VOT 结果格式对齐。
- [ ] Dockerfile 改成标准运行镜像：去掉 `sm_120` 断言、7 层重组和 checkpoint 拷贝，在运行时下载权重。
- [ ] 文档清理：删除所有 `E:\...` 私有路径；`docs/Prepare`、`docs/PostTrain` 合并成 `docs/design-notes/`（只保留结论版：实验设计、结果和取舍原因），或者移到 Wiki。
- [ ] 删除或标注死代码：`classifier.py`、`speculative_pipeline`（若 benchmark 不用就删除）、`*_placeholder.py` 测试。
- [ ] 加入 `LICENSE`（建议 Apache-2.0 或 MIT，需要与 ARTrack 上游许可兼容）、`third_party/artrackv2/LICENSE` 和 NOTICE。

**验收**：`grep -ri "competition\|InstaTest\|比赛\|E:\\\\" .` 无结果；CPU 环境下 `pytest -m "not cuda"` 全部通过。

### Phase 2：工程规范化（约 3–4 天）

- [ ] 包改名为 `omnitrack`（一次性重命名，用 `git mv` 保留历史）。
- [ ] PEP 8 重命名：函数和变量改为 snake_case，YAML 键改为 snake_case。用 ruff 的 `N` 规则辅助检查；分模块提交，每次提交保证测试通过。
- [ ] 统一 CLI（`omnitrack track/eval/benchmark/export/download`），删除 `run`、`getInstanceID` 这类通用名字。
- [ ] Python API：`OmniTracker.from_pretrained("artrackv2-b256").track("video.mp4", init_bfov=(clon, clat, fov_h, fov_v))`。
- [ ] 权重分发：上传到 Hugging Face Hub（或 GitHub Releases），`hub.py` 负责下载、sha256 校验和缓存；**从仓库和 LFS 中移除 1.6 GB checkpoint**，必要时用 `git filter-repo` 清理历史以缩小仓库。
- [ ] 质量工具：pre-commit（ruff format + ruff lint + mypy 宽松模式）、GitHub Actions CI（Python 3.11/3.12，CPU）。
- [ ] Geometry 提供纯 NumPy/Torch-CPU 参考实现，让几何测试能在 CI 运行；GPU 实现与参考实现做数值一致性测试。

**验收**：CI 通过；新用户按 README 三条命令就能跑通 demo。

### Phase 3：公开 Benchmark（约 4–6 天，最核心）

- [ ] `datasets/vot360.py`：读取 360VOT 的序列、真值（BFoV/rBFoV/BBox）和属性标签。
- [ ] `evaluation/`：实现或封装官方指标，加入**与官方 toolkit 的交叉验证测试**。
- [ ] `tools/benchmark.py`：一条命令跑完全量序列，输出 `results/<method>/<seq>.txt` 和 `report.json`（含 git commit、配置哈希、硬件、版本）。
- [ ] 跑 B0、B2、A1…Ours 和效率消融；画 success/precision 曲线、属性雷达图、速度-精度散点图。
- [ ] `docs/benchmark.md`：完整表格 + 复现命令。

**验收**：任何人用 README 中的命令都能复现主表，误差在 ±0.5 个点内（固定随机性，记录 cudnn 设置）。

### Phase 4：后端与性能优化（约 5–7 天）

- [ ] **Backend 注册表**：`@register_backend("artrackv2")`；统一的 `encode_template / infer_batch` 协议。
- [ ] **新增 OSTrack-B256**（社区最常见），证明框架与后端无关；**可选 HiT** 这类轻量模型，做速度档位（Model Zoo 中分 accuracy / speed 两档）。
- [ ] **推理优化**（每一项单独 A/B，精度门槛：S_dual 下降 ≤ 0.3 个点）：
  - FP16 / BF16 autocast；
  - `torch.compile`（模板编码与搜索 forward 分开编译）；
  - 模板特征缓存（同一模板，多视图 batch 只编码一次）；
  - ONNX → TensorRT FP16 导出（`omnitrack export`），在 Model Zoo 中给出速度对比；
  - 视图数自适应：TRACKING 置信度高时 4 视图单轮，低时 8 视图两轮（预期 forward 数降低 30–50%，需要用数据证明）。
- [ ] **算法改进候选**（有数据支撑才进入主线，沿用 PostTrain 报告中的“单变量 + 硬回归门槛”方法）：
  - 重新启用并调优 LOST → cubemap 恢复（之前“保留但不触发”）；用 360VOT 中目标消失再出现的序列评估恢复率；
  - 动态模板更新（之前的 NewPic 实验有正信号，宏平均 IoU 0.3116 → 0.3208）；
  - 可选：在 360 合成数据上微调后端（第 7 节风险中有说明）。

**验收**：Model Zoo 表中至少有 2 个后端 × 2 种精度的速度和精度数据。

### Phase 5：展示层（约 2–3 天）

- [ ] Demo GIF：同一段视频并排显示 ERP 全图（含跨缝框）和局部多视图，放在 README 顶部。
- [ ] Hugging Face Space / Gradio demo：上传全景视频，点选初始框，返回跟踪视频。
- [ ] mkdocs-material 文档站 + GitHub Pages，包括架构图（数据流：Decode → GPU Resample → Batched Backend → Fusion → Motion/State → Commit）。
- [ ] `CITATION.cff`，README 中致谢 ARTrack、360VOT、OSTrack。
- [ ] 发布 `v1.0.0` Release（附权重链接和 CHANGELOG）。

---

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 360VOT 上的结果不如论文中的基线 | 这仍然是有价值的结论；重点放在“相对 naive ERP 的提升”和效率优化上；同时通过消融找到短板（通常是尺度估计和恢复） |
| 数据集许可限制 | 只公开结果文件和代码，不分发数据；README 中指向官方下载页 |
| ARTrack 上游许可与本项目 LICENSE 不兼容 | Phase 1 先核对上游 LICENSE；必要时本项目也采用相同许可 |
| 4060 Laptop 显存只有 8 GB | 默认 batch ≤ 8 views，FP16；benchmark 中注明功率模式 |
| PEP 8 大规模重命名引入 bug | 分模块、小提交；每次提交都跑全量测试；用 ruff 自动检查 |
| 从 Git 历史中移除大文件会改写历史 | 单独做、提前备份；或者只在新提交中删除，并在 README 中说明 |
| 微调后端的训练成本 | 作为可选的 stretch goal，不阻塞 v1.0 |

---

## 8. 简历和面试素材（数字在 Phase 3/4 完成后填写）

**项目描述模板**：

> **OmniTrack：360° 全景视频单目标跟踪框架**（Python / PyTorch / CUDA / TensorRT）
> - 设计球面多视图跟踪框架：把 ERP 帧按预测 BFoV 重采样为多个透视视图，批量送入 ARTrackV2/OSTrack，再经跨缝融合、球面运动预测和自适应状态机输出球面框；在 360VOT 上 S_dual 从 __ 提升到 __（+__ pts），超过论文基线 AiATrack-360（0.534）__。
> - 实现 CUDA 球面重采样和“解码 / 推理”流水线，加上 FP16/TensorRT，单帧 P50 延迟 __ ms → __ ms（RTX 4060 Laptop），FPS ×__。
> - 建立可复现的 benchmark 与消融体系（与官方 toolkit 数值对齐、CI、按属性分层分析），开源后获得 __ star。

**面试可能被追问的点（提前准备）**：
1. 为什么不直接在 ERP 上跟踪？（畸变、跨缝、极点分辨率；B0 vs Ours 的数据）
2. 跨缝 bbox 的 IoU 和融合怎么算？（循环区间、最小覆盖弧）
3. 多视图结果冲突时如何融合？（Fusor 和参考面积裁剪）
4. 状态机阈值如何自适应？（ScoreGroup 分位数）
5. GPU 重采样如何保证和 CPU 结果一致？（像素回归测试）
6. 延迟的瓶颈在哪里？（profiler 分解：decode / crop / infer / project）
7. 哪些尝试失败了，为什么？（PostTrain 报告：ERP direct crop、refinement head，平均 IoU 上升但 loss rate 恶化）

---

## 9. 时间线（建议）

| 周 | 内容 | 产出 |
|---|---|---|
| W1 | Phase 1 + Phase 2 前半 | 干净的仓库、CI 通过 |
| W2 | Phase 2 后半 + Phase 3 数据和评测 | 能在 360VOT 上跑出 B0 和 Ours |
| W3 | Phase 3 消融 + Phase 4 性能 | 主表、效率表、曲线图 |
| W4 | Phase 4 第二个后端 + Phase 5 | Demo、文档站、v1.0.0 Release |

---

## 10. 待你决定的事项

1. 项目和包名：`OmniTrack` / `Track360` / 其他？
2. LICENSE：MIT 还是 Apache-2.0（取决于上游 ARTrack 的许可）？
3. 是否改写 Git 历史以彻底移除 1.6 GB 权重？
4. 是否把后端微调列入 v1.0 的范围？

（已确定：README 只保留中文版。）
