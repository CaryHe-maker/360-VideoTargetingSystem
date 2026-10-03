# Benchmark 数据集

本文记录测试集的选型结论和评测协议；每一轮的评测结果记录在 [evaluation-log.md](evaluation-log.md)。数据读取见 [Datasets](modules/datasets.md)；评测脚本是 [V2Plan](V2Plan.md) Phase 2 的任务。

## 结论

| 用途 | 数据集 | 理由 |
|---|---|---|
| **主 benchmark** | **360VOT 测试集**（120 条序列） | 唯一专门针对 360° 单目标跟踪的公开 benchmark；标注直接包含 BFoV；有官方 toolkit 和论文基线，结果可以直接对比 |
| 补充训练 / 验证 | 360VOTS 中的 360VOS 训练集（170 条序列） | 与测试集同源同分辨率，可用于调参（后端微调留到 V2.1），避免在测试集上调参；使用前需剔除与 360VOT 测试集重叠的序列 |
| 不采用 | PanoVOS、AirSim360 等 | 见下文 |

## 候选数据集对比

| 数据集 | 任务 | 规模 | 标注 | 许可 | 评价 |
|---|---|---|---|---|---|
| [360VOT](https://360vot.hkustvgd.com)（ICCV 2023） | 全景单目标跟踪 | 120 条序列，约 113K 帧，3840×1920，32 类 | BBox、rBBox、BFoV、rBFoV | toolkit 为 MIT；数据使用条款以官网为准 | **最匹配**：任务、目标表示（BFoV）和难点（跨缝、形变）与本项目完全一致 |
| [360VOTS](https://360vots.hkustvgd.com)（TPAMI 2025） | 全景跟踪 + 分割 | 360VOS 共 290 条序列（训练 170 / 测试 120），62 类 | 逐帧掩码，可转换为四种框 | CC BY-NC-SA 4.0 | 训练集适合做验证和微调；跟踪评测仍以 360VOT 为准 |
| PanoVOS（ECCV 2024） | 全景视频目标分割 | 150 条视频，约 19K 实例掩码 | 掩码 | — | 任务是分割，没有跟踪协议和 BFoV 基线，不适合作为主表 |
| AirSim360（V1 使用的数据） | 仿真全景序列 | — | 实例掩码（生成伪真值） | 比赛方私有 | 无法公开，结果不可复现，只用于开发调试 |

## 360VOT 详情

- **下载**：测试集与标注约 58.5 GB，从[官网下载页](https://360vot.hkustvgd.com)或 [Hugging Face](https://huggingface.co/datasets/xuyzshaun/360VOTS) 获取；
- **toolkit**：[HuajianUP/360VOT](https://github.com/HuajianUP/360VOT)，提供评测与可视化脚本；
- **数据布局**：每条序列一个 zip，内含 `NNNN/image/000000.jpg…` 和 `NNNN/label.json`；标注字段和坐标约定见 [Datasets](modules/datasets.md#360vot)；
- **结果文件每行的格式**（角度单位为度）：
  - BBox：`[x1, y1, w, h]`
  - rBBox：`[cx, cy, w, h, rotation]`
  - BFoV / rBFoV：`[clon, clat, fov_h, fov_v, rotation]`
- **结果格式**：每个跟踪器一个目录，每条序列一个 `NNNN.txt`（共 120 个文件），每行一帧；
- **评测命令**（官方 toolkit）：`-b` 给 BBox 结果目录，`-f` 给 BFoV 结果目录，`-d` 给解压后的数据集目录：

```bash
python scripts/eval_360VOT.py -d <dataset_dir> -b <results>/bbox -f <results>/bfov
```

### 指标

官方 toolkit 对两种结果表示给出不同的指标：

| 结果表示 | 指标 | 含义 |
|---|---|---|
| BBox | S<sub>dual</sub>（AUC） | **主指标**。IoU 在 21 个阈值上的成功率取平均；为了处理跨缝，会把真值平移一个图像宽度后再比一次 |
| BBox | P<sub>dual</sub> | 中心点像素误差 ≤ 20 像素的帧比例，同样考虑跨缝 |
| BBox / BFoV | P<sub>angle</sub> | 中心方向在球面上的夹角 ≤ 3° 的帧比例 |
| BFoV | S<sub>sphere</sub>（AUC） | 球面 IoU 的成功率 AUC |

**S<sub>dual</sub> 只对 BBox 结果计算**，论文表格里的 S<sub>dual</sub> / P<sub>dual</sub> 也是 BBox 结果。所以和论文对比用 BBox 结果；BFoV 结果给出的是 S<sub>sphere</sub> 和 P<sub>angle</sub>。

### 官方指标实现的几个特点

本项目直接使用官方 toolkit 的指标代码（见 [Evaluation](modules/evaluation.md#360vot-指标)），这样分数才和别人报告的一致。读分数时需要知道它的这些行为（上游提交 `a729a74`）：

- **目标缺席的帧也算在分母里**。这些帧不参与比较，但总帧数包含它们，所以它们永远算失败。测试集里约 2% 的帧目标缺席，任何方法的分数上限都低于 1。
- **S<sub>dual</sub> 只把真值向左平移**。代码里向右平移的那一项实际用的也是向左平移的真值。跨缝框的真值有两种写法（超出左边界，或超出右边界），预测框只有写成负的 `x1` 才能同时匹配这两种。本项目的结果写入器就是这样写的。
- **P<sub>dual</sub> 把真值中心坐标 ≤ 0 的帧直接算命中**。影响很小，只涉及中心贴着左边界的帧。
- **S<sub>sphere</sub> 的几何有疑问**。球面 IoU 的实现把第二个参数当作从极点量起的极角，而评测脚本传进去的是 `clat`（从赤道量起的纬度）。结果是赤道附近的目标被放到了极点附近，经度上的偏差被大幅缩小：我实测两个经度相差 90°、互不重叠的 30°×20° 框，在 `clat = 5°` 时算出的 IoU 是 0.50。这是按我对代码的阅读和数值测试得出的判断，没有向作者确认过。在弄清之前，**不要用 S<sub>sphere</sub> 下结论**，以 BBox 的 S<sub>dual</sub> 和 P<sub>angle</sub> 为准（P<sub>angle</sub> 的经纬度换算是对的）。

### 论文中的对照基线（BBox 标注）

| 方法 | S<sub>dual</sub> | P<sub>dual</sub> | P<sub>angle</sub> |
|---|---:|---:|---:|
| OSTrack | 0.447 | 0.433 | 0.484 |
| AiATrack | 0.405 | 0.369 | 0.423 |
| SimTrack | 0.400 | 0.373 | 0.424 |
| MixFormer | 0.395 | 0.378 | 0.424 |
| **AiATrack-360**（论文的 360 框架） | **0.534** | **0.506** | **0.574** |

> 数值摘自 [360VOTS 论文](https://arxiv.org/abs/2404.13953)，正式引用前请对照原文表格核实。

## 评测协议（本项目）

1. **对照组**：
   - B0：ARTrackV2 直接在 ERP 帧上跟踪（最简单的基线）；
   - B1：引用论文基线（上表）；
   - B2：ARTrackV2 + 以上一帧 BFoV 为中心的单个透视视图；
   - Ours：完整系统，并逐个组件做消融（多视图、两轮搜索、融合、运动、状态机）。
2. **一致性**：同一份结果文件分别跑官方脚本和本项目的评测代码，结果必须一致。已验证：24 条序列、两组人为加噪的结果、BBox 和 BFoV 两种表示，官方脚本打印的 12 个数字与本项目全部相同（官方只打印三位小数）。
3. **不在测试集上调参**：所有参数和开关在 360VOS 训练集（或其中划出的验证子集）上确定，测试集只跑最终配置。
4. **效率**：固定 GPU（RTX 4060 Laptop），注明驱动和 CUDA 版本，报告 FPS、P50 / P95 延迟、每帧前向次数和峰值显存。批量运行工具记录的是端到端的墙钟时间（两帧结果提交之间的间隔，含解码）；分阶段的 CUDA 计时在 Phase 5 做。
5. **分层分析**：按 360VOT 的挑战属性（快速运动、跨缝、极点区域、小目标、遮挡等）分别报告。
6. **可复现**：结果目录中保存 git commit、配置哈希、硬件和依赖版本。

## 复现命令

数据集根目录下放各序列的 zip 或解压后的目录都可以。三种方法各跑一遍，再统一打分：

```bash
python tools/benchmark.py run  --dataset-root <360VOT-test> --output-root outputs/360vot --method b0
python tools/benchmark.py run  --dataset-root <360VOT-test> --output-root outputs/360vot --method b2
python tools/benchmark.py run  --dataset-root <360VOT-test> --output-root outputs/360vot --method ours
python tools/benchmark.py eval --dataset-root <360VOT-test> --output-root outputs/360vot --json outputs/360vot/scores.json
```

| 方法 | 含义 |
|---|---|
| `b0` | ARTrackV2 直接在原始分辨率的 ERP 帧上跟踪：搜索区域跟随上一帧的框，不处理跨缝和形变 |
| `b2` | 单个透视视图跟随上一帧的 BFoV（`backendTuning.singleView: true`），其余与 `ours` 相同 |
| `ours` | `--config` 指定的配置，默认 `configs/default.yaml` |

- **断点续跑**：中断后重跑同一条命令，结果文件已完整的序列会跳过；加 `--no-resume` 强制重跑。
- **拆分到多个进程**：`--shard 0/2` 和 `--shard 1/2` 各跑一半序列。每个进程各加载一份模型（约 1 GB 显存）。
- **只跑部分序列**：`--sequences 0001,0003`。
- **冒烟测试**：`--max-frames 25` 只跑每条序列的前 25 帧。这样的结果比序列短，`eval` 会拒绝；确实要看的话加 `--allow-partial`，但这种分数不能记入评测记录。
- 一条序列出错不会中断整轮，错误记在 `reports/<方法>/<序列>.json` 里，最后命令以非零退出码结束。

输出目录的结构与官方 toolkit 要求的一致，可以直接交给官方脚本复核：

```text
outputs/360vot/bbox/<方法>/0001.txt      x1,y1,w,h
outputs/360vot/bfov/<方法>/0001.txt      clon,clat,fov_h,fov_v,rotation
outputs/360vot/reports/<方法>/run.json   方法、生效的配置、配置哈希、git commit、环境
outputs/360vot/reports/<方法>/0001.json  帧数、FPS、P50 / P95 延迟、无效帧数
```

在 RTX 4060 Laptop 上用两条序列各 25 帧试跑的速度（只用来估算总耗时，不是正式的效率数据）：`ours` 约 5 FPS，`b2` 和 `b0` 约 12–18 FPS。按 11.3 万帧估算，`ours` 跑完整个测试集约需 6 小时。
