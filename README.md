<div align="center">

# 360° 全景视频目标跟踪系统

**让普通的单目标跟踪器直接用于 360° 全景视频。**

球面多视图搜索 · 跨缝融合 · 球面运动预测 · GPU 重采样

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c)](https://pytorch.org)
[![Backend](https://img.shields.io/badge/backend-ARTrackV2--B--256-6f42c1)](https://github.com/miv-xjtu/artrack)
[![Status](https://img.shields.io/badge/status-V2%20重构中-orange)](V2plan.md)

</div>

---

## 简介

等距柱状投影（ERP）的 360° 全景帧会让普通跟踪器失效：

- 越靠近两极，目标的形变越严重；
- 目标可能跨越图像左右边界（经线接缝，下文称“跨缝”）；
- 整帧分辨率太大，无法直接送入 256×256 输入的跟踪器。

本项目不改动跟踪器本身，只改变**“看哪里”以及“如何综合多个结果”**：

1. 在单位球面上用 **BFoV**（Bounding Field of View：`clon, clat, fov_h, fov_v`）表示目标；
2. 每一帧由**控制器**围绕预测的目标方向规划若干个局部透视视图；
3. 所有视图在 **GPU 上**从 ERP 帧重采样，**一次性批量**送入后端 **ARTrackV2-B-256**；
4. 局部框回投到球面后，经过**跨缝融合**，再结合**球面运动模型**打分，**每帧只提交一次**结果。

Geometry、Controller 和 I/O 只依赖 `core/` 中定义的协议，不依赖具体模型，因此更换跟踪后端不会影响其余部分。

> **项目状态**：本仓库最初是一个比赛提交项目，正在 `SystemV2` 分支上重构为通用的开源框架。计划包括在公开 benchmark [360VOT](https://360vot.hkustvgd.com) 上评测、支持更多后端、支持 TensorRT 导出等，完整路线见 **[V2plan.md](V2plan.md)**。

## 核心特性

| 模块 | 已实现的能力 |
|---|---|
| **球面几何** | ERP ↔ 球面 ↔ 透视视图转换；BFoV 与 ERP 框互转（边界采样回投）；循环（跨缝）区间与最小覆盖弧 |
| **GPU 重采样** | ERP 帧只上传一次显存，透视网格和跨缝双线性采样都在 CUDA 中完成；与 CPU 参考实现对比，P99 像素误差为 0，最大误差 1 个灰度级 |
| **多视图搜索** | 按预测目标角尺寸生成四角视图，第二轮以第一轮融合结果为中心再搜索；另有旋转 cubemap 规划器用于找回目标 |
| **融合** | 跨缝两框融合，按参考面积自适应裁剪 |
| **运动模型** | 多帧球面速度与尺度估计（切平面 Huber 拟合，限制最大角速度和尺度变化率） |
| **状态机** | 根据历史分数分位数（ScoreGroup）自适应阈值，在 `TRACKING` 和 `UNCERTAIN` 之间切换 |
| **帧事务** | 带 revision 校验的帧事务，中间搜索轮次不会污染已提交的状态 |
| **运行时** | 后台预取解码、批量推理、分阶段性能统计 |
| **评估** | 循环 ERP IoU、球面 BFoV IoU（按 cos 纬度加权）、大圆中心误差、success 曲线 / AUC、跟踪丢失率 |

## 系统架构

```text
视频 ──▶ 预取解码线程 ──▶ ERP 帧
                            │
                            ▼
           ┌──────── TrackController ◀──────────────────┐
           │  规划视图（基于预测 BFoV，第 1 / 2 轮）       │
           ▼                                            │
   GPU Geometry：ERP → N 个透视视图（256×256）           │
           │                                            │
           ▼                                            │
   跟踪后端：ARTrackV2-B-256（N 个视图一个 batch）        │
           │                                            │
           ▼                                            │
   Geometry：局部框 → 球面（BFoV / 跨缝 ERP 框）          │
           │                                            │
           ▼                                            │
   融合 + 运动打分 + 状态机 ────── 每帧一次提交 ──────────┘
           │
           ▼
   结果输出（每帧 BFoV / ERP 框）
```

| 包 | 职责 |
|---|---|
| `core/` | 类型化数据模型、协议、YAML 配置 schema、错误类型 |
| `geometry/` | 投影数学、BFoV 投影器、跨缝处理、CUDA 采样器 |
| `controller/` | 视图规划、候选评估与融合、运动估计、状态机、跟踪控制器 |
| `tracker/` | ARTrackV2 会话（模板 / 搜索裁剪、400-bin 坐标解码）、后端适配 |
| `app/` | 运行时装配（`driver.py`）与命令行入口 |
| `eval/` | 平面 / 循环 / 球面指标，运行时性能统计 |
| `visualization/` | 中间视图与结果可视化 |
| `vendor/artrackv2/` | 上游 ARTrackV2 模型代码（推理子集） |

## 安装

环境要求：Python 3.11+、NVIDIA GPU 及 CUDA 版 PyTorch 2.11、[Git LFS](https://git-lfs.com)。

```bash
git clone https://github.com/CaryHe-maker/360-VideoTargetingSystem.git
cd 360-VideoTargetingSystem
git lfs install
git lfs pull                      # 下载 models/artrackv2_b_256.pth.tar（约 1.6 GB）
pip install -e ".[dev]"
```

确认模型文件已完整下载（SHA-256 应为 `a99b7f8086e4827ecfe32ec8a9d32ad41c1ca9ff3cac551b62ec95576ca01d05`）：

```bash
git lfs ls-files                  # 显示 "*" 表示实体文件已下载，"-" 表示只有指针
```

## 快速上手

### 跟踪单个视频

初始框使用 ERP 像素坐标 `x,y,width,height`：

```bash
python -m instatarget.track \
  --input path/to/video.mp4 \
  --init-box 1200,640,180,140 \
  --output outputs/video_result.txt \
  --config configs/RGBonly.yaml
```

### 批量模式（序列文件夹 → BFoV 文本）

每个序列目录包含一个 `.mp4` 和 `init.txt`（内容为 `clon,clat,fov_h,fov_v`，单位为度）。可以用 `seqlist.txt` 指定处理顺序。

```bash
DATASET_DIR=path/to/sequences RESULT_DIR=outputs/bfov python track.py
```

输出每帧一行 `clon,clat,fov_h,fov_v`（单位为度，保留三位小数）；没有可靠测量的帧写为 `0.000,0.000,0.000,0.000`。

### 带真值评估与可视化（AirSim360 格式数据）

```bash
python tools/run_airsim360_dataset.py --dataset-root path/to/airsim360_seq --list-instances
python tools/run_airsim360_dataset.py --dataset-root path/to/airsim360_seq \
  --config configs/RGBonly.yaml --target-instance <id>
```

## 配置

所有运行参数都在 [`configs/RGBonly.yaml`](configs/RGBonly.yaml) 中：

| 配置段 | 示例 |
|---|---|
| `model` | 后端、权重路径、精度 |
| `geometry` | 局部视图尺寸（256×256）、FoV 范围（20°–120°）、边界采样数 |
| `tracking` / `evaluator` | 候选分数阈值、每帧视图数、融合重叠阈值 |
| `motion` | Huber 参数、过程噪声、最大角速度 / 尺度变化率 |
| `recovery` | 找回目标时的 cubemap / 环形视图布局 |
| `runtime` | 解码 / 推理队列容量 |

## Benchmark

V2 阶段会在公开数据集 **[360VOT](https://360vot.hkustvgd.com)**（ICCV 2023）上给出完整结果。计划的对比方法如下：

| 方法 | 说明 |
|---|---|
| ARTrackV2 直接跟踪 ERP | 最简单的基线：直接在全景图上跟踪 |
| 360VOT 框架基线 | 引用 360VOT 论文中报告的结果（如 AiATrack-360） |
| 单局部视图 | 以上一帧 BFoV 为中心生成一个透视视图 |
| **本项目**（+ 多视图、融合、运动、状态机） | 完整系统，并逐个组件做消融 |

指标与官方 toolkit 一致：dual success（S<sub>dual</sub>，AUC）、dual precision（P<sub>dual</sub>）、angle precision（P<sub>angle</sub>），另外在固定 GPU 上报告 FPS 和 P50/P95 延迟。评测方案与进度见 [V2plan.md 第 4 节](V2plan.md#4-benchmark-方案)。

## 测试

```bash
pytest                            # 部分测试需要 CUDA 和模型权重
ruff check src tests tools
```

## 文档

设计文档位于 [`docs/`](docs/Overall/README.md)：
[系统设计](docs/Overall/structure.md) ·
[Controller](docs/Controller/structure.md) ·
[Geometry](docs/Geometry/structure.md) ·
[Tracker](docs/Tracker/structure.md) ·
[Runtime](docs/Runtime/structure.md) ·
[Evaluation](docs/Evaluation/structure.md)

## 路线图

- [ ] 移除比赛专用入口和 Docker 分层，改为标准运行镜像
- [ ] 包改名、统一为 PEP 8 命名、统一 CLI（`track / eval / benchmark / export`）并提供 Python API
- [ ] 权重改为从 Hugging Face Hub / Releases 下载，不再使用 Git LFS
- [ ] 360VOT / 360VOTS 数据加载器，评测结果与官方 toolkit 交叉验证
- [ ] 消融与效率 benchmark（GPU 几何、流水线、FP16、TensorRT）
- [ ] 后端注册表，新增 OSTrack 后端和轻量速度档
- [ ] CI、文档站、Demo GIF、Gradio 在线演示

## 致谢

- [ARTrack / ARTrackV2](https://github.com/miv-xjtu/artrack)：跟踪后端。`src/instatarget/vendor/artrackv2` 中的模型代码改编自官方实现。
- [360VOT](https://github.com/HuajianUP/360VOT)：全景跟踪 benchmark、BFoV 表示和评测协议。
- [OSTrack](https://github.com/botaoye/OSTrack)、[pytracking](https://github.com/visionml/pytracking)：跟踪器接口与评测设计的参考。

## 许可证

V2 第一阶段会在确认与上游 ARTrack 许可兼容后补充项目许可证。`vendor/` 下的第三方代码沿用其原始许可证。
