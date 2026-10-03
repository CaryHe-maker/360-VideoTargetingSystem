<div align="center">

# Track360

**360° 全景视频单目标跟踪框架：让普通的单目标跟踪器直接用于 360° 全景视频。**

球面多视图搜索 · 跨缝融合 · 球面运动预测 · GPU 重采样

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c)](https://pytorch.org)
[![Backend](https://img.shields.io/badge/backend-ARTrackV2--B--256-6f42c1)](https://github.com/miv-xjtu/artrack)
[![Status](https://img.shields.io/badge/status-V2%20重构中-orange)](docs/V2Plan.md)

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
3. 所有视图从 ERP 帧重采样（支持 CUDA），**一次性批量**送入后端 **ARTrackV2-B-256**；
4. 局部框回投到球面后，经过**跨缝融合**和**自适应状态机**，**每帧只提交一次**结果。

Geometry、Controller 和 I/O 只依赖 `core/` 中定义的协议，不依赖具体模型，因此更换跟踪后端不会影响其余部分。

> **项目状态**：已完成与比赛代码的解耦和结构整理，正在推进 **V2.0**（V1 为比赛版本）：在公开 benchmark [360VOT](https://360vot.hkustvgd.com) 上评测并超过已发表的 360 跟踪基线，同时完成效率优化和产品化。分阶段路线见 **[V2Plan](docs/V2Plan.md)**。

## 核心特性

| 模块 | 已实现的能力 |
|---|---|
| **球面几何** | ERP ↔ 球面 ↔ 透视视图转换；局部框边界一次回投得到 BFoV 与 ERP 框；循环（跨缝）区间与最小覆盖弧 |
| **GPU 重采样** | ERP 帧只上传一次显存，透视网格和跨缝双线性采样都在 CUDA 中完成；与 CPU 参考实现对比，P99 像素误差为 0 |
| **多视图搜索** | 按预测目标角尺寸生成四角视图（默认每帧一轮 4 个视图）；可选第二轮以第一轮最佳候选为中心再搜索；另有随预测中心旋转的 cubemap 规划器 |
| **融合** | 跨缝两框融合：几何平均 + 一致性奖励，并限制分数上限 |
| **运动模型** | 切平面 Huber 拟合的球面速度与 log 尺度估计，不会在 ±180° 经线处跳变 |
| **状态机** | 根据最近分数的分位数（ScoreGroup）自适应阈值，在 `TRACKING` 和 `UNCERTAIN` 之间切换 |
| **帧事务** | 带 revision 校验的帧事务，中间轮次不会污染已提交的状态；单帧出错时仍保证每帧一行输出 |
| **评估** | 循环 ERP IoU、球面 BFoV IoU（按 cos 纬度加权）、大圆中心误差、success 曲线 / AUC、跟踪丢失率 |

## 系统架构

```text
视频 ──▶ 预取解码线程 ──▶ ERP 帧
                            │
                            ▼
           ┌──────── TrackController ◀──────────────────┐
           │  规划视图（基于预测 BFoV，第 1 / 2 轮）       │
           ▼                                            │
   Geometry：ERP → N 个透视视图（256×256）               │
           │                                            │
           ▼                                            │
   跟踪后端：ARTrackV2-B-256（N 个视图一个 batch）        │
           │                                            │
           ▼                                            │
   Geometry：局部框 → 球面（BFoV / 跨缝 ERP 框）          │
           │                                            │
           ▼                                            │
   融合 + 打分 + 状态机 ────────── 每帧一次提交 ──────────┘
           │
           ▼
   结果输出（每帧一行）
```

```text
src/track360/
├── cli.py          统一命令行入口
├── core/           数据类型、协议、配置 schema、错误类型
├── geometry/       球面数学、BFoV 投影、跨缝区间、CUDA 重采样
├── controller/     视图规划、候选融合、运动预测、状态机、帧事务
├── backends/       ARTrackV2 推理会话与批量适配
├── runtime/        组件装配与逐帧循环
├── datasets/       360VOT / AirSim360 / 视频 / 图像序列读取
├── io/             图像与视频读取、结果写入
├── evaluation/     平面 / 循环 / 球面指标与性能统计
├── visualization/  中间视图与结果图
└── third_party/    上游 ARTrackV2 模型代码（推理子集）
```

详见 [系统架构](docs/architecture.md)。

## 安装

环境要求：Python 3.11+、NVIDIA GPU 及 CUDA 版 PyTorch 2.11。

```bash
git clone https://github.com/CaryHe-maker/360-VideoTargetingSystem.git
cd 360-VideoTargetingSystem
pip install -e ".[dev]"
```

模型权重不随仓库分发：请下载官方 ARTrackV2-B-256 checkpoint（约 1.6 GB），放到 `models/artrackv2_b_256.pth.tar`，下载方式和校验见 [models/README.md](models/README.md)。

## 快速上手

```bash
# 跟踪一个视频：初始框为第 0 帧的 ERP 像素框 x,y,width,height
track360 track --input path/to/video.mp4 --init-box 1200,640,180,140 --output outputs/result.txt

# AirSim360 格式数据：先列出首帧实例，再跟踪其中一个
track360 list-instances path/to/airsim360_seq
track360 airsim360 --dataset-root path/to/airsim360_seq --target-instance <id> --output outputs/tracking.txt
```

- 不传 `--config` 时使用 [`configs/default.yaml`](configs/default.yaml)；
- 输出每帧一行 `x,y,width,height`（ERP 像素坐标，跨缝目标的 `x + width` 可以超过图像宽度）；
- 带真值评估的完整产物使用 `python tools/run_airsim360_dataset.py`。

更多用法和退出码见 [快速上手](docs/getting-started.md)，参数说明见 [配置说明](docs/configuration.md)。

## Benchmark

V2 阶段会在公开数据集 **[360VOT](https://360vot.hkustvgd.com)**（ICCV 2023，120 条序列，约 113K 帧）上给出完整结果，参数只在 360VOS 训练集上调整。计划的对比方法：

| 方法 | 说明 |
|---|---|
| ARTrackV2 直接跟踪 ERP | 最简单的基线：直接在全景图上跟踪 |
| 论文基线 | 360VOT 论文中报告的结果，例如 AiATrack-360（S<sub>dual</sub> 0.534） |
| 单局部视图 | 以上一帧 BFoV 为中心生成一个透视视图 |
| **本项目** | 完整系统，并逐个组件做消融 |

指标与官方 toolkit 一致：dual success（S<sub>dual</sub>，AUC）、dual precision（P<sub>dual</sub>）、angle precision（P<sub>angle</sub>），另外在固定 GPU 上报告 FPS 和 P50 / P95 延迟。数据集选型和评测协议见 [Benchmark 数据集](docs/benchmark.md)。

## 测试

```bash
pytest                            # 没有 CUDA 时，GPU 几何测试会自动跳过
pytest -m "not slow"              # 跳过耗时约 1.5 分钟的开关组合回归
ruff check src tests tools
```

## 文档

全部文档见 [docs/](docs/README.md)：
[快速上手](docs/getting-started.md) ·
[系统架构](docs/architecture.md) ·
[配置说明](docs/configuration.md) ·
[Benchmark](docs/benchmark.md) ·
[V2Plan](docs/V2Plan.md) ·
[评测记录](docs/evaluation-log.md) ·
[历史实验结论](docs/experiments.md)

## 路线图

- [x] 移除比赛专用入口、Docker 分层和死代码，重新组织包结构，统一 CLI
- [x] 文档重组为中文文档集，确定公开 benchmark 测试集（360VOT）
- [x] 把后端调参用的环境变量迁入 YAML 配置，统一测试与运行行为；CI、运行记录、端到端回归测试
- [x] 项目、Python 包和命令行统一命名为 Track360 / `track360`
- [ ] 统一为 PEP 8 命名，提供 Python API
- [x] 权重不再通过 Git LFS 随仓库分发
- [ ] 权重发布到 Hugging Face Hub / Releases，提供自动下载与校验
- [ ] 360VOT / 360VOS 数据加载器，评测结果与官方 toolkit 交叉验证
- [ ] 消融与效率 benchmark（GPU 几何、流水线、FP16、TensorRT）
- [ ] 后端注册表，新增 OSTrack 后端和轻量速度档
- [ ] CI、文档站、Demo GIF、Gradio 在线演示

## 致谢

- [ARTrack / ARTrackV2](https://github.com/miv-xjtu/artrack)（Apache-2.0）：跟踪后端。`src/track360/third_party/artrackv2` 中的模型代码来自官方实现。
- [360VOT](https://github.com/HuajianUP/360VOT)：全景跟踪 benchmark、BFoV 表示和评测协议。
- [OSTrack](https://github.com/botaoye/OSTrack)、[pytracking](https://github.com/visionml/pytracking)：跟踪器接口与评测设计的参考。

## 许可证

Track360 使用 [Apache License 2.0](LICENSE)。`src/track360/third_party/artrackv2/` 中的代码来自 ARTrack，同样使用 Apache-2.0，原许可证保留在该目录下，来源说明见 [NOTICE](NOTICE)。模型权重和 360VOT / 360VOTS 数据集不属于本仓库，分别遵循其发布方的许可。
