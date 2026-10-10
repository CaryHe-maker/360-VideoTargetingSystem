<div align="center">

# Track360

**让普通的单目标跟踪器直接用于 360° 全景视频。**

球面局部视图 · 跨缝回投 · 球面运动预测 · 每帧一次前向

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c)](https://pytorch.org)
[![Backend](https://img.shields.io/badge/backend-ARTrackV2--B--256-6f42c1)](https://github.com/MIV-XJTU/ARTrack)
[![Status](https://img.shields.io/badge/status-V2.0%20开发中-orange)](docs/V2Plan.md)

</div>

---

## 简介

等距柱状投影（ERP）的 360° 全景帧会让普通跟踪器失效：

- 越靠近两极，目标的形变越严重；
- 目标可能跨越图像左右边界（经线接缝，下文称“跨缝”）；
- 整帧分辨率太大，无法直接送入 256×256 输入的跟踪器。

Track360 不改动跟踪器本身，只改变**“看哪里”**，做法与 360VOT 论文的 360 跟踪框架一致：

1. 在单位球面上用 **BFoV**（`clon, clat, fov_h, fov_v`）表示目标；
2. 每一帧围绕运动模型预测的方向取**一个**局部视图，目标约占视图边长的四分之一，和跟踪器训练时见到的搜索区域一样；
3. 视图送入 **ARTrackV2-B-256**（序列级模型：同时输入前 7 帧的目标框），每帧一次前向；
4. 局部框回投到球面，得到 BFoV 和可以跨缝的 ERP 框。

## 现状和数字

V2.0 还没有发布。下面的数字来自项目自己的开发集（360VOS 训练集里的序列），**不是 360VOT 官方测试集的结果**；官方测试集上只有早期代码的数字，当前代码的还没有跑。

| 配置 | 怎么选 | S<sub>dual</sub>（dev train，33 条） | 每帧前向 | FPS（RTX 4060 Laptop） |
|---|---|---:|---:|---:|
| 直接在 ERP 上跟踪（对照） | — | 见下 | 1 | 没有重测 |
| **基线 1：默认** | `--preset default` | 0.48–0.49 | 1.00 | 39.6 |
| 基线 1，速度档 | `--preset default --precision tf32` | 0.48（与上一行分不出高低） | 1.00 | 43.2 |
| 基线 2：丢失后重新寻找目标（实验） | `--preset loss_handling` | 0.517 | 平均 1.16 | 约 25 |

- 在全部 66 条开发序列上，基线 1 是 0.504，直接在 ERP 上跟踪是 0.412。
- **单次运行的分数有约 0.009 的标准差来自数值上的偶然**：给输入加一个看不见的噪声，同样的代码在 0.470–0.495 之间。所以基线 2 相对基线 1 的 +0.024 还不能算确定的提升，它是实验功能，默认关闭。
- 用真值在丢失后把跟踪器放回目标，上限是 0.65 左右：丢失后的重新寻找还有很大空间，这是 V2.1 的方向。
- 速度是完整流程的数字（读图、解码、取视图、前向、回投），4K 的 JPEG 序列。

每个数字的来历、测法和局限见 [两条基线](docs/baselines.md) 和 [评测记录](docs/evaluation-log.md)。

## 安装

环境要求：Python 3.11+，NVIDIA GPU 和 CUDA 版 PyTorch 2.11。

```bash
git clone https://github.com/CaryHe-maker/360-VideoTargetingSystem.git
cd 360-VideoTargetingSystem
pip install -e ".[dev]"
track360 download        # ARTrackV2-B-256 权重，约 1.6 GB，下载后核对 SHA-256
```

权重不随仓库分发，由 `track360 download` 从作者发布的地址下载；手动下载的方法见 [models/README.md](models/README.md)。上游仓库声明该项目不用于商业用途，使用前请自行确认许可。

## 快速上手

```bash
# 第 0 帧的目标用 ERP 像素框给出：x,y,width,height
track360 track --input path/to/video.mp4 --init-box 1200,640,180,140 --output out/result.txt

# 或者用球面框（度）：clon,clat,fov_h,fov_v；同时输出一段演示视频
track360 track --input path/to/frames/ --init-bfov=12.0,-3.5,20,35 \
  --output out/result.txt --demo out/demo.mp4 --gif out/demo.gif

# 速度档；丢失处理
track360 track --input video.mp4 --init-box 1200,640,180,140 --output out/a.txt --precision tf32
track360 track --input video.mp4 --init-box 1200,640,180,140 --output out/b.txt --preset loss_handling
```

- 输出每帧一行 `x,y,width,height`（ERP 像素坐标，跨缝目标的 `x + width` 可以超过图像宽度）；
- 演示视频左边是全景画面和结果，右边是跟踪器实际看到的局部视图；
- 读视频文件需要系统里装有 ffmpeg，图像序列目录不需要。

Python 里：

```python
from track360.api import Track360Tracker

with Track360Tracker.fromPretrained("default") as tracker:
    results = tracker.track("panorama.mp4", initBfov=(12.0, -3.5, 20.0, 35.0))
    for result in results:
        print(int(result.frameIndex), result.bbox, result.status.name)
```

更多参数见 [快速上手](docs/getting-started.md) 和 [配置说明](docs/configuration.md)。

## 怎么工作

```text
帧 ──▶ 解码线程 ──▶ ERP 帧
                      │
        ┌─────── TrackController ◀────────────────────────┐
        │  运动模型预测这一帧的方向和大小，规划一个视图       │
        ▼                                                 │
   Geometry：ERP → 局部视图（256×256；大目标用球面视图）    │
        │                                                 │
        ▼                                                 │
   ARTrackV2-B-256：模板 + 视图 + 前 7 帧的框 → 一个框      │
        │                                                 │
        ▼                                                 │
   Geometry：局部框 → 球面 BFoV 和跨缝的 ERP 框            │
        │                                                 │
        ▼                                                 │
   状态判定 ─────────────── 每帧提交一个结果 ───────────────┘
```

| 模块 | 做什么 |
|---|---|
| **球面几何** | ERP、球面、局部视图之间的转换；局部框边界一次回投得到 BFoV 和 ERP 框；跨缝区间和最小覆盖弧。目标搜索区域超过 120° 时改用球面视图（以目标为中心的局部 ERP），避免透视投影在大视场下的拉伸 |
| **视图规划** | 每帧一个视图：以运动预测为中心，边长是目标平均尺寸的 4 倍 |
| **运动模型** | 切平面上 Huber 拟合的球面速度和 log 尺度，不会在 ±180° 经线处跳变 |
| **后端** | ARTrackV2 的序列级用法：轨迹输入和每帧更新的外观特征。上游模型代码的推理子集在 `third_party/` 下 |
| **状态** | 每帧报告 `TRACKING` / `UNCERTAIN` / `LOST`。默认配置下它只是报告，不影响跟踪 |
| **丢失处理（实验）** | 用后端分数和对第 0 帧模板的外观相似度（DINOv2）相对本序列自身水平的下降来判断丢失；可疑帧不更新后端的外观记忆；丢失后先原地重看，再用放大的视图定位、正常大小的视图确认，分数够高才跳回去 |
| **逐帧协议** | “规划 → 提交”两步，带 revision 校验；单帧出错时仍然保证每帧一行输出 |

```text
src/track360/
├── api.py          Python 入口：Track360Tracker
├── cli.py          命令行入口
├── hub.py          权重下载和校验
├── core/           数据类型、协议、配置、错误类型
├── geometry/       球面数学、视图投影、跨缝区间
├── controller/     视图规划、运动预测、状态机、丢失处理
├── backends/       ARTrackV2 推理会话、外观验证器
├── runtime/        组件装配和逐帧循环、批量评测
├── datasets/       360VOT / AirSim360 / 视频 / 图像序列读取
├── io/             图像和视频读取、结果写入
├── evaluation/     360VOT 指标、丢失率、配对 bootstrap、计时
├── visualization/  中间视图、结果图、演示视频
└── third_party/    上游 ARTrackV2 模型代码（推理子集）、360VOT 指标代码
```

详见 [系统架构](docs/architecture.md) 和各 [模块文档](docs/README.md)。

## 评测

```bash
track360 benchmark run  --dataset-root <360VOT> --output-root outputs/run --method ours
track360 benchmark eval --dataset-root <360VOT> --output-root outputs/run
track360 benchmark compare --dataset-root <360VOT> \
  --baseline outputs/a:ours --candidate outputs/b:ours
```

指标与官方 toolkit 一致：dual success（S<sub>dual</sub>，AUC）、dual precision（P<sub>dual</sub>）、angle precision（P<sub>angle</sub>），另有丢失率和配对 bootstrap 的 95% 区间。数据集的准备和划分见 [Benchmark 数据集](docs/benchmark.md)。

比较两种做法时请注意上面说的偶然波动：单次运行之间小于约 0.02–0.03 的差别不足以下结论。

## 测试

```bash
pytest                    # 不需要 GPU 和权重
ruff check src tests tools
```

回归测试里有一组在合成序列上逐位比对的轨迹；默认配置在真实数据上的结果也要求与记录逐字节相同，做法见 [两条基线](docs/baselines.md)。

## 文档

[快速上手](docs/getting-started.md) ·
[系统架构](docs/architecture.md) ·
[配置说明](docs/configuration.md) ·
[两条基线](docs/baselines.md) ·
[Benchmark 数据集](docs/benchmark.md) ·
[评测记录](docs/evaluation-log.md) ·
[V2 计划](docs/V2Plan.md) ·
[全部文档](docs/README.md)

## 路线图

- [x] 与比赛代码解耦，重组包结构，统一命令行和配置
- [x] 360VOT / 360VOS 数据加载，评测结果与官方 toolkit 交叉验证
- [x] 换成与跟踪器训练条件一致的视图取法；大目标的球面视图；序列级模型
- [x] 速度：默认配置 22 → 40 FPS，结果不变；TF32 速度档
- [x] 权重下载和校验、Python API、演示视频
- [ ] 在 360VOT 官方测试集上给出当前代码的结果
- [ ] V2.0 发布
- [ ] V2.1：丢失后的重新寻找（出事时把位置定住、候选的确认）
- [ ] 更多后端；TensorRT

## 引用

见 [CITATION.cff](CITATION.cff)。使用本项目时请同时引用 ARTrackV2 和 360VOT。

## 致谢

- [ARTrack / ARTrackV2](https://github.com/MIV-XJTU/ARTrack)：跟踪后端。`src/track360/third_party/artrackv2` 中的模型代码来自官方实现。
- [360VOT](https://github.com/HuajianUP/360VOT)：全景跟踪 benchmark、BFoV 表示和评测协议。
- [DINOv2](https://github.com/facebookresearch/dinov2)：丢失处理里的外观验证器。
- [OSTrack](https://github.com/botaoye/OSTrack)、[pytracking](https://github.com/visionml/pytracking)：跟踪器接口和评测设计的参考。

## 许可证

Track360 使用 [Apache License 2.0](LICENSE)。`src/track360/third_party/artrackv2/` 中的代码来自 ARTrack，同样使用 Apache-2.0，原许可证保留在该目录下，来源说明见 [NOTICE](NOTICE)。模型权重不属于本仓库，其使用条件以上游的声明为准。
