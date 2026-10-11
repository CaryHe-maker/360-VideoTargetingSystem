# 快速上手

## 环境要求

- Python 3.11 或更高
- NVIDIA GPU 和 CUDA 版 PyTorch 2.11（CPU 也能运行，但很慢）
- 读视频文件需要系统里装有 ffmpeg；图像序列目录不需要

## 安装

```bash
git clone https://github.com/CaryHe-maker/360-VideoTargetingSystem.git
cd 360-VideoTargetingSystem
pip install -e ".[dev]"
```

要用源码安装（`-e`）：配置文件在仓库的 `configs/` 下，不随包安装。安装后会注册命令行入口 `track360`。

## 模型权重

```bash
track360 download
```

把官方的 ARTrackV2-B-256 检查点（约 1.6 GB）下载到 `models/artrackv2_b_256.pth.tar`，并核对 SHA-256。下载中断后再运行一次会接着下；已经在位且校验通过的文件不会重下。

- `track360 download --check`：只核对已有的文件；
- `track360 download --url <地址>`：换一个来源（镜像，或者 `file:///...` 的本地副本）；
- 来源是作者发布在 Google Drive 上的文件。如果那边返回的是网页而不是文件（需要登录或确认），命令会报错并提示手动下载，方法见 [models/README.md](../models/README.md)。

上游仓库声明这个项目不用于商业用途，使用权重前请自行确认许可。

丢失处理（`--preset loss_handling`）另外需要 DINOv2 ViT-S/14，放在 `models/hub/` 下，第一次使用时由 `torch.hub` 下载。

## 命令行

```text
track360 <command> [options]

commands:
  track           跟踪一个视频或图像序列中的目标，可以同时输出演示视频
  download        下载模型权重并校验
  benchmark       在 360VOT 格式的数据集上运行和打分（即 tools/benchmark.py）
  airsim360       跟踪 AirSim360 序列中的一个实例，可输出诊断图
  list-instances  列出 AirSim360 序列首帧中可见的实例 ID
```

### 跟踪视频

```bash
track360 track --input path/to/video.mp4 --init-box 1200,640,180,140 --output outputs/result.txt
```

| 参数 | 说明 |
|---|---|
| `--input` | 视频文件或图像序列目录（PNG / JPG，目录模式可加 `--recursive`） |
| `--init-box` | 第 0 帧目标的 ERP 像素框 `x,y,width,height` |
| `--init-bfov` | 第 0 帧目标的球面框 `clon,clat,fov_h,fov_v`（度，360VOT 的约定：向右、向上为正）。与 `--init-box` 二选一 |
| `--output` | 结果文本。每帧一行 `x,y,width,height`（ERP 像素坐标）；跨缝目标的 `x + width` 可以超过图像宽度 |
| `--preset` | `default`（默认）：每帧一次前向；`loss_handling`：丢失后重新寻找目标，实验功能 |
| `--precision` | `fp32`（默认）或 `tf32`：更快，结果末位有差别，只在 RTX 30 系及之后的显卡上有效 |
| `--demo` / `--gif` | 输出并排的演示视频（MP4）或动图：左边是全景画面和结果，右边是跟踪器看到的局部视图 |
| `--max-frames` | 只处理前若干帧 |
| `--config` / `--weights` | 用自己的配置文件代替预设；权重不在 `models/` 时指定路径 |

`--output`、`--demo`、`--gif` 至少给一个。四种“预设 × 精度”的组合对应的数字见 [baselines.md](baselines.md)。

```bash
# 丢失处理 + TF32，同时出一段演示视频
track360 track --input frames/ --init-bfov=12.0,-3.5,20,35 \
  --preset loss_handling --precision tf32 --output out/result.txt --demo out/demo.mp4
```

`--init-bfov` 的值以负号开头时要写成 `--init-bfov=...`。

### Python API

```python
from track360.api import Track360Tracker

with Track360Tracker.fromPretrained("default", precision="tf32") as tracker:
    results = tracker.track("panorama.mp4", initBfov=(12.0, -3.5, 20.0, 35.0))
    for result in results:
        print(int(result.frameIndex), result.bbox, result.status.name, result.confidence)
```

- 权重在第一次调用 `track` 时加载，同一个对象可以接着跟踪别的视频；
- `track` 的 `source` 可以是视频文件、图像目录，或者任何带 `read()`、按顺序返回帧的对象；
- `initBox=(x, y, width, height)` 和 `initBfov=(clon, clat, fov_h, fov_v)` 二选一；
- `output=`、`demo=`、`gif=` 和命令行的同名参数一样；
- 返回每帧一个 `TrackResult`：`bbox`（ERP 像素）、`bfov`（球面框）、`confidence`（后端分数）、`status`（`TRACKING` / `UNCERTAIN` / `LOST`）、`valid`。

### Benchmark

```bash
track360 benchmark run --dataset-root <数据集> --output-root outputs/run --method ours
track360 benchmark eval --dataset-root <数据集> --output-root outputs/run
```

子命令和参数与 `python tools/benchmark.py` 相同，见 [Benchmark 数据集](benchmark.md#复现命令)。

### AirSim360 数据

```bash
track360 list-instances path/to/airsim360_seq
track360 airsim360 --dataset-root path/to/airsim360_seq --target-instance <id> \
  --output outputs/airsim360/tracking.txt --result-visual-root outputs/airsim360/visual
```

不传 `--config` 时使用 `configs/default.yaml`。带真值评估的完整产物（逐帧 IoU、汇总指标、实例目录）使用：

```bash
python tools/run_airsim360_dataset.py --dataset-root path/to/airsim360_seq \
  --config configs/default.yaml --target-instance <id>
```

## 退出码

| 退出码 | 含义 |
|---:|---|
| 0 | 成功 |
| 2 | 配置错误 |
| 3 | 解码错误 |
| 4 | 模型加载或推理错误 |
| 5 | 输出写入错误 |
| 10 | 几何或协议不变量被破坏 |
