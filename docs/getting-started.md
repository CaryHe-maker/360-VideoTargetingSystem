# 快速上手

## 环境要求

- Python 3.11 或 3.12
- NVIDIA GPU 和 CUDA 版 PyTorch 2.11

## 安装

```bash
git clone https://github.com/CaryHe-maker/360-VideoTargetingSystem.git
cd 360-VideoTargetingSystem
pip install -e ".[dev]"
```

安装后会注册命令行入口 `track360`，也可以用 `python -m track360` 调用。

## 模型权重

默认配置读取 `models/artrackv2_b_256.pth.tar`（官方 ARTrackV2-B-256 checkpoint，约 1.6 GB）。权重不随仓库分发，需要单独下载并放到该路径，下载方式和 SHA-256 校验见 [models/README.md](../models/README.md)。

## 命令行

```text
track360 <command> [options]

commands:
  track           从初始 ERP 框开始跟踪一个视频或图像序列中的目标
  airsim360       跟踪 AirSim360 序列中的一个实例，可输出诊断图
  list-instances  列出 AirSim360 序列首帧中可见的实例 ID
```

不传 `--config` 时自动使用仓库内的 `configs/default.yaml`。

### 跟踪视频

```bash
track360 track --input path/to/video.mp4 --init-box 1200,640,180,140 --output outputs/result.txt
```

- `--input`：视频文件或图像序列目录（目录模式可加 `--recursive`）。
- `--init-box`：第 0 帧目标的 ERP 像素框 `x,y,width,height`。
- `--output`：结果文本路径。每帧一行 `x,y,width,height`（ERP 像素坐标）；跨缝目标的 `x + width` 可以超过图像宽度。

### AirSim360 数据

```bash
track360 list-instances path/to/airsim360_seq
track360 airsim360 --dataset-root path/to/airsim360_seq --target-instance <id> \
  --output outputs/airsim360/tracking.txt --result-visual-root outputs/airsim360/visual
```

带真值评估的完整产物（逐帧 IoU、汇总指标、实例目录）使用：

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

## 测试

```bash
pytest
ruff check src tests tools
```

GPU 几何测试在没有 PyTorch 或 CUDA 时会自动跳过；其余测试只需要 NumPy、OpenCV 和 PyYAML。
