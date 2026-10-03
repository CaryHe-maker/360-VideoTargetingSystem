# Datasets：数据读取

Datasets 把磁盘上的各种格式统一成 `FramePacket` 流，跟踪流程不关心数据来自哪里。

| 文件 | 职责 |
|---|---|
| `datasets/vot360.py` | 360VOT benchmark：序列发现、帧读取、四种真值标注 |
| `datasets/airsim360_source.py` | AirSim360 仿真序列（开发数据，不是 benchmark） |
| `datasets/pseudo_track_builder.py` | 从 AirSim360 的实例掩码生成伪真值框 |
| `datasets/registry.py` | 按格式名创建数据源：`openDataset(root, format=..., sequenceId=...)` |
| `io/video_source.py` | 通用的视频 / 图像序列读取（`track360 track` 使用） |
| `io/image_reader.py` | 图像解码 |

## 图像解码

RGB 帧统一用 OpenCV 解码，支持 PNG 和 JPG。4K 帧解码约 0.15 秒。

`readImageArray()` 保留了原来的 PNG 解码器，只用于 AirSim360 的分割掩码这类标签图：它们的通道布局和位深必须原样保留，不能经过颜色转换。

## 360VOT

### 目录结构

数据集根目录下每条序列是一个目录，或者是发布时的 zip 压缩包，两种可以混用：

```text
<root>/0001/image/000000.jpg …      <root>/0002.zip
<root>/0001/label.json                 （压缩包内的结构相同）
```

**不解压也能直接读**：zip 内的 JPG 没有再压缩，直接读取每帧约 40 毫秒，和读解压后的文件差不多。同一条序列的目录和 zip 同时存在时，使用目录。

注意：官方 toolkit 的评测脚本只认解压后的目录结构。需要运行官方脚本时，至少要把每条序列的 `label.json` 解压出来。

### 标注

`label.json` 以帧文件名为键，每帧有四种标注，角度单位是度：

| 键 | 字段 | 说明 |
|---|---|---|
| `bfov` | `clon, clat, fov_h, fov_v, rotation` | 球面框，`rotation` 恒为 0 |
| `rbfov` | 同上 | 带旋转的球面框 |
| `bbox` | `cx, cy, w, h, rotation` | ERP 像素框，**中心点表示**，`rotation` 恒为 0 |
| `rbbox` | 同上 | 带旋转的 ERP 像素框 |

- 宽或高为 0 表示这一帧**目标不在画面中**；
- 目标跨越左右边界时，`bbox` 的范围会超出 `[0, 宽)`，左右两侧都可能超出。

### 坐标约定的对应关系

| 360VOT | 本项目 | 说明 |
|---|---|---|
| `clon`（向右为正，图像中心为 0） | yaw | 相同 |
| `clat`（向上为正） | pitch | 相同 |
| 像素下标 `u`（像素中心在 `u + 0.5`） | 像素边缘坐标 `x` | `x = u + 0.5` |
| 跨缝框超出图像范围 | `x` 保持在 `[0, 宽)`，`x + width` 可以超过宽度 | 读取时自动换算 |

半像素偏移用已下载的数据核对过：在 3.7 万帧靠近赤道的小目标上，`bfov` 中心换算到像素后比 `bbox` 中心在水平方向偏右 0.50 像素。垂直方向只偏 0.09 像素，原因没有查清（可能是两种标注各自从掩码拟合造成的偏差），目前按 toolkit 的定义统一加 0.5。

### 接口

```python
from track360.datasets import Vot360Dataset

dataset = Vot360Dataset("E:/datasets/360VOTS/360VOT-test")
sequence = dataset.sequence("0001")

sequence.frameCount            # 帧数
sequence.frameSize             # (宽, 高)
sequence.readRgb(0)            # uint8 RGB 图像
sequence.initialBfov()         # 第 0 帧的 BFoV，用于初始化跟踪
sequence.annotation(10)        # 项目类型的标注；目标不在时各字段为 None
sequence.groundTruth("bfov")   # (N×5 数组, 目标是否存在的掩码)
```

`annotation()` 和 `groundTruth()` 面向两种不同的用途：

- `annotation()` 返回本项目的类型（`BFoV`、`BBoxXYWH`，弧度和像素边缘坐标），给跟踪初始化和可视化用；
- `groundTruth()` 返回与官方 toolkit **完全相同**的数组布局（度、像素下标、跨缝框保留超出范围的坐标），给评测用，这样评测结果才能与官方脚本对齐。

逐帧读取用 `Vot360DataSource`，它符合通用的数据源接口，可以直接交给 `runTracking()`：

```python
from track360.datasets import openDataset

source = openDataset(root, format="360vot", sequenceId="0001")
```

### 尚未实现

- 属性标签（跨缝、极点、快速运动等）。标签在数据集发布的属性表里，等下载完成、确认文件格式后再接入；
- 360VOS 训练集的读取，以及与 360VOT 测试集的序列去重。
