# Datasets：数据读取

Datasets 把磁盘上的各种格式统一成 `FramePacket` 流，跟踪流程不关心数据来自哪里。

| 文件 | 职责 |
|---|---|
| `datasets/vot360.py` | 360VOT benchmark：序列发现、帧读取、四种真值标注；也读 360VOS 训练序列 |
| `datasets/vots_info.py` | 序列信息表 `360vots-info.csv`：两个 benchmark 的序列对应关系、挑战属性 |
| `datasets/mask_labels.py` | 从 360VOS 的分割掩码拟合 360VOT 格式的标注 |
| `datasets/tune_split.py` | 排除会泄漏测试集的训练序列，选出 tune 集 |
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

**不解压也能直接读**：每次只从 zip 里取出当前这一帧，不会解压整个压缩包。zip 对 JPG 做了 deflate 压缩（压缩率只有 2%），所以取一帧要多做一次解压，实测每帧约 1.9 毫秒，读解压后的文件约 0.5 毫秒。相比之下 JPG 解码约 25 毫秒，跟踪一帧约 100–185 毫秒，这点差别可以忽略。同一条序列的目录和 zip 同时存在时，使用目录。

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

## 序列信息表与挑战属性

`360vots-info.csv` 每行是一段源视频片段，记录它在 360VOS 中的划分和编号、在 360VOT 中的编号（如果它同时是 360VOT 的测试序列），以及 20 个挑战属性（`CB` 跨缝、`HL` 高纬度、`FMS` 球面快速运动、`LFoV` 大视场等）。

```python
from track360.datasets.vots_info import loadVotsInfo

info = loadVotsInfo("360vots-info.csv")
info.votAttributes()["0001"]        # 360VOT 序列 0001 的属性集合
info.vosAttributes("train")["006"]  # 360VOS 训练序列 006 的属性集合
```

## 360VOS 训练集与 tune 集

### 训练集和测试集的重叠

**360VOS 的 170 条训练序列里，有 97 条就是 360VOT 的测试序列**（同一段视频片段）。另有 5 条虽然不是同一个片段，但和某条测试序列剪自同一个源视频。在这些序列上调参等于在测试集上调参，所以全部排除。

| 排除原因 | 数量 |
|---|---:|
| 就是 360VOT 测试序列 | 97 |
| 与 360VOT 测试序列剪自同一个源视频 | 5 |
| 多目标序列 | 1 |
| 掩码与帧的文件名对不上（`052`） | 1 |
| **剩余可用** | **66** |

排除列表在 [`configs/splits/360vos_train_excluded.csv`](../../configs/splits/360vos_train_excluded.csv)。

### tune 集

从可用的 66 条里选 25 条（共 14,923 帧），列表在 [`configs/splits/360vos_tune.txt`](../../configs/splits/360vos_tune.txt)。选法是确定性的：每一步挑“属性在已选序列中最稀缺”的那条，并列时取更短的；超过 1000 帧的序列不选，保证跑一轮 tune 集的时间可控。选择只看属性表和帧数，不看任何跟踪结果。

### 标注来自掩码

训练序列的压缩包里只有 `image/` 和 `mask/`，没有 `label.json`。`mask_labels.py` 从掩码拟合出标注：

- `bbox`：掩码占据的列取最短的循环区间（目标跨缝时仍是一个框），行取上下界；
- `bfov`：把掩码轮廓上的像素转成球面方向，拟合最紧的球面框；
- 掩码为空的帧记为目标缺席；`rbbox` / `rbfov` 没有单独拟合，直接重复不带旋转的结果。

这些标注不是作者的标注流程生成的，和官方标注不会逐位相同。在 25 条同时有掩码和官方标注、帧数一致的序列上抽 300 帧对比：

| 项目 | 结果 |
|---|---|
| 框 IoU | 中位数 0.990，98.6% 的帧高于 0.9 |
| BFoV 中心偏差 | 中位数 0.05°，95 分位 0.81° |
| BFoV 尺寸偏差 | 中位数 0.1°，95 分位水平 1.0° / 垂直 2.9° |
| 目标是否存在 | 全部一致 |

少数帧偏差较大（BFoV 尺寸最多差几十度），集中在大视场目标上：官方对超过 90° 的目标用了另一种 BFoV 定义。所以 **tune 集的分数只用来比较方法之间的相对好坏，不能和 360VOT 测试集的分数直接比较**。

### 生成步骤

```bash
# 1. 生成排除列表和 tune 列表（结果已提交，一般不需要重跑）
python tools/prepare_tune_set.py split --info <360vots-info.csv> --train-root <train>

# 2. 为 tune 序列生成标注，放在仓库外；4K 掩码逐帧解码，全部 25 条约需 40 分钟
python tools/prepare_tune_set.py labels --train-root <train> --label-root <labels/train>

# 3. 可选：重新做上表的对比
python tools/prepare_tune_set.py validate --info <360vots-info.csv> --train-root <train> --vot-root <360VOT-test>
```
