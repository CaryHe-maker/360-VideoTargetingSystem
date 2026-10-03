# Visualization：诊断产物

Visualization 只读取帧、视图、观测和结果来生成诊断图片，不参与任何算法决策。

| 文件 | 职责 |
|---|---|
| `visualization/recorder.py` | 三个阶段的中间产物：局部视图、后端框、回投框 |
| `visualization/result.py` | 每帧一张的最终 ERP 结果图 |
| `visualization/image.py` | 画框、标签和跨缝框 |
| `visualization/png.py` | 无额外依赖的 PNG 写入 |
| `visualization/time_counter.py` | 处理区间计时与 `time.json` |

## 中间产物

由 YAML 的 `visualization` 段控制：

| 阶段 | 内容 |
|---|---|
| `local_rgb` | 每个局部透视视图的原始图像 |
| `backend_box` | 局部视图上后端预测的框和原始分数 |
| `geometry_box` | 回投到原始 ERP 图上的框，带 SingleScore、运动分、外观概率和包络膨胀比 |

## 最终结果图

`ResultVisualizationRecorder` 只显示控制器已提交的结果，不会重新选择候选：

- 普通框画一个矩形；跨缝框通过 `splitSeamBox()` 在右侧和左侧各画一段，共用一个标签；
- 标签格式为 `state=<状态>/rounds=<轮数>/stateScore=<分数>`，初始化帧显示 `rounds=0` 和 `stateScore=N/A`；
- 颜色、线宽和标签间距是代码常量，不属于跟踪参数。

排查结果异常时，建议按逆序检查：`geometry_box`（分数、膨胀比）→ `backend_box`（原始框和分数）→ `local_rgb`（视图是否覆盖了目标）。最终框正确但 `valid=False`，通常表示结果来自弱观测或运动预测，而不是绘图错误。

## 计时边界

所有图片都在处理计时停止之后写出。开启可视化会增加进程总耗时和磁盘占用，但不会改变 `time.json` 中的数值。
