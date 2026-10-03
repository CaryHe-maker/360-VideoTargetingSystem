# Geometry：球面几何与重采样

Geometry 负责 ERP 图像、球面方向、BFoV、局部透视视图和循环（跨缝）框之间的全部转换。

| 文件 | 职责 |
|---|---|
| `geometry/projection_math.py` | 球面点、单位向量、ERP 像素和透视射线之间的数学转换 |
| `geometry/bfov_projector.py` | 从 ERP 图像采样出一个局部透视视图（CPU / OpenCV） |
| `geometry/spherical_geometry.py` | 几何模块的门面：裁剪视图、框与 BFoV 互转、局部框回投 |
| `geometry/gpu_geometry.py` | 同一接口的 CUDA 实现 |
| `geometry/seam.py` | ERP 水平方向的循环区间运算 |

## 坐标约定

- 经度（yaw）范围为 [-π, π)，纬度（pitch）范围为 [-π/2, π/2]；
- ERP 像素 x 对应经度、y 对应纬度，水平方向是**循环**的；
- 局部视图是以 `ViewSpec.bfov.center` 为光轴的针孔相机，焦距由视场角和输出尺寸决定。在极点附近，相机基向量使用专门的回退，避免出现退化。

## 局部框回投

后端在局部视图中给出一个像素框后，Geometry 把它转换回球面：

1. 在框的四条边上各取 `boundarySamplesPerEdge` 个点，**只投影一次**到球面；
2. 用这组球面边界点迭代修正中心，再按水平 / 垂直角度范围拟合无旋转的 BFoV；
3. ERP 框直接由同一组边界点得到：x 取**最小循环覆盖区间**，y 取最小值 / 最大值。ERP 框不再从 BFoV 二次包络得到，避免框被额外放大；
4. 记录 `envelopeInflation`（包络膨胀比）、`normalizedRadius`、`edgeMargin` 等诊断字段，用来区分模型误差、透视边缘畸变和包络损失。

## 跨缝处理

ERP 的水平坐标是一个圆周，不是一条直线。

- **归一化**：`wrapPixelX()` 把任意 x 映射到 [0, W)。跨缝框表示为 x 接近 W、且 x + width > W；
- **最小覆盖区间**：`minimalCircularInterval()` 先排序，再找圆周上最大的空隙，去掉这段空隙后剩下的弧就是目标的最小覆盖。这比直接取最小值 / 最大值更适合同时出现在左右边缘的同一个目标；
- **拆分**：`splitSeamBox()` 把跨缝框拆成右段和左段，用于绘制、求交集和 IoU，逻辑上仍是一个框；
- **交集与 IoU**：两个框各自拆成最多两个 x 区间，逐对求交，再乘以 y 方向的交集。

> 任何直接使用 `max(x1, x2)` 计算 ERP 框的代码都需要重点审查。必须测试的场景包括：不跨缝、一个框跨缝、两个框都跨缝、刚好接触但面积为零、全宽框和极窄的边界框。

## GPU 重采样

`GpuGeometryImpl` 与 CPU 实现接口相同：

- ERP 帧只上传一次显存，同一帧的多个视图复用；
- 透视网格、旋转、球面投影和跨缝双线性采样都在 CUDA 中完成，网格按输出尺寸缓存；
- 输出直接是归一化后的 FP32 `[3, H, W]` 张量，可以直接送入后端；
- 与 CPU 结果对比，P99 像素误差为 0，最大误差 1 个灰度级（`tests/unit/test_gpu_geometry.py`）。

为了与 ARTrack 官方预处理在数值上一致，默认使用 CPU 路径；把 `geometry.resampler` 设为 `cuda` 启用 CUDA 路径。在 HiT 后端阶段的单序列测试中，GPU 几何把单帧 P50 延迟从 355.9 ms 降到 90.0 ms（见 [历史实验结论](../experiments.md)）。在 ARTrackV2 上需要重新做 A/B。
