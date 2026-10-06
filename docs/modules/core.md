# Core：数据类型、协议与错误

Core 定义所有模块共享的值对象、协议、配置和异常，本身不执行任何模型、几何或 I/O 算法。

| 文件 | 职责 |
|---|---|
| `core/types.py` | 帧、视域、观测、搜索计划和结果类型 |
| `core/protocols.py` | 模块之间的最小接口（几何、后端、控制器、数据源、结果输出） |
| `core/config.py` | 严格的 YAML schema 与参数约束，见 [配置说明](../configuration.md) |
| `core/errors.py` | 分层异常类型 |

## 图像与空间类型

- `FramePacket`：一帧的唯一跨模块容器，包含序列 ID、帧号、单调时间戳、ERP RGB 图像和可选的分割掩码（`SegmentationPlane`）。掩码必须与 RGB 对齐。
- `SphericalPoint`：同时保存 yaw / pitch 和三维单位向量。跨经线和极点附近的计算使用单位向量，配置、显示和文件格式使用 yaw / pitch。
- `BFoV`：用球面中心、水平视场角、垂直视场角和 roll 描述球面上的目标范围。
- `BBoxXYWH`：ERP 或局部图像上的像素框，必须结合所在图像的宽高解释。

## 查询链类型

一帧中的数据按以下顺序流动：

| 类型 | 含义 |
|---|---|
| `ViewSpec` | 请求裁剪的球面方向和输出图像尺寸 |
| `LocalView` | 几何模块实际裁出的局部 RGB（或 CUDA 张量）及其 `ViewSpec` |
| `LocalObservation` | 后端在局部图中预测的框、模型分数和外观分数 |
| `LocalBoxProjection` | 局部框四条边一次回投得到的 BFoV、ERP 框、边界点和包络膨胀比 |
| `ProjectedObservation` | 回投结果 + 外观概率、运动分、`singleScore` 和投影质量诊断 |

每一帧的 `LocalView` 相机中心都不同，局部像素坐标不能跨帧比较，所以控制器只使用 `ProjectedObservation`（球面 / ERP 坐标）。

## 控制与结果类型

- `SearchPlan`：绑定帧身份、状态 revision、这一帧的搜索视图、模板命令和运动预测；
- `TrackResult`：对外结果，包含 ERP 框、BFoV、置信度、状态（`TRACKING` / `UNCERTAIN`；`LOST` 保留在类型里，目前不会出现）、`valid` 和结果来源。`valid=False` 时仍可能带有运动预测框，但这个框不会写入可靠的测量历史。

## 错误类型

所有预期内的失败都继承自 `Track360Error`：

| 异常 | 场景 |
|---|---|
| `ConfigError` | 配置缺失、未知字段或取值非法 |
| `DecodeError` | 视频 / 图像解码失败 |
| `ModelError` | 权重加载或推理失败 |
| `GeometryError` | 投影结果非有限值或越界 |
| `ProtocolError` | 逐帧协议、revision 或数据形状不满足约定 |
| `OutputError` | 结果写入失败或帧数不一致 |

## 修改原则

类型的构造函数负责范围和形状校验。新增字段时要写清单位、坐标系、是否可为空以及由谁写入，并同步更新 `tests/unit/test_core_types.py`。
