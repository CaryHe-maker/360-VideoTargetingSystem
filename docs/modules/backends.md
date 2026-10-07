# Backends：跟踪器后端

当前只有一个后端：官方 **ARTrackV2-B-256**（[miv-xjtu/ARTrack](https://github.com/miv-xjtu/artrack)，CVPR 2024）。

| 文件 | 职责 |
|---|---|
| `backends/artrack_model.py` | `PyTorchARTrackV2Session`：加载官方网络、模板 / 搜索裁剪、坐标解码；`ARTrackBackend`：批量输入输出校验 |
| `backends/artrack_backend.py` | `TrackerBackendImpl`：实现 `core.protocols.TrackerBackend`，把模板 revision、局部框和控制器的模板命令串起来 |
| `backends/template_cache.py` | 模板缓存：anchor 模板与 recent 模板、revision 管理 |
| `backends/observation.py` | 把模型预测转换为 `LocalObservation` |
| `third_party/artrackv2/` | 上游模型代码的推理子集（ViT 主干、配置） |

## 推理流程

1. **初始化**：在第 0 帧的模板视图上，以目标框为中心裁出 128×128 模板并编码；
2. **搜索**：在这一帧的局部透视视图上，以模板框位置为中心裁出 256×256 的搜索区域，一次前向（`fullViewSearch: true` 时不再裁剪，整个视图就是搜索区域）；
3. **解码**：网络以 400 个 bin 自回归输出框坐标，按搜索裁剪的缩放系数映射回局部视图像素坐标并裁剪到视图范围内；
4. **分数**：使用网络 score head 的 sigmoid 输出。这个分数在当前权重上大多集中在 0.5 附近，更适合作为排序信号，而不是校准过的概率；
5. **回投**：局部框交给 Geometry 回投到球面，后续由 Controller 处理。

## 权重加载

- 默认路径为 `models/artrackv2_b_256.pth.tar`，加载 `net` 状态字典，缺少任何参数都会报错；
- 首先尝试 `torch.load(weights_only=True)`。官方压缩包中带有旧版训练统计对象，安全加载失败时才回退到兼容模式（只对显式指定的本地文件这样做）；
- 有 CUDA 时使用 GPU，否则使用 CPU。

## 已知限制

- 推理始终是 FP32，`model.precision` 只接受 `fp32`。FP16 / TensorRT 是 [V2Plan](../V2Plan.md) Phase 5 的工作；
- 后端在 `runtime/driver.py` 中直接创建，还没有注册表机制。

## 接入新后端

新后端只需要实现 `core/protocols.py` 中的 `TrackerBackend` 协议：

| 方法 | 说明 |
|---|---|
| `initialize(template, templateBox)` | 用第 0 帧的模板视图和框初始化 |
| `infer(views, command)` | 对一组局部视图推理，返回与视图顺序一致的 `LocalObservation`；`command` 指定模板保持或更新。运行时每帧只传一个视图 |
| `close()` | 释放显存和其他资源 |

接入步骤：

1. 在 `backends/` 下新建会话类和适配器，第三方代码放在 `third_party/<name>/` 并附上原许可证；
2. 在 `runtime/driver.py::buildRuntime()` 中按 `model.variant` 创建对应后端；
3. 在 `core/config.py` 中放开新的 `model.variant` 取值；
4. 用 `tests/unit/test_artrack_backend.py` 中的假会话方式补充单元测试。
