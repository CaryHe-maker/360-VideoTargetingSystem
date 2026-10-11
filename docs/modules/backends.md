# Backends：跟踪器后端

当前只有一个后端：官方 **ARTrackV2-B-256**（[miv-xjtu/ARTrack](https://github.com/miv-xjtu/artrack)，CVPR 2024）。

| 文件 | 职责 |
|---|---|
| `backends/artrack_model.py` | 模板和预测的数据类型、裁剪与坐标的公共函数；`ARTrackBackend`：批量输入输出校验 |
| `backends/artrack_backend.py` | `TrackerBackendImpl`：实现 `core.protocols.TrackerBackend`，把模板 revision、局部框和控制器的模板命令串起来 |
| `backends/template_cache.py` | 模板缓存：anchor 模板与 recent 模板、revision 管理 |
| `backends/observation.py` | 把模型预测转换为 `LocalObservation` |
| `third_party/artrackv2/` | 上游模型代码的推理子集：序列级模型（`lib/models/artrackv2_seq`）、外观解码器、配置 |

## 这份权重是序列级模型

官方发布的 ARTrackV2-B-256 权重（训练 40 个 epoch 的那一份）是**序列级**训练的结果。它比“模板 + 搜索图”的帧级模型多两样东西：

- **轨迹提示**：前 7 帧的目标框，换算成当前搜索图里的坐标，作为输入序列的一部分；
- **外观特征**：第二个模板位不是一张图，而是模型自己每帧重写的一组特征（由一个 8 层的解码器生成）。

2026-10-07 之前，本项目用帧级的模型代码加载这份权重：权重里的 106 组参数（轨迹位置嵌入、外观解码器）因为模型里没有对应的位置被悄悄丢掉，推理时也没有喂轨迹。模型在它没有被训练过的输入形式下运行，分数头的输出因此和 IoU 无关（[评测记录](../evaluation-log.md) E008）。现在只用序列级的模型代码，权重严格加载，一个参数都不多、不少；帧级的模型代码已经删除。

| 文件 | 职责 |
|---|---|
| `backends/artrack_seq_session.py` | `PyTorchARTrackV2SeqSession`：序列级推理；`createArtrackSession()` 按配置选择会话 |

## 推理流程（序列级）

做法与上游的 `lib/test/tracker/artrackv2_seq.py` 一致：

1. **初始化**：在第 0 帧的模板视图上，以目标框为中心按 2 倍裁出 128×128 的模板；
2. **搜索图**：以视图给出的 `priorBox`（目标预计的位置和大小）为中心按 4 倍裁出 256×256 的搜索图；
3. **轨迹**：视图带着前 7 帧的目标框（`ViewSpec.trajectory`，已经换算成这一帧视图里的像素坐标）。每个框再换算成搜索图里的 `x0, y0, x1, y1`，除以搜索图边长，限制在 [−0.5, 1.5]，映射到 800 个坐标格；
4. **前向**：模板、外观特征、搜索图和轨迹一起送入网络；
5. **解码**：四个坐标各输出一个 800 格上的分布。取概率最大的格子和整个分布的期望，两者平均，再映射回视图的像素坐标；
6. **外观特征**：网络输出的新外观特征存起来给下一帧用。它挂在目标的第 0 帧模板对象上，所以换一个目标（新的模板）就从头开始；
7. **分数**：分数头的原始输出，限制到 [0, 1]。它在训练时直接回归“预测框与真值的 IoU”，不需要 sigmoid；
8. **回投**：局部框交给 Geometry 回投到球面，后续由 Controller 处理。

轨迹由 Controller 提供，因为只有它知道之前的框在球面上的位置：它保存最近 7 帧提交的 BFoV，规划视图时把它们投影到新视图里（见 [Controller](controller.md#视图规划)）。直接在 ERP 上跟踪的基线 `b0` 坐标系不变，直接用前 7 帧的 ERP 框。

序列级模型自己管理外观，不接受框架的模板更新；框架只在第 0 帧编码一次模板。

## 权重加载

- 默认路径为 `models/artrackv2_b_256.pth.tar`，加载 `net` 状态字典。序列级模型用严格模式加载，缺少参数或有多余参数都会报错；
- 首先尝试 `torch.load(weights_only=True)`。官方压缩包中带有旧版训练统计对象，安全加载失败时才回退到兼容模式（只对显式指定的本地文件这样做）；
- 有 CUDA 时使用 GPU，否则使用 CPU。

## 外观验证器

跟踪器的分数回答的是“这个框框得准不准”，跟到别的物体上之后它照样很高。`backends/appearance.py` 里的 `AppearanceVerifier` 回答另一个问题：框里的东西和第 0 帧的模板是不是同一个。

- 做法：把模板和当前框各裁成“以框为中心、边长为长边 1.1 倍”的正方形（超出图像的部分补黑），缩放到 224，用一个冻结的图像模型提特征，取余弦相似度。模板的特征只在第 0 帧算一次，之后不更新：它是唯一的参照。跟着跟踪结果更新的参照会在跟错之后被错的物体带偏（[评测记录](../evaluation-log.md) E019），已经移除；
- 模型：默认 DINOv2 ViT-S/14（约 2200 万参数，一次前向约 5 ms）。也可以选 DINO ViT-S/16 或 ResNet-18，效果都明显不如它（[评测记录](../evaluation-log.md) E016）；
- 权重放在 `models/hub/`，通过 `torch.hub` 加载，不随仓库分发。第一次使用时需要联网下载（DINOv2 约 85 MB）；
- 只在 `backendTuning.lossHandling: true` 时创建，由 Runtime 在后端推理之后调用，结果写进 `ProjectedObservation.appearanceSimilarity`。

为了配合丢失处理，后端门面还提供三个操作：

| 方法 | 作用 |
|---|---|
| `saveState()` / `restoreState(state)` | 保存、恢复跟踪器的外观记忆，用来撤销一帧的影响 |
| `resetState()` | 清空外观记忆，跟踪器从模板重新开始 |
| `inferDetached(views)` | 无状态推理：只看第 0 帧模板，不带轨迹，也不读写外观记忆。用来在扫描视图里找候选 |

## 已知限制

- 推理始终是 FP32，`model.precision` 只接受 `fp32`。FP16 / TensorRT 是 [V2Plan](../V2Plan.md) Phase 5 的工作；
- 后端由 `createArtrackSession()` 按配置创建，还没有注册表机制；

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
