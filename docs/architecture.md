# 系统架构

## 问题与思路

ERP（等距柱状投影）全景帧有三个特点，导致普通单目标跟踪器无法直接使用：

1. **形变**：纬度越高，水平方向拉伸越严重，目标在两极附近严重变形；
2. **跨缝**：经度 ±180° 处是图像左右边界，目标可能一半在左边、一半在右边；
3. **分辨率**：整帧通常是 3840×1920 甚至更大，而跟踪器的输入只有 256×256。

本项目不修改跟踪器，而是在跟踪器外面加一层“球面控制”，做法与 360VOT 论文的 360 跟踪框架一致：

- 用球面上的 **BFoV**（中心经纬度 + 水平 / 垂直视场角）表示目标；
- 每帧围绕预测方向生成**一个局部透视视图**，在这个视图里目标接近普通图像中的样子，也不存在跨缝；
- 跟踪器在局部视图中预测框，再**回投到球面**，得到 BFoV 和跨缝的 ERP 框。

## 数据流

```text
视频 / 图像序列
   │  预取线程（_PrefetchReader）按顺序解码
   ▼
FramePacket（ERP RGB）
   │
   ▼
TrackController.beginFrame() ──▶ SearchPlan（1 个视图）
   │
   ▼
Geometry.cropViews()            ERP → 256×256 透视视图（CPU 或 CUDA）
   │
   ▼
TrackerBackend.infer()          ARTrackV2-B-256，每帧一次前向
   │
   ▼
Geometry 回投 + 打分             局部框 → BFoV / 跨缝 ERP 框；外观分（可选运动分）→ SingleScore
   │
   ▼
TrackController.consume()       判断是否接受测量，状态机转移，提交 TrackResult
   │
   ▼
ResultSink                      每帧一行结果
```

## 包结构

```text
src/track360/
├── cli.py              统一命令行入口（track / airsim360 / list-instances）
├── core/               数据类型、协议、配置 schema、错误类型
├── geometry/           球面数学、BFoV 投影、跨缝区间、CUDA 重采样
├── controller/         视图规划、测量判定、运动预测、状态机、丢失处理
├── backends/           ARTrackV2 会话、推理适配、模板缓存
├── runtime/            组件装配（buildRuntime）与逐帧循环（runTracking）
├── datasets/           360VOT / AirSim360 / 视频 / 图像序列读取，伪真值生成
├── io/                 图像与视频读取、结果写入
├── evaluation/         平面、循环、球面指标与性能统计
├── visualization/      中间视图与结果图
└── third_party/        上游 ARTrackV2 模型代码（推理子集）、360VOT toolkit 的指标代码
```

### 依赖规则

```text
cli ─▶ runtime ─┬─▶ controller ─▶ geometry ─▶ core
                ├─▶ backends ─▶ third_party
                └─▶ io / datasets / visualization / evaluation
```

- `core` 不依赖任何其他包；所有跨模块的数据都使用 `core/types.py` 中的类型。
- `controller` 和 `geometry` 只通过 `core/protocols.py` 中的协议使用后端，不依赖具体模型类型，因此更换后端不影响控制逻辑。
- `third_party/` 只能被对应的模块导入：`backends/` 导入 ARTrackV2 模型代码，`evaluation/` 导入 360VOT toolkit 的指标代码。
- `runtime/driver.py` 是唯一的组合根：在这里创建并连接所有组件。

## 逐帧协议

每帧一次规划、一次推理、一次提交：

1. `beginFrame(frame)` 返回 `SearchPlan`：一个视图、模板命令和运动预测；
2. 运行时裁剪视图、推理、回投，得到一个 `ProjectedObservation`；回投失败时为 `None`；
3. `consume(plan, observation)` 校验这份计划就是控制器正在等待的那一份，然后更新状态并返回 `TrackResult`。

控制器在两步之间只记住“待提交的计划”，状态在 `consume()` 里一次性写入。这样可以防止：上一帧迟到的结果污染当前帧、同一份计划被消费两次、预测框被误当作真实测量写入运动历史。

协议类型位于 `core/protocols.py` 与 `core/types.py`，实现位于 `controller/track_controller.py`，详见 [Controller](modules/controller.md#逐帧协议)。

## 失败处理

- 单帧的规划或推理出错时，控制器调用 `commitFallback()`：保留上一个可信状态，输出一条 `valid=False` 的结果并推进 revision，保证每个输入帧都有一行输出；
- 命令行入口把异常映射为固定的退出码（见 [快速上手](getting-started.md#退出码)）。

## 历史：多视图方案

V1 和 V2 早期的控制器每帧生成 4 个“四角视图”（可选第二轮再 4 个，丢失时还有 cubemap），再把多个视图的框融合成一个结果。在 tune 集上它的 S<sub>dual</sub> 只有 0.065，远低于单视图的 0.271，延迟是单视图的 3.5 倍（[评测记录](evaluation-log.md) E001），因此整套多视图、两轮搜索和融合逻辑已经从代码中删除。需要查看旧实现时，见提交 `acd253d` 及更早的历史。
