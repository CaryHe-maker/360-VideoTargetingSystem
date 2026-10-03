# 系统架构

## 问题与思路

ERP（等距柱状投影）全景帧有三个特点，导致普通单目标跟踪器无法直接使用：

1. **形变**：纬度越高，水平方向拉伸越严重，目标在两极附近严重变形；
2. **跨缝**：经度 ±180° 处是图像左右边界，目标可能一半在左边、一半在右边；
3. **分辨率**：整帧通常是 3840×1920 甚至更大，而跟踪器的输入只有 256×256。

本项目不修改跟踪器，而是在跟踪器外面加一层“球面控制”：

- 用球面上的 **BFoV**（中心经纬度 + 水平 / 垂直视场角）表示目标；
- 每帧围绕预测方向生成若干个**局部透视视图**，在这些视图里目标接近普通图像中的样子；
- 跟踪器在局部视图中预测框，再**回投到球面**，统一比较、融合和提交。

## 数据流

```text
视频 / 图像序列
   │  预取线程（_PrefetchReader）按顺序解码
   ▼
FramePacket（ERP RGB）
   │
   ▼
TrackController.beginFrame() ──▶ SearchPlan（4 个视图）
   │
   ▼
Geometry.cropViews()            ERP → N 个 256×256 透视视图（CPU 或 CUDA）
   │
   ▼
TrackerBackend.infer()          ARTrackV2-B-256，一次 batch 处理 N 个视图
   │
   ▼
Geometry 回投 + 打分             局部框 → BFoV / 跨缝 ERP 框；外观分 + 运动分 → SingleScore
   │
   ▼
TrackController.consume()       候选融合，状态机转移，一次性提交（FrameCommitted）
   │                            带门槛的模式下可以先返回第 2 轮 SearchPlan（MoreViewsRequired）
   ▼
ResultSink                      每帧一行结果
```

## 包结构

```text
src/track360/
├── cli.py              统一命令行入口（track / airsim360 / list-instances）
├── core/               数据类型、协议、配置 schema、错误类型
├── geometry/           球面数学、BFoV 投影、跨缝区间、CUDA 重采样
├── controller/         视图规划、候选评估与融合、运动预测、状态机、帧事务
├── backends/           ARTrackV2 会话、批量推理适配、模板缓存
├── runtime/            组件装配（buildRuntime）与逐帧循环（runTracking）
├── datasets/           视频 / 图像序列 / AirSim360 读取，伪真值生成
├── io/                 图像与视频读取、结果写入
├── evaluation/         平面、循环、球面指标与性能统计
├── visualization/      中间视图与结果图
└── third_party/        上游 ARTrackV2 模型代码（推理子集）
```

### 依赖规则

```text
cli ─▶ runtime ─┬─▶ controller ─▶ geometry ─▶ core
                ├─▶ backends ─▶ third_party
                └─▶ io / datasets / visualization / evaluation
```

- `core` 不依赖任何其他包；所有跨模块的数据都使用 `core/types.py` 中的类型。
- `controller` 和 `geometry` 只通过 `core/protocols.py` 中的协议使用后端，不依赖具体模型类型，因此更换后端不影响控制逻辑。
- 只有 `backends/` 可以导入 `third_party/`。
- `runtime/driver.py` 是唯一的组合根：在这里创建并连接所有组件。

## 帧事务协议

同一帧最多进行两轮查询。**默认配置（`backendTuning.acceptAnyCandidate: true`）每帧只做一轮**，第二轮只在带门槛的模式下出现（见 [配置说明](configuration.md#backendtuning)）。协议按两轮设计：如果每一轮都立即修改状态，第二轮就会在“半更新”的状态上运行，出错时也无法回滚。因此控制器把一帧内的所有尝试暂存在 `FrameTransaction` 中，到提交点一次性写入。

1. `beginFrame(frame)` 创建事务，返回 `attemptIndex=0` 的 `SearchPlan`；
2. 运行时必须按计划中的视图顺序返回 `ProjectedObservation`；
3. `consume(plan, observations)` 校验序列 ID、帧号、事务 ID、轮次、状态 revision 和模板 revision；
4. 需要第 2 轮时，第 1 轮结束后返回 `MoreViewsRequired` 和第 2 轮计划，旧计划不能被再次消费；
5. 最后一轮结束后合并本帧所有候选，执行融合和状态转移，返回 `FrameCommitted`。

这一协议可以防止：上一帧迟到的结果污染当前帧、同一计划被消费两次、模板 revision 跳号、预测框被误当作真实测量写入运动历史。

协议类型位于 `core/protocols.py` 与 `core/types.py`，事务数据位于 `controller/state_model.py`，校验与提交位于 `controller/track_controller.py`。

## 失败处理

- 单帧的规划或推理出错时，控制器调用 `commitFallback()`：保留上一个可信状态，输出一条 `valid=False` 的结果并推进 revision，保证每个输入帧都有一行输出；
- 命令行入口把异常映射为固定的退出码（见 [快速上手](getting-started.md#退出码)）。
