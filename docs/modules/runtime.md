# Runtime：组件装配与逐帧循环

Runtime 是组合根和执行器，本身不实现状态判定或模型算法。

| 文件 | 职责 |
|---|---|
| `runtime/driver.py` | `buildRuntime()` 装配组件；`runTracking()` 逐帧多轮循环；`_PrefetchReader` 后台预取 |
| `runtime/track_video.py` | `track360 track`：视频 / 图像序列入口 |
| `runtime/track_airsim360.py` | `track360 airsim360`：AirSim360 入口，负责生命周期和计时产物 |
| `cli.py` | 统一命令行，分发到上面两个入口和 `list-instances` |

## 组件装配

`buildRuntime(config)` 按配置创建并返回 `RuntimeBundle`：

1. Geometry：默认 `SphericalGeometryImpl`（CPU），设置 `TRACK360_GPU_GEOMETRY=1` 时使用 `GpuGeometryImpl`；
2. 后端：`PyTorchARTrackV2Session` → `ARTrackBackend` → `TrackerBackendImpl`；
3. 控制器：`TrackControllerImpl(geometry, config)`；
4. 结果输出：`FileResultSink`；
5. 可选的中间视图记录器；
6. 分数校准：有 `scoring.calibrationArtifact` 时加载并校验，否则对 ARTrackV2 使用未校准的原始分数。

测试可以通过 `artrackSessionFactory` 和 `geometryFactory` 注入假实现，不需要真实权重和 GPU。

## 逐帧循环

```text
第 0 帧：buildInitialization → 裁剪模板视图 → backend.initialize → commitInitialization → 写结果
之后每帧：
  frame = 预取队列.read()
  plan  = controller.beginFrame(frame)
  循环：
    views        = geometry.cropViews(frame, plan.views)
    observations = backend.infer(views, plan.templateCommand)
    projected    = 回投 + 外观校准 + 视图运动先验 + SingleScore
    outcome      = controller.consume(plan, projected)
    MoreViewsRequired → plan = outcome.plan，继续
    FrameCommitted    → 写结果，结束本帧
```

- `_PrefetchReader` 在后台线程中解码，队列保持输入顺序；解码线程出错时，异常会在主线程读到对应位置时重新抛出；
- 每帧结束后释放 GPU 帧张量，避免显存随帧数增长；
- 规划或推理出错时调用 `commitFallback()`，保证每个输入帧都有一行输出。

## 计时与性能统计

| 工具 | 用途 |
|---|---|
| `visualization/time_counter.py::TimeCounter` | 正式产物 `time.json`：用 `perf_counter_ns()` 累计“跟踪处理”区间，不含可视化、结果写入、配置加载和清理 |
| `evaluation/profiler.py::RuntimeProfiler` | 开发诊断：按名称统计 crop、backend、controller 等代码段的次数、总耗时、最小 / 最大 / 平均值 |

测量 CUDA 耗时时，普通的 `perf_counter` 可能只记录了异步提交的时间。需要在测量边界显式同步，或使用 CUDA event，并在报告中注明同步方式。

## 配置中的 runtime 段

`runtime.*QueueCapacity` 是为将来的并行流水线预留的。当前循环是顺序执行的（只有解码在后台线程），这些值只会被校验，不影响运行和延迟。
