# Track360 文档

| 文档 | 内容 |
|---|---|
| [快速上手](getting-started.md) | 安装、模型权重、命令行用法、输出格式 |
| [系统架构](architecture.md) | 数据流、包结构、逐帧协议、依赖规则 |
| [配置说明](configuration.md) | `configs/default.yaml` 各配置段的含义和约束 |
| [Benchmark 数据集](benchmark.md) | 测试集选型、评测指标、对照基线 |
| [V2 计划](V2Plan.md) | 到 V2.0 发布的分阶段路线图与按模块的优化建议 |
| [360VOT 基准框架](benchmarkFramework.md) | 论文里的 360 跟踪框架做了什么、有哪些参数、各贡献多少，以及和本项目的逐项对照 |
| [优化方向与可行性](optimization-options.md) | 当前可选的优化方向：依据、实现方式、工作量和风险 |
| [评测记录](evaluation-log.md) | 每一轮评测结果与 benchmark 参考数据的长期记录 |
| [冻结的两条基线](baselines.md) | 丢失找回关闭和开启两条对照基线的配置、数字和用法 |
| [历史实验结论](experiments.md) | V1（比赛阶段）的 A/B 实验结果与经验 |

## 模块文档

| 模块 | 内容 |
|---|---|
| [Core](modules/core.md) | 数据类型、协议、错误类型 |
| [Geometry](modules/geometry.md) | 球面 / ERP / 透视投影、BFoV、跨缝处理、GPU 重采样 |
| [Controller](modules/controller.md) | 视图规划、测量判定、运动预测、状态机 |
| [Backends](modules/backends.md) | ARTrackV2 推理运行时与后端接入方式 |
| [Runtime](modules/runtime.md) | 运行时装配、逐帧循环、预取、计时 |
| [Datasets](modules/datasets.md) | 360VOT / AirSim360 读取、图像解码、标注的坐标约定 |
| [Evaluation](modules/evaluation.md) | 平面 / 循环 / 球面指标与性能统计 |
| [Visualization](modules/visualization.md) | 中间视图与结果图的诊断产物 |

## 文档约定

- 所有文档使用中文；代码、注释和标识符使用英文。
- 文档只描述当前代码的真实行为；计划中的能力写在 [V2Plan.md](V2Plan.md)。
- 文档中不出现本地私有路径；数据集一律通过命令行参数指定。
