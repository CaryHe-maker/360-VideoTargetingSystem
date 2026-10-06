# Controller：视图规划、运动与状态机

Controller 决定“看哪里、是否接受这一帧的框、下一帧处于什么状态”。每帧只规划**一个**透视视图，后端在这个视图里给出一个框，控制器据此提交一次结果。

| 文件 | 职责 |
|---|---|
| `track_controller.py` | 状态所有权、逐帧的“规划 → 提交”协议 |
| `state_model.py` | 运动预测、单帧观测、状态转移的数据结构 |
| `view_planner.py` | 规划搜索视图：以预测方向为中心，按目标角尺寸确定视场 |
| `state_evaluator.py` | 判断这一帧的框是否作为测量被接受 |
| `motion_estimator.py` | 球面多帧运动预测 |
| `fused_score.py` | 外观分校准、视图运动先验、SingleScore 合成 |
| `score_calibration.py` | 可选的、与权重绑定的分数校准产物加载 |
| `state_machine.py` | 跨帧的纯状态转移 |
| `template_policy.py` | 模板策略：固定第 0 帧 anchor，可选 recent / stable 模板更新 |

## 视图规划

`ViewPlanner.searchView(center, width, height)` 返回一个视图：

- 中心是运动模型预测的目标方向；
- `width` / `height` 是预测的目标角尺寸（运动模型还没有给出尺寸时，用上一次提交的 BFoV）。视图的水平、垂直视场分别是它们的 3 倍，限制在 `geometry.minFovDeg` 到 `geometry.maxFovDeg` 之间，再受 `backendTuning.viewHorizontalFovCapDeg` / `viewVerticalFovCapDeg`（默认 90°）限制；
- 输出尺寸为 `geometry.viewWidthPx × viewHeightPx`（256×256）。水平和垂直视场各自按目标尺寸确定，所以视图在两个方向上的角分辨率一般不相等。

`TRACKING` 和 `UNCERTAIN` 两种状态使用相同的规划，没有“丢失后全局搜索”的路径。

## 测量是否被接受

运行时把局部框回投到球面并计算 `singleScore`，再交给 `StateEvaluator`：

- 默认 `backendTuning.acceptAnyCandidate: true`：只要这一帧有框，就作为测量接受。ARTrackV2 的分数集中在 0.5 附近，不是校准过的概率，不适合直接做门槛；
- `acceptAnyCandidate: false`：分数低于 `tracking.candidateMinScore` 的框不被接受；
- 这一帧没有框（回投失败）时，输出运动预测的范围，`valid=False`。

只有被接受的测量才会更新当前的 bbox / BFoV、运动样本和模板。测量未被接受、且目标面积不小于画面的 10% 时，`holdWeakBox` 让输出保持上一帧的框。

## 球面运动预测

`SphericalMotionEstimator` 输出下一帧的搜索中心、目标角尺寸和不确定度。

- **样本**：只把可靠测量放入长度为 `windowLength` 的窗口，预测帧不进入；
- **角速度**：在最新点建立 east / north 切平面，把历史单位向量投影到二维，每个轴拟合“截距 + 时间 × 速度”，再做三轮 Huber 重加权以压制离群点。速度大小受 `maxAngularSpeedRadPerSec` 限制；残差过大时速度退化为 0。整个过程不直接对 yaw 做差，所以在 ±180° 经线处不会跳变；
- **尺度**：水平 / 垂直角尺寸在 log 空间分别线性拟合，变化率受 `maxLogScaleRatePerSec` 限制；
- **不确定度**：由拟合残差、过程噪声和样本数共同决定，输出 2×2 中心协方差和 2×2 log 尺度协方差；
- **重新捕获**：`resetFromMeasurement()` 清空旧窗口，避免丢失前的速度把下一帧推离目标。

## 打分

- **外观分**：后端分数。提供校准产物时经过与权重绑定的单调 Beta 校准，否则直接使用原始分数；
- **视图运动先验**：视图中心与预测中心的大圆夹角越大，分数越低。视图本身以预测中心为中心，所以这一项在当前规划下恒为最高值；
- **SingleScore**：`useMotionScore: true` 且有校准产物时是外观分与运动分的加权，否则就是外观分。

## 状态机

每个状态机维护一个容量为 10 的 `ScoreGroup`，保存最近若干帧的最终分数（StateScore）。阈值不是固定常数，而是由历史分数自适应得到：

- 第 1、2 个跟踪状态：无条件保持 `TRACKING`；
- 第 3 到第 10 个：`UT = 0.5 × max + 0.5 × min`，`LT = 0.2 × max + 0.8 × min`；
- 第 11 个起：把 ScoreGroup 降序排列，`UT` 取第 5 大的值，`LT` 取第 8 大的值。

```text
StateScore ≥ UT        → TRACKING
LT ≤ StateScore < UT   → UNCERTAIN
StateScore < LT        → UNCERTAIN（记录 HARD_MISS）
```

状态只在 `TRACKING` 和 `UNCERTAIN` 之间转移。`TrackStatus.LOST` 仍是对外类型的一部分，但控制器不会进入这个状态：目标丢失后的重新检测还没有实现，见 [V2Plan](../V2Plan.md)。

状态目前只影响模板更新的节奏（连续稳定帧数），不影响视图规划。

## 模板策略

模板固定使用第 0 帧初始化时的 anchor。`onlineTemplate: true`（默认）时，分数不低于 `templateMinConfidence` 的观测可以刷新 recent 模板（大约每两帧一次），连续稳定 `stableFramesBeforeUpdate` 帧后刷新 stable 模板；anchor 始终保留，防止目标漂移后模板被完全污染。

## 逐帧协议

1. `beginFrame(frame)` 返回 `SearchPlan`（一个视图 + 模板命令 + 运动预测），并记住这份计划；
2. 运行时裁剪视图、推理、回投，得到一个 `ProjectedObservation`（或 `None`）；
3. `consume(plan, observation)` 校验计划就是刚才那一份、观测属于计划中的视图，然后提交并返回 `TrackResult`。

同一份计划不能被消费两次，上一帧的计划也不能用于当前帧，否则抛出 `ProtocolError`。规划或推理出错时运行时调用 `commitFallback()`，保留上一个可信状态并输出一条 `valid=False` 的结果。
