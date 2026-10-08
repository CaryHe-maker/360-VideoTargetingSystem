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

每帧的视图就是 ARTrackV2 训练时见到的那种搜索区域（`backendTuning.alignedSearch: true`，默认）。`ViewPlanner.searchView(center, width, height)` 返回的视图：

- 中心是运动模型预测的目标方向；
- 是正方形，两个方向的角分辨率相同，目标保持原来的长宽比；
- 边长是目标平均尺寸 `sqrt(宽 × 高)` 的 4 倍（ARTrackV2 训练时的搜索倍数），在成像平面上计算。`width` / `height` 是预测的目标角尺寸，运动模型还没有给出尺寸时用上一次提交的 BFoV；
- 视场下限是 `alignedMinFovDeg`（2°），上限是 `viewHorizontalFovCapDeg` / `viewVerticalFovCapDeg`（默认 90°）和 `geometry.maxFovDeg`；
- 带一个 `priorBox`：目标在视图里预计的位置和大小。后端以它为中心裁 4 倍的搜索区域。视场没有被上限截住时，这个裁剪正好是整个视图；视场被截住时（搜索区域在 90° 到球面视图的切换点之间），裁剪范围超出视图，超出部分补黑边，目标在搜索区域里仍然占 1/4 左右；
- 带一个 `trajectory`：目标在前 7 帧的框，换算成这一帧视图里的像素坐标，最旧的在前。控制器保存最近 7 帧提交的 BFoV（初始化时是 7 份初始目标），每帧把它们投影到新视图里：中心按透视投影计算，大小按“位于视图中心时”的大小计算。落在视图外甚至视图背面的历史框也照样给出，由后端限制到它的坐标范围。序列级后端把它作为轨迹提示；
- 输出尺寸为 `geometry.viewWidthPx × viewHeightPx`（256×256）。

模板视图（`ViewPlanner.templateView()`）同样是正方形，边长是目标平均尺寸的 `templateFovScale` 倍。

### 大目标：球面视图

透视视图表示不了 180° 以上的范围，超过 90° 边缘就拉伸得很厉害。搜索区域（目标平均角尺寸的 4 倍）达到 `sphericalSearchFovDeg`（默认 120°，即目标约 30°）时，视图换成**球面视图**（`ViewSpec.projection = SPHERICAL`，`backendTuning.sphericalSearch: true`，默认）：

- 像素在经度和纬度上均匀分布，经纬度是把球转到“视图中心位于 (0, 0)”之后的。相当于一块以目标为中心的 ERP，目标落在形变最小的位置。做法与 360VOT 的扩展 BFoV 相同，见 [360VOT 基准框架](../benchmarkFramework.md#22-从-erp-上取一块局部搜索区域)；
- 每个方向跨目标平均尺寸的 4 倍，尺度定为“这 4 倍正好是 256 像素”，两个方向的角分辨率相同；
- 横向最多 360°，纵向最多 180°，超出就截断，所以视图不一定是正方形（目标 180° 时是整个球面，约 128×64）。截掉的部分由后端裁搜索图时补黑边；
- 细长的目标按“平均尺寸的 4 倍”取景可能装不下，这时那个方向至少取目标自身的 1.25 倍；
- `priorBox` 和 `trajectory` 的大小按角度线性换算。目标偏离视图的“赤道”时，同样的宽度占更多的经度，按纬度的余弦修正。

模板视图用同样的条件切换。球面视图里得到的 BFoV 是“中心方向 + 经纬度跨度”，可以超过 180°；透视视图里得到的是“中心方向 + 相机视场”。两者在小角度下一致。

`sphericalSearch: false` 是之前的行为：透视视场封顶在 `viewHorizontalFovCapDeg` / `viewVerticalFovCapDeg`，超出部分补黑边。

`TRACKING` 和 `UNCERTAIN` 两种状态使用相同的规划，没有“丢失后全局搜索”的路径。这种取法不支持 `geometry.resampler: cuda`，同时配置会直接报错。

### 旧的取法

`alignedSearch: false` 保留了 2026-10-07 之前的默认行为，只用于对照：

- 视图的水平、垂直视场分别是目标宽、高的 3 倍，限制在 `geometry.minFovDeg`（20°）到上限之间。两个方向各自确定，所以目标会被拉成接近正方形；
- 后端在视图里再按目标的 4 倍裁一次搜索区域（`fullViewSearch: true` 时不裁，整个视图就是搜索区域）；
- 模板视图的两个方向各取目标的 `templateFovScale` 倍。

换成现在的取法后，66 条可用训练序列上的 S<sub>dual</sub> 从 0.278 升到 0.473（[评测记录](../evaluation-log.md) E005、E007）。

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

## 丢失处理

`backendTuning.lossHandling: true` 时启用（默认关闭，实验结果见 [评测记录](../evaluation-log.md) E017 起）。它在正常的逐帧跟踪之外加了三件事：

**一、判断这一帧可不可疑。** 两个信号，任何一个不过关就算可疑：

- 跟踪器的分数低于 `suspectScore`（0.50）：模型自己觉得框得不准；
- 框里的图像和第 0 帧模板的外观相似度低于 `suspectSimilarity`（0.30）：框得再准，也不像原来的目标。相似度由一个冻结的小模型（`verifierModel`，默认 DINOv2 ViT-S/14）计算，见 [Backends](backends.md#外观验证器)。

两个信号是互补的：分数完全看不出“跟到了别的物体上”，而相似度能抓到其中约四成（E016）。

**二、可疑的帧不让跟踪器学习。** 跟踪照常进行，框照常输出，位置和轨迹也照常更新；但这一帧对跟踪器外观记忆的影响会被撤销（恢复到这一帧之前的状态）。这样偶尔一帧误判几乎没有代价，而真的跟错时外观记忆不会被错误目标逐帧改写。

**三、连续可疑之后在别处找。** 连续 `lostAfterFrames`（4）帧可疑后，每帧除了正常的那个视图，再取 `scanViewsPerFrame`（4）个扫描视图：

- 扫描视图和正常的搜索视图一样大（目标占四分之一边长），相邻视图错开半个视图，铺满整个球面；
- 顺序是从“最后一次可信的位置”由近到远，每帧取接下来的几个，扫完一圈从头再来；
- 每个扫描视图用**无状态**的方式前向一次（只看第 0 帧模板，不带轨迹和外观记忆），得到一个候选框，再算它和模板的相似度；
- 候选同时满足三条才会被采纳：相似度不低于 `reacquireSimilarity`（0.45）；比当前正在跟的框高出至少 `reacquireMargin`（0.15）；无状态前向的分数不低于 `reacquireScore`（0.70）。错误的候选分数普遍很低（中位数 0.07），这一条是主要的把关；
- 采纳后跟踪从候选的位置重新开始：轨迹重置为 7 份候选框，运动模型重置，外观记忆清空。

正在跟的框一旦不再可疑，可疑计数清零，扫描停止，“最后一次可信的位置”更新为当前位置。

和 360VOT 公开框架的三段式（等待 → 逐帧扩大 → 整帧）相比：等待的帧数相同；扩大搜索的方式不同，这里不放大视图（那会让目标在搜索图里变小，偏离后端的训练条件），而是用同样大小的视图由近到远扫描；丢失的判据不同，它用的是分割模型的空掩码，这里是分数加外观相似度。

每条序列的可疑帧数、扫描帧数和发生跳转的帧号记在运行报告里（`suspectFrames`、`scanFrames`、`reacquiredAt`）。

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

默认的序列级模型自己每帧更新外观特征，框架不做模板更新，本节只适用于 `sequenceModel: false`。

模板固定使用第 0 帧初始化时的 anchor。`onlineTemplate: true` 时，分数不低于 `templateMinConfidence` 的观测可以刷新 recent 模板（大约每两帧一次），连续稳定 `stableFramesBeforeUpdate` 帧后刷新 stable 模板；anchor 始终保留，防止目标漂移后模板被完全污染。

## 逐帧协议

1. `beginFrame(frame)` 返回 `SearchPlan`（一个视图 + 模板命令 + 运动预测），并记住这份计划；
2. 运行时裁剪视图、推理、回投，得到一个 `ProjectedObservation`（或 `None`）；
3. `consume(plan, observation)` 校验计划就是刚才那一份、观测属于计划中的视图，然后提交并返回 `TrackResult`。

同一份计划不能被消费两次，上一帧的计划也不能用于当前帧，否则抛出 `ProtocolError`。规划或推理出错时运行时调用 `commitFallback()`，保留上一个可信状态并输出一条 `valid=False` 的结果。
