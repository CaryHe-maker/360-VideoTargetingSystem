# Controller：视图规划、融合、运动与状态机

Controller 决定“看哪里、相信哪个候选、是否继续查询、下一帧处于什么状态”。

| 文件 | 职责 |
|---|---|
| `track_controller.py` | 状态所有权、帧事务、一次性提交 |
| `state_model.py` | 状态、证据、候选、事务的数据结构 |
| `recovery_planner.py` | 按状态规划局部视图（四角视图、cubemap） |
| `state_evaluator.py` | 每轮候选评估、调用 Fusor、判断测量是否可接受 |
| `fusor.py` | 跨缝的两框融合与最佳候选选择 |
| `motion_estimator.py` | 球面多帧运动预测 |
| `fused_score.py` | 外观分校准、视图运动先验、SingleScore 合成 |
| `score_calibration.py` | 可选的、与权重绑定的分数校准产物加载 |
| `state_machine.py` | 跨帧的纯状态转移 |
| `template_policy.py` | 模板策略：固定第 0 帧 anchor，可选 recent 模板更新 |
| `decision_gate.py` | 旧的聚合打分，仅为兼容保留，生产路径不使用 |

> 下文描述的是 YAML 配置决定的基础行为。ARTrackV2 后端会通过环境变量调整其中一部分（例如关闭运动分、放宽融合门槛、启用 recent 模板），详见 [配置说明](../configuration.md#环境变量待迁移)。

## 视图规划

### 四角视图（VStype1）

`ViewSpecType1(center, width, height)` 返回 4 个视图，顺序固定为左上、右上、左下、右下：

- `width` / `height` 是上一帧预测的目标角尺寸，每个视图的视场为 `3 × width` 和 `3 × height`，并限制在 30° 到 120° 之间（下限避免小目标视图过窄，上限避免透视相机看到背面）；
- 各视图中心在局部相机坐标系中偏移最终视场的 1/3，相邻视图的重叠比例保持固定；
- 中心通过 forward / right / up 基向量计算，靠近极点时仍保持“四角”的语义。

### 旋转 cubemap（VStype2）

cubemap 的 front 面指向预测中心，其余五个面由同一个局部正交基生成，每个面 120°。所以 cubemap 会随预测中心旋转，而不是固定在世界坐标系上。

### 按状态路由

| 状态 | 第 1 轮 | 第 2 轮 |
|---|---|---|
| `TRACKING` | 预测中心周围的动态四角视图（4 张） | 以第 1 轮最佳候选为中心的四角视图（4 张） |
| `UNCERTAIN` | 同上，但每张固定 120°×120° | 同上 |
| `LOST`（保留组件） | 6 张 cubemap + 4 张四角视图，单轮完成 | — |

第 1 轮没有候选时，第 2 轮以运动预测中心为中心。单帧视图预算上限为 12 张，不足时抛出 `ProtocolError`，不会生成不完整的布局。

## 候选评估与融合

运行时先把每个局部框回投并计算 `singleScore`，再交给 `StateEvaluator`。`Fusor` 的规则：

1. 每个观测先作为单框候选；
2. 枚举所有观测对，只有两个来源的分数都不低于 `fusionSourceMinConfidence`，且 `OverlapRate = 交集 / 较小框面积 ≥ 0.70` 时，才生成融合候选；
3. 融合分数使用几何平均加一致性奖励，并设置上限：

```text
agreementIoU = ERP 交集 / ERP 并集
base         = sqrt(a * b)
bonus        = 0.15 * agreementIoU * (1 - |a - b|) * (1 - base)
fusionScore  = min(base + bonus, max(a, b) + 0.03, 0.99)
```

4. 所有单框和融合候选统一排序，只返回一个最佳结果（同分时优先融合候选，再选 viewId 较小者）。

默认的 `fusionBoxMode: best_source` 下，融合候选胜出时，最终框直接取两个来源中分数较高的那个。这样可以避免不同视图的框直接求交 / 求并造成尺度失真。`reference_adaptive` 模式会按上一个可信框的面积在交集框和并集框之间裁剪，作为可选实验保留。

## 球面运动预测

`SphericalMotionEstimator` 输出下一帧的搜索中心、目标角尺寸和不确定度。

- **样本**：只把可靠测量放入长度为 `windowLength` 的窗口，预测帧和弱候选不进入；
- **角速度**：在最新点建立 east / north 切平面，把历史单位向量投影到二维，每个轴拟合“截距 + 时间 × 速度”，再做三轮 Huber 重加权以压制离群点。速度大小受 `maxAngularSpeedRadPerSec` 限制；残差过大时速度退化为 0。整个过程不直接对 yaw 做差，所以在 ±180° 经线处不会跳变；
- **尺度**：水平 / 垂直角尺寸在 log 空间分别线性拟合，变化率受 `maxLogScaleRatePerSec` 限制；
- **不确定度**：由拟合残差、过程噪声和样本数共同决定，输出 2×2 中心协方差和 2×2 log 尺度协方差；
- **重新捕获**：`resetFromMeasurement()` 清空旧窗口，避免丢失前的速度把下一帧推离目标。

## 打分

- **外观分**：后端分数。提供校准产物时经过与权重绑定的单调 Beta 校准，否则直接使用原始分数；
- **视图运动先验**：局部视图中心与预测中心的大圆夹角越大，分数越低（0° 为 1.0，每 30° 下降 0.1）。同一视图内的候选共享该分数；
- **SingleScore**：外观分与运动分各占 50%。

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

当前正常路径只在 `TRACKING` 和 `UNCERTAIN` 之间转移。`LOST` 状态、cubemap 规划和找回逻辑作为组件保留并有测试，但不会被自动触发。重新启用并评估找回路径是 [V2Plan](../V2Plan.md) 中的待办项。

状态转移只看 StateScore；是否把当前结果写入测量历史（`acceptMeasurement`）由评估器单独判断。只有被接受的测量才会更新 bbox / BFoV、运动样本和模板。

## 模板策略

模板固定使用第 0 帧初始化时的 anchor。启用 recent 模板（ARTrackV2 后端默认启用）时，只有高分且已确认的观测才能刷新 recent 模板，anchor 始终保留，防止目标漂移后模板被污染。同一帧的两轮推理使用同一个模板快照，第 2 轮强制保持（`KEEP`）。
