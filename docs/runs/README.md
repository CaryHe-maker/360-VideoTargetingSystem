# 评测运行记录

本表由 `tools/benchmark.py archive` 生成，每一行对应一次运行。每次运行还有一个同名的 JSON 记录（逐条序列的分数和耗时、生效的配置、commit 和环境），和结果文件（`outputs/`）一样只保存在本地，不随仓库提交；随仓库提交的只有这张表。需要某次运行的数字时先查这里和本地的记录，不要重跑。

延迟和 FPS 只有“计时”一列为 `solo`（运行时机器上没有别的评测任务）的才可以相互比较；每帧前向次数不受机器影响。实验的结论和上下文见 [评测记录](../evaluation-log.md)。

| 记录 | 实验 | 划分 | 方法 | 运行时间 (UTC) | Commit | 配置哈希 | 序列 | S<sub>dual</sub> | P<sub>angle</sub> | 丢失率 | 前向 / 帧 | P50 ms | P95 ms | FPS | 计时 | 说明 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `E001-tune-b0` | E001 | tune | `b0` | 2026-10-03 07:32 | `8de8d95` | `4e028e32` | 25 | 0.310 | 0.343 | 0.570 | — | 52 | 62 | 18.8 | unknown | b0，帧级用法 |
| `E001-tune-b2` | E001 | tune | `b2` | 2026-10-03 07:45 | `8de8d95` | `af87fefc` | 25 | 0.271 | 0.320 | 0.551 | — | 66 | 76 | 15.3 | unknown | 单个透视视图（旧取法），帧级 |
| `E001-tune-multiview` | E001 | tune | `ours` | 2026-10-03 08:02 | `8de8d95` | `4e028e32` | 25 | 0.065 | 0.054 | 0.820 | — | 237 | 263 | 4.4 | unknown | 四角视图 + 融合，已删除 |
| `E002-tune-fullview` | E002 | tune | `b2` | 2026-10-03 09:09 | `acd253d` | `92ac3a31` | 25 | 0.311 | 0.392 | 0.501 | — | 75 | 86 | 13.2 | unknown | fullViewSearch，未采纳 |
| `E003-tune-single` | E003 | tune | `ours` | 2026-10-06 02:27 | `acd253d*` | `b7ed6f0c` | 25 | 0.271 | 0.320 | 0.551 | — | 71 | 79 | 14.0 | unknown | 删除多视图后的单视图（旧取法） |
| `E004-test-legacy` | E004 | test | `ours` | 2026-10-06 05:26 | `c0414cf*` | `b7ed6f0c` | 120 | 0.310 | 0.327 | 0.487 | — | 95 | 132 | 9.6 | unknown | 测试集第一次运行；单视图旧取法，帧级 |
| `E004-test-b0` | E004 | test | `b0` | 2026-10-06 05:39 | `c0414cf*` | `b7ed6f0c` | 120 | 0.368 | 0.393 | 0.523 | — | 84 | 121 | 10.2 | unknown | 测试集第一次运行；b0，帧级用法 |
| `E006-tune-template-off` | E006 | tune | `ours` | 2026-10-06 07:11 | `81f9491*` | `0ba2e878` | 25 | 0.267 | 0.315 | 0.559 | — | 80 | 87 | 12.4 | unknown | 关闭在线模板，未采纳 |
| `E007-holdout-legacy` | E007 | hold-out | `ours` | 2026-10-06 09:36 | `b7ca6c0*` | `f775e982` | 41 | 0.283 | 0.368 | 0.509 | — | 67 | 74 | 14.9 | unknown | 单视图旧取法，帧级 |
| `E007-holdout-aligned-bug` | E007 | hold-out | `ours` | 2026-10-06 10:14 | `b7ca6c0*` | `a73e87d8` | 41 | 0.504 | 0.607 | 0.311 | — | 66 | 73 | 15.4 | unknown | alignedSearch，含 010 整条失败的缺陷 |
| `E007-holdout-b0` | E007 | hold-out | `b0` | 2026-10-06 23:09 | `0f6c666` | `f775e982` | 41 | 0.459 | 0.543 | 0.340 | — | 53 | 66 | 16.5 | unknown | b0，帧级用法 |
| `E008-tune-aligned` | E008 | tune | `ours` | 2026-10-06 23:16 | `0f6c666*` | `a73e87d8` | 25 | 0.421 | 0.489 | 0.428 | — | 68 | 79 | 14.6 | unknown | alignedSearch，帧级用法 |
| `E009-holdout-aligned` | E009 | hold-out | `ours` | 2026-10-07 00:09 | `39c60ef*` | `a73e87d8` | 41 | 0.520 | 0.621 | 0.292 | — | 58 | 66 | 16.5 | unknown | alignedSearch，帧级，修复 010 之后 |
| `E010-tune-seq` | E010 | tune | `ours` | 2026-10-07 01:53 | `204c5f3*` | `7e388ffa` | 25 | 0.418 | 0.489 | 0.432 | — | 69 | 80 | 14.4 | solo | 序列级模型；修复顺序依赖之前 |
| `E010-tune-seq-b0` | E010 | tune | `b0` | 2026-10-07 02:11 | `204c5f3*` | `7e388ffa` | 25 | 0.335 | 0.360 | 0.511 | — | 60 | 70 | 16.9 | solo | b0，序列级；修复顺序依赖之前 |
| `E010-holdout-seq` | E010 | hold-out | `ours` | 2026-10-07 02:26 | `204c5f3*` | `7e388ffa` | 41 | 0.526 | 0.641 | 0.314 | — | 75 | 80 | 13.2 | solo | 序列级模型；修复顺序依赖之前 |
| `E010-holdout-seq-b0` | E010 | hold-out | `b0` | 2026-10-07 03:09 | `204c5f3*` | `7e388ffa` | 41 | 0.458 | 0.547 | 0.341 | — | 59 | 67 | 16.2 | solo | b0，序列级；修复顺序依赖之前 |
| `E011-tune-sph90-prefix` | E011 | tune | `ours` | 2026-10-07 05:20 | `b4b2ac3*` | `67e72130` | 25 | 0.423 | 0.486 | 0.426 | 1.00 | 69 | 80 | 14.8 | solo | 球面采样 90° 切换；修复顺序依赖之前 |
| `E011-tune-sph90` | E011 | tune | `ours` | 2026-10-07 05:43 | `b4b2ac3*` | `67e72130` | 25 | 0.424 | 0.493 | 0.421 | 1.00 | 71 | 82 | 14.6 | solo | 修复后，球面采样 90° 切换 |
| `E011-tune-persp` | E011 | tune | `ours` | 2026-10-07 06:00 | `b4b2ac3*` | `fdb370d5` | 25 | 0.421 | 0.487 | 0.436 | 1.00 | 76 | 85 | 12.9 | solo | 修复后，sphericalSearch=false |
| `E011-tune-sph120-cap120` | E011 | tune | `ours` | 2026-10-07 06:26 | `4acbc63*` | `a7a6db24` | 25 | 0.446 | 0.519 | 0.393 | 1.00 | 74 | 85 | 13.7 | solo | 修复后，120° 切换且透视上限 120° |
| `E011-holdout-default` | E011 | hold-out | `ours` | 2026-10-07 06:47 | `4acbc63*` | `ebbd867c` | 41 | 0.528 | 0.634 | 0.277 | 1.00 | 113 | 153 | 7.1 | parallel | 修复后的默认配置（球面 120°）；当前基准 |
| `E011-tune-default` | E011 | tune | `ours` | 2026-10-07 06:47 | `4acbc63*` | `ebbd867c` | 25 | 0.424 | 0.498 | 0.427 | 1.00 | 181 | 228 | 5.6 | parallel | 修复后的默认配置（球面 120°）；当前基准 |
| `E011-holdout-persp` | E011 | hold-out | `ours` | 2026-10-07 06:48 | `4acbc63*` | `0e286d68` | 41 | 0.498 | 0.603 | 0.338 | 1.00 | 114 | 164 | 7.1 | parallel | 修复后，sphericalSearch=false |
| `E011-tune-b0` | E011 | tune | `b0` | 2026-10-07 22:32 | `d0f5195` | `ebbd867c` | 25 | 0.344 | 0.377 | 0.504 | 1.00 | 54 | 62 | 17.8 | solo | b0，修复后；当前基准 |
| `E011-holdout-b0` | E011 | hold-out | `b0` | 2026-10-07 22:46 | `d0f5195` | `ebbd867c` | 41 | 0.454 | 0.540 | 0.332 | 1.00 | 54 | 59 | 18.5 | solo | b0，修复后；当前基准 |
| `E012-tune-nopred` | E012 | tune | `ours` | 2026-10-07 23:23 | `199239d*` | `72c11a0b` | 25 | 0.408 | 0.460 | 0.434 | 1.00 | 60 | 69 | 16.8 | solo | predictiveSearch=false，未采纳 |
| `E012-holdout-nopred` | E012 | hold-out | `ours` | 2026-10-07 23:38 | `199239d*` | `72c11a0b` | 41 | 0.500 | 0.602 | 0.286 | 1.00 | 59 | 68 | 16.7 | solo | predictiveSearch=false，未采纳 |
| `E014-tune-opencv` | E014 | tune | `ours` | 2026-10-08 01:18 | `5bbbf88*` | `5b604d22` | 25 | 0.445 | 0.516 | 0.406 | 1.00 | 35 | 44 | 27.6 | solo | resampler=opencv（cv2.remap 取色） |
| `E014-tune-default-solo` | E014 | tune | `ours` | 2026-10-08 01:27 | `5bbbf88*` | `58a48d67` | 25 | 0.424 | 0.498 | 0.427 | 1.00 | 64 | 75 | 15.6 | solo | 默认配置（resampler=cpu）单独重跑，结果与 E011-tune-default 逐字节相同，用于计时 |
| `E014-holdout-opencv` | E014 | hold-out | `ours` | 2026-10-08 01:44 | `8832720*` | `5b604d22` | 41 | 0.516 | 0.618 | 0.301 | 1.00 | 41 | 56 | 22.6 | solo | resampler=opencv（cv2.remap 取色） |
| `E015-tune-align` | E015 | tune | `ours` | 2026-10-08 02:27 | `f128368*` | `5b604d22` | 25 | 0.431 | 0.505 | 0.406 | 1.00 | 38 | 51 | 24.8 | solo | BFoV 垂直方向改用纬度跨度 + ERP 包络加内部点；当前基准 |
| `E015-holdout-align` | E015 | hold-out | `ours` | 2026-10-08 02:38 | `f128368*` | `5b604d22` | 41 | 0.549 | 0.663 | 0.253 | 1.00 | 45 | 61 | 21.4 | solo | BFoV 垂直方向改用纬度跨度 + ERP 包络加内部点；当前基准 |
| `E016-tune-probe` | E016 | tune | `ours` | 2026-10-08 03:40 | `247f275*` | `5b604d22` | 25 | 0.431 | 0.505 | 0.406 | 2.00 | 81 | 122 | 11.5 | parallel | 带探针（全部信号）的运行；跟踪结果与 E015 相同，耗时包含探针 |
| `E017a-tune-loss` | E017 | tune | `ours` | 2026-10-08 04:13 | `247f275*` | `ac5ff9b4` | 25 | 0.415 | 0.468 | 0.434 | 1.67 | 57 | 186 | 7.3 | parallel | 丢失处理第一版：只对模板的相似度，采纳条件 0.50/0.50；未采纳 |
| `E017b-tune-loss` | E017 | tune | `ours` | 2026-10-08 05:06 | `305fef9*` | `030ed6e9` | 25 | 0.430 | 0.500 | 0.417 | 1.28 | 48 | 76 | 16.5 | solo | 加门控的滑动外观参考，采纳条件 0.50/0.50；持平 |
| `E017c-tune-loss` | E017 | tune | `ours` | 2026-10-08 05:48 | `c5d6251*` | `b8162653` | 25 | 0.441 | 0.515 | 0.394 | 1.40 | 48 | 71 | 15.7 | solo | 采纳条件 0.45/0.70；tune 上的最终配置 |
| `E017c-holdout-loss` | E017 | hold-out | `ours` | 2026-10-08 06:05 | `c5d6251*` | `b8162653` | 41 | 0.554 | 0.667 | 0.251 | 1.22 | 50 | 82 | 13.8 | parallel | 丢失处理最终配置（采纳条件 0.45/0.70）；前 17 条序列运行时有另一个任务并行 |
| `E018-tune-fused` | E018 | tune | `ours` | 2026-10-08 08:07 | `94e5af6*` | `2d288366` | 25 | 0.447 | 0.522 | 0.398 | 1.48 | 83 | 147 | 8.3 | parallel | 融合状态分数（0.40/0.55/0.05，阈值 0.60）的丢失处理；运行时有数据采集任务并行 |
| `E018-holdout-fused` | E018 | hold-out | `ours` | 2026-10-08 08:38 | `94e5af6*` | `2d288366` | 41 | 0.546 | 0.653 | 0.261 | 1.24 | 48 | 78 | 16.6 | parallel | 融合状态分数的丢失处理；运行时有其他任务并行 |
| `E020-tune-template-only` | E020 | tune | `ours` | 2026-10-08 09:42 | `eabbcd5*` | `73980d83` | 25 | 0.427 | 0.497 | 0.415 | 1.30 | 43 | 55 | 18.5 | parallel | 外观验证器只对第 0 帧模板，融合分 0.40/0.55/0.05、阈值 0.42，丢失处理开启；后半段有离线分析并行 |
| `E020-holdout-template-only` | E020 | hold-out | `ours` | 2026-10-08 09:56 | `eabbcd5*` | `73980d83` | 41 | 0.557 | 0.671 | 0.238 | 1.20 | 47 | 69 | 18.2 | parallel | 外观验证器只对第 0 帧模板，融合分阈值 0.42，丢失处理开启；运行时有离线分析和冒烟测试并行 |
| `E022-tune-split-judge` | E022 | tune | `ours` | 2026-10-08 10:27 | `eabbcd5*` | `c99151ad` | 25 | 0.431 | 0.501 | 0.405 | 1.00 | 46 | 58 | 18.7 | parallel | 分开规则，只判定不动作（lossActions none）；跟踪结果等于当时代码下的丢失处理关闭；开头几分钟有评测并行 |
| `E022-tune-fused-latch` | E022 | tune | `ours` | 2026-10-08 10:41 | `eabbcd5*` | `57087ee0` | 25 | 0.469 | 0.547 | 0.354 | 2.26 | 43 | 175 | 11.7 | solo | 融合分 0.42 加锁（回到 0.62 以上 3 帧才解除），扫描和跳转 |
| `E022-tune-split-jump` | E022 | tune | `ours` | 2026-10-08 11:02 | `eabbcd5*` | `bab8f102` | 25 | 0.443 | 0.496 | 0.394 | 1.92 | 42 | 163 | 13.9 | solo | 分开规则，扫描和跳转，无试用期 |
| `E022-tune-split-probation` | E022 | tune | `ours` | 2026-10-08 11:20 | `eabbcd5*` | `7032e59e` | 25 | 0.441 | 0.493 | 0.398 | 2.01 | 43 | 166 | 13.4 | solo | 分开规则，跳转后试用 10 帧，失败退回并记住干扰物 |
| `E023-tune-latch-budget40` | E023 | tune | `ours` | 2026-10-08 23:54 | `2c4b527*` | `86fed71f` | 25 | 0.453 | 0.530 | 0.370 | 1.29 | 42 | 77 | 19.4 | parallel | 融合分加锁，扫描预算每帧 0.5、存量上限 40；运行时有离线分析并行 |
| `E023-tune-latch-budget4` | E023 | tune | `ours` | 2026-10-09 00:07 | `2c4b527*` | `281b79aa` | 25 | 0.448 | 0.525 | 0.376 | 1.19 | 44 | 73 | 20.0 | parallel | 融合分加锁，扫描预算每帧 0.5、存量上限 4；运行时有离线分析并行 |
| `E023-tune-latch-unlimited` | E023 | tune | `ours` | 2026-10-09 00:20 | `2c4b527*` | `37f46338` | 25 | 0.469 | 0.548 | 0.352 | 2.25 | 42 | 170 | 12.1 | parallel | 融合分加锁，不限预算，修正置信度来源后重跑；运行时有离线分析并行 |
| `E030-train-zoom-a` | E030 | dev-train | `ours` | 2026-10-09 03:04 | `aa056fd*` | `5c5f0867` | 33 | 0.517 | 0.615 | 0.285 | 1.16 | 62 | 121 | 14.5 | parallel | 相对量判定，先粗后细：最后可信位置为中心，2 倍（前 20 帧）到 4 倍；预算 0.5 |
| `E030-train-zoom-b` | E030 | dev-train | `ours` | 2026-10-09 03:30 | `bca08e4*` | `34c05767` | 33 | 0.495 | 0.589 | 0.341 | 1.13 | 56 | 112 | 15.3 | unknown | 同 A，放大视图以跟踪器当前的框为中心 |
| `E030-train-zoom-c` | E030 | dev-train | `ours` | 2026-10-09 03:55 | `bca08e4*` | `7cf9827b` | 33 | 0.513 | 0.611 | 0.289 | 1.17 | 47 | 102 | 17.9 | unknown | 同 A，倍数 2 到 3 到 4（10 帧、40 帧） |
| `E030-train-tiles` | E030 | dev-train | `ours` | 2026-10-09 04:16 | `bca08e4*` | `bc12e406` | 33 | 0.493 | 0.582 | 0.324 | 1.20 | 50 | 89 | 16.8 | unknown | 相对量判定，分块扫描对照；预算 0.5，采纳只看分数 |
| `E031-train-zoom-same-place` | E031 | dev-train | `ours` | 2026-10-09 05:30 | `1e961b1*` | `adc4c72b` | 33 | 0.494 | 0.576 | 0.331 | 1.20 | 45 | 99 | 18.8 | parallel | E030 A 组的配置；候选落在原框上不跳转，原地重检一致时确认；运行时有离线分析并行 |
| `E033-train-a` | E033 | dev-train | `ours` | 2026-10-09 06:34 | `2499796*` | `de10028e` | 33 | 0.517 | 0.615 | 0.285 | 1.16 | 127 | 259 | 7.1 | parallel | E030 A 的行为（samePlaceAction=jump），在当前代码上复现；三个任务并行 |
| `E033-train-b` | E033 | dev-train | `ours` | 2026-10-09 06:34 | `2499796*` | `4fcbb849` | 33 | 0.507 | 0.601 | 0.306 | 1.16 | 127 | 215 | 7.1 | parallel | E030 A 的行为（候选落在原框上也跳转），放大视图的中心从可信帧按运动外推（最多 10 帧）；三个任务并行 |
| `E033-train-c` | E033 | dev-train | `ours` | 2026-10-09 06:37 | `2499796*` | `c6abed08` | 33 | 0.511 | 0.609 | 0.295 | 1.16 | 124 | 175 | 7.3 | parallel | 同 a，丢失 20 帧后取两个半重叠的 4 倍视图，两个视图指向同一处时采纳门槛降到 0.5；三个任务并行 |
| `E039-train-tf32` | E039 | dev-train | `ours` | 2026-10-10 06:01 | `ba078c9*` | `5a23258a` | 33 | 0.470 | 0.562 | 0.364 | 1.00 | 26 | 35 | 36.7 | solo | model.precision=tf32（三项无损优化之后、常量缓存之前的代码）；精度对照 |
| `E039-train-noise-fp32-s1` | E039 | dev-train | `ours` | 2026-10-10 06:21 | `ba078c9*` | `cdb29a9f` | 33 | 0.485 | 0.584 | 0.337 | 1.00 | 55 | 64 | 19.0 | parallel | fp32，网络输入加标准差 0.001 的高斯噪声（种子 1），用来估计数值噪声下分数的波动；两个任务并行 |
| `E039-train-noise-tf32-s1` | E039 | dev-train | `ours` | 2026-10-10 06:22 | `ba078c9*` | `5a23258a` | 33 | 0.479 | 0.567 | 0.356 | 1.00 | 53 | 62 | 19.1 | parallel | tf32，网络输入加标准差 0.001 的高斯噪声（种子 1），用来估计数值噪声下分数的波动；两个任务并行 |
| `E039-train-noise-fp32-s2` | E039 | dev-train | `ours` | 2026-10-10 06:41 | `ba078c9*` | `cdb29a9f` | 33 | 0.487 | 0.565 | 0.336 | 1.00 | 54 | 64 | 18.3 | parallel | fp32，网络输入加标准差 0.001 的高斯噪声（种子 2），用来估计数值噪声下分数的波动；两个任务并行 |
| `E039-train-noise-tf32-s2` | E039 | dev-train | `ours` | 2026-10-10 06:42 | `ba078c9*` | `5a23258a` | 33 | 0.495 | 0.584 | 0.334 | 1.00 | 54 | 63 | 18.5 | parallel | tf32，网络输入加标准差 0.001 的高斯噪声（种子 2），用来估计数值噪声下分数的波动；两个任务并行 |
| `E039-train-noise-fp32-s3` | E039 | dev-train | `ours` | 2026-10-10 07:02 | `ba078c9*` | `cdb29a9f` | 33 | 0.471 | 0.559 | 0.373 | 1.00 | 52 | 64 | 19.0 | parallel | fp32，网络输入加标准差 0.001 的高斯噪声（种子 3），用来估计数值噪声下分数的波动；两个任务并行 |
| `E039-train-noise-tf32-s3` | E039 | dev-train | `ours` | 2026-10-10 07:03 | `ba078c9*` | `5a23258a` | 33 | 0.479 | 0.568 | 0.351 | 1.00 | 51 | 63 | 19.2 | parallel | tf32，网络输入加标准差 0.001 的高斯噪声（种子 3），用来估计数值噪声下分数的波动；两个任务并行 |
| `E042-test-base1_fp32` | E042 | test | `ours` | 2026-10-10 14:56 | `b04b2f1` | `25ea3065` | 120 | 0.512 | 0.549 | 0.313 | 1.00 | 26 | 30 | 36.6 | solo | 360VOT 测试集，基线 1（默认配置，FP32）；单独运行 |
| `E042-test-base1_tf32` | E042 | test | `ours` | 2026-10-10 15:48 | `b04b2f1` | `a8461bae` | 120 | 0.499 | 0.532 | 0.333 | 1.00 | 24 | 31 | 39.6 | solo | 360VOT 测试集，基线 1 加 model.precision=tf32；单独运行 |
| `E042-test-base2_fp32` | E042 | test | `ours` | 2026-10-10 16:36 | `b04b2f1` | `1669b475` | 120 | 0.532 | 0.568 | 0.287 | 1.11 | 32 | 40 | 28.3 | solo | 360VOT 测试集，基线 2（丢失找回开启，FP32）；单独运行 |
| `E042-test-base2_tf32` | E042 | test | `ours` | 2026-10-10 17:43 | `b04b2f1` | `d55b51ce` | 120 | 0.510 | 0.544 | 0.309 | 1.12 | 29 | 40 | 30.9 | solo | 360VOT 测试集，基线 2 加 model.precision=tf32；单独运行 |
| `E042-test-b0_fp32` | E042 | test | `b0` | 2026-10-11 02:04 | `a561b84` | `25ea3065` | 120 | 0.368 | 0.381 | 0.532 | 1.00 | 47 | 55 | 20.3 | solo | 360VOT 测试集，b0 对照（同一个序列级后端直接在 ERP 上跟踪，FP32）；单独运行 |
| `E043-train-base1-s1` | E043 | dev-train | `ours` | 2026-10-13 05:19 | `c371554` | `25ea3065` | 33 | 0.509 | 0.609 | 0.306 | 1.00 | 27 | 31 | 36.4 | solo | 基线 1（默认配置），输入噪声种子 1（标准差 0.001） |
| `E043-train-base2-s1` | E043 | dev-train | `ours` | 2026-10-13 05:30 | `c371554*` | `1669b475` | 33 | 0.499 | 0.590 | 0.332 | 1.15 | 33 | 76 | 26.8 | solo | 基线 2（丢失找回），输入噪声种子 1（标准差 0.001） |
| `E043-train-base1-s2` | E043 | dev-train | `ours` | 2026-10-13 05:49 | `c371554*` | `25ea3065` | 33 | 0.475 | 0.567 | 0.360 | 1.00 | 51 | 60 | 20.5 | parallel | 基线 1（默认配置），输入噪声种子 2（标准差 0.001） |
| `E043-train-base2-s2` | E043 | dev-train | `ours` | 2026-10-13 05:59 | `c371554*` | `1669b475` | 33 | 0.501 | 0.596 | 0.320 | 1.16 | 60 | 144 | 14.5 | parallel | 基线 2（丢失找回），输入噪声种子 2（标准差 0.001） |
| `E043-train-base1-s3` | E043 | dev-train | `ours` | 2026-10-13 06:07 | `7d00866` | `25ea3065` | 33 | 0.484 | 0.580 | 0.344 | 1.00 | 50 | 59 | 20.4 | parallel | 基线 1（默认配置），输入噪声种子 3（标准差 0.001） |
| `E043-train-base2-s3` | E043 | dev-train | `ours` | 2026-10-13 06:25 | `7d00866` | `1669b475` | 33 | 0.485 | 0.576 | 0.338 | 1.16 | 59 | 131 | 14.9 | parallel | 基线 2（丢失找回），输入噪声种子 3（标准差 0.001） |
| `E043-train-base1-s4` | E043 | dev-train | `ours` | 2026-10-13 06:26 | `7d00866` | `25ea3065` | 33 | 0.490 | 0.573 | 0.336 | 1.00 | 50 | 58 | 20.3 | parallel | 基线 1（默认配置），输入噪声种子 4（标准差 0.001） |
| `E043-train-base1-s5` | E043 | dev-train | `ours` | 2026-10-13 06:44 | `7d00866` | `25ea3065` | 33 | 0.470 | 0.561 | 0.360 | 1.00 | 50 | 58 | 20.5 | parallel | 基线 1（默认配置），输入噪声种子 5（标准差 0.001） |
| `E043-train-base2-s4` | E043 | dev-train | `ours` | 2026-10-13 06:50 | `7d00866` | `1669b475` | 33 | 0.471 | 0.563 | 0.350 | 1.18 | 60 | 144 | 14.3 | parallel | 基线 2（丢失找回），输入噪声种子 4（标准差 0.001） |
| `E043-train-base2-s5` | E043 | dev-train | `ours` | 2026-10-13 07:16 | `7d00866` | `1669b475` | 33 | 0.464 | 0.552 | 0.369 | 1.16 | 57 | 83 | 17.1 | parallel | 基线 2（丢失找回），输入噪声种子 5（标准差 0.001） |
| `E048-train-base2-s1` | E048 | dev-train | `ours` | 2026-10-14 03:25 | `b23b004` | `1669b475` | 33 | 0.484 | 0.575 | 0.334 | 1.16 | 32 | 79 | 26.7 | solo | 基线 2（丢失找回），输入噪声种子 1（标准差 0.00001） |
| `E048-train-base2-s2` | E048 | dev-train | `ours` | 2026-10-14 03:39 | `b23b004` | `1669b475` | 33 | 0.468 | 0.550 | 0.348 | 1.16 | 35 | 57 | 25.0 | solo | 基线 2（丢失找回），输入噪声种子 2（标准差 0.00001） |
| `E048-train-base2-s3` | E048 | dev-train | `ours` | 2026-10-14 03:54 | `b23b004` | `1669b475` | 33 | 0.500 | 0.594 | 0.301 | 1.17 | 34 | 69 | 25.1 | solo | 基线 2（丢失找回），输入噪声种子 3（标准差 0.00001） |
| `E048-train-base2-s4` | E048 | dev-train | `ours` | 2026-10-14 04:09 | `b23b004` | `1669b475` | 33 | 0.487 | 0.573 | 0.339 | 1.15 | 35 | 59 | 25.0 | solo | 基线 2（丢失找回），输入噪声种子 4（标准差 0.00001） |
| `E048-train-base2-s5` | E048 | dev-train | `ours` | 2026-10-14 04:24 | `b23b004` | `1669b475` | 33 | 0.483 | 0.572 | 0.335 | 1.18 | 37 | 84 | 22.5 | solo | 基线 2（丢失找回），输入噪声种子 5（标准差 0.00001） |
| `E048-train-base1-s1` | E048 | dev-train | `ours` | 2026-10-14 04:41 | `b23b004` | `25ea3065` | 33 | 0.482 | 0.578 | 0.347 | 1.00 | 27 | 31 | 35.3 | solo | 基线 1（默认配置），输入噪声种子 1（标准差 0.00001） |
| `E048-train-base1-s2` | E048 | dev-train | `ours` | 2026-10-14 04:51 | `b23b004` | `25ea3065` | 33 | 0.487 | 0.582 | 0.340 | 1.00 | 28 | 39 | 33.9 | solo | 基线 1（默认配置），输入噪声种子 2（标准差 0.00001） |
| `E048-train-base1-s3` | E048 | dev-train | `ours` | 2026-10-14 05:02 | `b23b004` | `25ea3065` | 33 | 0.506 | 0.602 | 0.310 | 1.00 | 28 | 38 | 33.2 | solo | 基线 1（默认配置），输入噪声种子 3（标准差 0.00001） |
| `E048-train-base1-s4` | E048 | dev-train | `ours` | 2026-10-14 05:14 | `b23b004` | `25ea3065` | 33 | 0.493 | 0.585 | 0.335 | 1.00 | 28 | 37 | 33.5 | solo | 基线 1（默认配置），输入噪声种子 4（标准差 0.00001） |
| `E048-train-base1-s5` | E048 | dev-train | `ours` | 2026-10-14 05:25 | `b23b004` | `25ea3065` | 33 | 0.481 | 0.570 | 0.353 | 1.00 | 27 | 31 | 35.7 | solo | 基线 1（默认配置），输入噪声种子 5（标准差 0.00001） |
| `E051-train-judgeT1-s1` | E051 | dev-train | `ours` | 2026-10-15 01:08 | `b5e4216` | `1669b475` | 33 | 0.511 | 0.610 | 0.320 | 1.20 | 30 | 40 | 28.0 | unknown | candidate judge T1 (threshold 0.3855), seed 1 |
| `E051-train-judgeT1-s2` | E051 | dev-train | `ours` | 2026-10-15 01:21 | `e7b3fec` | `1669b475` | 33 | 0.494 | 0.587 | 0.305 | 1.23 | 30 | 75 | 28.3 | unknown | candidate judge T1 (threshold 0.3855), seed 2 |
| `E051-train-judgeT1-s3` | E051 | dev-train | `ours` | 2026-10-15 01:35 | `e7b3fec` | `1669b475` | 33 | 0.482 | 0.573 | 0.336 | 1.22 | 30 | 35 | 28.4 | unknown | candidate judge T1 (threshold 0.3855), seed 3 |
| `E051-train-judgeT1-s4` | E051 | dev-train | `ours` | 2026-10-15 01:48 | `e7b3fec` | `1669b475` | 33 | 0.466 | 0.551 | 0.351 | 1.24 | 30 | 55 | 27.9 | unknown | candidate judge T1 (threshold 0.3855), seed 4 |
| `E051-train-judgeT1-s5` | E051 | dev-train | `ours` | 2026-10-15 02:01 | `e7b3fec` | `1669b475` | 33 | 0.486 | 0.577 | 0.330 | 1.22 | 30 | 52 | 28.2 | unknown | candidate judge T1 (threshold 0.3855), seed 5 |
| `E051-train-judgeT2-s1` | E051 | dev-train | `ours` | 2026-10-15 02:14 | `e7b3fec` | `1669b475` | 33 | 0.495 | 0.584 | 0.322 | 1.24 | 30 | 93 | 28.1 | unknown | candidate judge T2 (threshold 0.4839), seed 1 |
| `E051-train-judgeT2-s2` | E051 | dev-train | `ours` | 2026-10-15 02:28 | `e7b3fec` | `1669b475` | 33 | 0.492 | 0.582 | 0.323 | 1.27 | 30 | 94 | 27.3 | unknown | candidate judge T2 (threshold 0.4839), seed 2 |
| `E051-train-judgeT2-s3` | E051 | dev-train | `ours` | 2026-10-15 02:41 | `e7b3fec` | `1669b475` | 33 | 0.477 | 0.567 | 0.347 | 1.25 | 30 | 36 | 27.8 | unknown | candidate judge T2 (threshold 0.4839), seed 3 |
| `E051-train-judgeT2-s4` | E051 | dev-train | `ours` | 2026-10-15 02:55 | `e7b3fec` | `1669b475` | 33 | 0.474 | 0.558 | 0.330 | 1.30 | 31 | 96 | 26.3 | unknown | candidate judge T2 (threshold 0.4839), seed 4 |
| `E051-train-judgeT2-s5` | E051 | dev-train | `ours` | 2026-10-15 03:09 | `e7b3fec` | `1669b475` | 33 | 0.485 | 0.568 | 0.321 | 1.24 | 30 | 95 | 27.4 | unknown | candidate judge T2 (threshold 0.4839), seed 5 |
| `E052-train-oracleAhi-s1` | E052 | dev-train | `ours` | 2026-10-22 02:43 | `cd6d61e` | `1669b475` | 21 | 0.359 | 0.429 | 0.436 | 1.25 | 31 | 78 | 26.9 | unknown | partial oracle A=0.55 (evaluation only), seed 1 |
| `E052-train-oracleAhi-s2` | E052 | dev-train | `ours` | 2026-10-22 02:53 | `cd6d61e*` | `1669b475` | 21 | 0.388 | 0.473 | 0.401 | 1.28 | 31 | 79 | 26.1 | unknown | partial oracle A=0.55 (evaluation only), seed 2 |
| `E052-train-oracleBhi-s1` | E052 | dev-train | `ours` | 2026-10-22 03:03 | `cd6d61e*` | `1669b475` | 21 | 0.410 | 0.501 | 0.397 | 1.15 | 35 | 75 | 25.7 | unknown | partial oracle B=0.43 (evaluation only), seed 1 |
| `E052-train-oracleBhi-s2` | E052 | dev-train | `ours` | 2026-10-22 03:14 | `cd6d61e*` | `1669b475` | 21 | 0.435 | 0.543 | 0.356 | 1.17 | 32 | 76 | 27.8 | unknown | partial oracle B=0.43 (evaluation only), seed 2 |
| `E052-train-oracleChi-s1` | E052 | dev-train | `ours` | 2026-10-22 03:23 | `cd6d61e*` | `1669b475` | 21 | 0.407 | 0.497 | 0.384 | 1.21 | 32 | 78 | 26.8 | unknown | partial oracle C=0.49 (evaluation only), seed 1 |
| `E052-train-oracleChi-s2` | E052 | dev-train | `ours` | 2026-10-22 03:33 | `cd6d61e*` | `1669b475` | 21 | 0.419 | 0.523 | 0.362 | 1.21 | 32 | 80 | 25.8 | unknown | partial oracle C=0.49 (evaluation only), seed 2 |
| `E052-train-oracleDhi-s1` | E052 | dev-train | `ours` | 2026-10-22 03:44 | `cd6d61e*` | `1669b475` | 21 | 0.386 | 0.463 | 0.414 | 1.19 | 38 | 85 | 23.2 | unknown | partial oracle D=0.45 (evaluation only), seed 1 |
| `E052-train-oracleDhi-s2` | E052 | dev-train | `ours` | 2026-10-22 03:55 | `cd6d61e*` | `1669b475` | 21 | 0.386 | 0.475 | 0.420 | 1.21 | 39 | 91 | 22.5 | unknown | partial oracle D=0.45 (evaluation only), seed 2 |
| `E052-train-oracleEhi-s1` | E052 | dev-train | `ours` | 2026-10-22 04:08 | `cd6d61e*` | `1669b475` | 33 | 0.496 | 0.578 | 0.316 | 1.17 | 39 | 86 | 22.7 | unknown | partial oracle E=0.75 (evaluation only), seed 1 |
| `E052-train-oracleEhi-s2` | E052 | dev-train | `ours` | 2026-10-22 04:24 | `cd6d61e*` | `1669b475` | 33 | 0.496 | 0.594 | 0.314 | 1.18 | 40 | 88 | 22.3 | unknown | partial oracle E=0.75 (evaluation only), seed 2 |
| `E052-train-oracleBlo-s1` | E052 | dev-train | `ours` | 2026-10-22 04:43 | `663fa6a` | `1669b475` | 21 | 0.412 | 0.502 | 0.394 | 1.15 | 38 | 79 | 24.2 | unknown | partial oracle B=0.21 (evaluation only), seed 1 |
| `E052-train-oracleBlo-s2` | E052 | dev-train | `ours` | 2026-10-22 04:54 | `663fa6a` | `1669b475` | 21 | 0.445 | 0.552 | 0.339 | 1.20 | 39 | 89 | 23.0 | unknown | partial oracle B=0.21 (evaluation only), seed 2 |
| `E052-train-oracleClo-s1` | E052 | dev-train | `ours` | 2026-10-22 05:06 | `663fa6a` | `1669b475` | 21 | 0.396 | 0.483 | 0.400 | 1.22 | 39 | 91 | 22.0 | unknown | partial oracle C=0.24 (evaluation only), seed 1 |
| `E052-train-oracleClo-s2` | E052 | dev-train | `ours` | 2026-10-22 05:19 | `663fa6a` | `1669b475` | 21 | 0.409 | 0.509 | 0.379 | 1.22 | 39 | 90 | 22.5 | unknown | partial oracle C=0.24 (evaluation only), seed 2 |
| `E052-train-oracleBCDlo-s1` | E052 | dev-train | `ours` | 2026-10-22 05:31 | `663fa6a` | `1669b475` | 21 | 0.488 | 0.596 | 0.279 | 1.17 | 39 | 83 | 23.3 | unknown | partial oracle B=0.21 C=0.24 D=0.21 (evaluation only), seed 1 |
| `E052-train-oracleBCDlo-s2` | E052 | dev-train | `ours` | 2026-10-22 05:42 | `663fa6a` | `1669b475` | 21 | 0.476 | 0.576 | 0.293 | 1.17 | 38 | 83 | 23.8 | unknown | partial oracle B=0.21 C=0.24 D=0.21 (evaluation only), seed 2 |
| `E052-train-oracleBCDlo-s3` | E052 | dev-train | `ours` | 2026-10-22 05:54 | `663fa6a` | `1669b475` | 21 | 0.472 | 0.582 | 0.278 | 1.17 | 39 | 82 | 22.5 | unknown | partial oracle B=0.21 C=0.24 D=0.21 (evaluation only), seed 3 |
| `E052-train-oracleBCDhi-s1` | E052 | dev-train | `ours` | 2026-10-22 06:06 | `663fa6a` | `1669b475` | 21 | 0.519 | 0.644 | 0.215 | 1.15 | 33 | 77 | 26.0 | unknown | partial oracle B=0.43 C=0.49 D=0.45 (evaluation only), seed 1 |
| `E052-train-oracleBCDhi-s2` | E052 | dev-train | `ours` | 2026-10-22 06:16 | `663fa6a` | `1669b475` | 21 | 0.499 | 0.618 | 0.227 | 1.10 | 32 | 57 | 28.7 | unknown | partial oracle B=0.43 C=0.49 D=0.45 (evaluation only), seed 2 |
| `E052-train-oracleABCDhi-s1` | E052 | dev-train | `ours` | 2026-10-22 06:25 | `663fa6a` | `1669b475` | 21 | 0.514 | 0.630 | 0.205 | 1.17 | 32 | 58 | 27.6 | unknown | partial oracle A=0.55 B=0.43 C=0.49 D=0.45 (evaluation only), seed 1 |
| `E052-train-oracleABCDhi-s2` | E052 | dev-train | `ours` | 2026-10-22 06:35 | `663fa6a` | `1669b475` | 21 | 0.535 | 0.668 | 0.146 | 1.14 | 31 | 56 | 29.0 | unknown | partial oracle A=0.55 B=0.43 C=0.49 D=0.45 (evaluation only), seed 2 |
| `E053-train-camera-s1` | E053 | dev-train | `ours` | 2026-10-22 09:23 | `4ec45b0` | `1669b475` | 33 | 0.538 | 0.638 | 0.277 | 1.14 | 40 | 62 | 17.9 | unknown | base 2 + enlarged searches turned with the camera rotation, seed 1 |
| `E053-train-camera-s2` | E053 | dev-train | `ours` | 2026-10-22 09:44 | `4ec45b0*` | `1669b475` | 33 | 0.502 | 0.597 | 0.310 | 1.13 | 37 | 74 | 20.1 | unknown | base 2 + enlarged searches turned with the camera rotation, seed 2 |
| `E053-train-camera-s3` | E053 | dev-train | `ours` | 2026-10-22 10:03 | `4ec45b0*` | `1669b475` | 33 | 0.509 | 0.610 | 0.303 | 1.14 | 38 | 65 | 18.3 | unknown | base 2 + enlarged searches turned with the camera rotation, seed 3 |
| `E053-train-camera-s4` | E053 | dev-train | `ours` | 2026-10-22 10:23 | `4ec45b0*` | `1669b475` | 33 | 0.477 | 0.566 | 0.353 | 1.15 | 36 | 57 | 19.0 | unknown | base 2 + enlarged searches turned with the camera rotation, seed 4 |
| `E053-train-camera-s5` | E053 | dev-train | `ours` | 2026-10-22 10:43 | `4ec45b0*` | `1669b475` | 33 | 0.492 | 0.590 | 0.323 | 1.15 | 36 | 80 | 18.9 | unknown | base 2 + enlarged searches turned with the camera rotation, seed 5 |
| `E054-train-global-g10-s1` | E054 | dev-train | `ours` | 2026-10-22 11:20 | `23093c4` | `1669b475` | 33 | 0.546 | 0.651 | 0.255 | 1.13 | 37 | 52 | 18.2 | unknown | E053 + whole-sphere OWLv2 detection (g10), seed 1 |
| `E054-train-global-g10-s2` | E054 | dev-train | `ours` | 2026-10-22 11:41 | `23093c4` | `1669b475` | 33 | 0.510 | 0.608 | 0.293 | 1.13 | 37 | 65 | 18.2 | unknown | E053 + whole-sphere OWLv2 detection (g10), seed 2 |
| `E054-train-global-g5-s1` | E054 | dev-train | `ours` | 2026-10-22 12:02 | `23093c4` | `1669b475` | 33 | 0.517 | 0.613 | 0.310 | 1.13 | 37 | 51 | 17.2 | unknown | E053 + whole-sphere OWLv2 detection (g5), seed 1 |
| `E054-train-global-g5-s2` | E054 | dev-train | `ours` | 2026-10-22 12:23 | `23093c4` | `1669b475` | 33 | 0.503 | 0.594 | 0.307 | 1.14 | 37 | 62 | 17.5 | unknown | E053 + whole-sphere OWLv2 detection (g5), seed 2 |
| `E054-train-global-g10-s3` | E054 | dev-train | `ours` | 2026-10-22 12:46 | `23093c4` | `1669b475` | 33 | 0.510 | 0.611 | 0.302 | 1.12 | 37 | 56 | 18.3 | unknown | E053 + whole-sphere OWLv2 detection (g10), seed 3 |
| `E054-train-global-g10-s4` | E054 | dev-train | `ours` | 2026-10-22 13:07 | `2973b4f` | `1669b475` | 33 | 0.482 | 0.576 | 0.333 | 1.15 | 37 | 59 | 18.1 | unknown | E053 + whole-sphere OWLv2 detection (g10), seed 4 |
| `E054-train-global-g10-s5` | E054 | dev-train | `ours` | 2026-10-22 13:27 | `2973b4f` | `1669b475` | 33 | 0.497 | 0.597 | 0.309 | 1.13 | 37 | 64 | 18.3 | unknown | E053 + whole-sphere OWLv2 detection (g10), seed 5 |
| `E054-train-global-late-s1` | E054 | dev-train | `ours` | 2026-10-22 13:48 | `2973b4f` | `1669b475` | 33 | 0.529 | 0.628 | 0.289 | 1.08 | 37 | 54 | 18.9 | unknown | E053 + whole-sphere OWLv2 detection (late), seed 1 |
| `E054-train-global-late-s2` | E054 | dev-train | `ours` | 2026-10-22 14:08 | `2973b4f` | `1669b475` | 33 | 0.500 | 0.593 | 0.309 | 1.10 | 37 | 55 | 18.8 | unknown | E053 + whole-sphere OWLv2 detection (late), seed 2 |

Commit 后的 `*` 表示运行时工作区有未提交的改动。
