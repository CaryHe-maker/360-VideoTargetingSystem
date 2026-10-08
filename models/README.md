# 模型权重

默认配置读取本目录下的 `artrackv2_b_256.pth.tar`，即官方 [ARTrackV2-B-256](https://github.com/MIV-XJTU/ARTrack) checkpoint（约 1.6 GB）。这是官方的**序列级**权重（训练 40 个 epoch，带轨迹位置嵌入和外观解码器，共 303 组参数），默认按序列级模型加载，见 [Backends](../docs/modules/backends.md#这份权重是序列级模型)。权重**不随仓库分发**，需要单独下载后放到本目录，并重命名为 `artrackv2_b_256.pth.tar`。本目录中除本说明外的文件都被 `.gitignore` 忽略，不会被提交。

## 获取方式

1. **官方发布（推荐）**：从 ARTrack 仓库 README 中的 ARTrackV2-B-256 下载链接获取（Google Drive，或者百度网盘）。
2. **本仓库的历史版本**：提交 `f04c76d` 及更早的提交通过 Git LFS 保存了这个文件，可以这样取出：

   ```bash
   git lfs install
   git checkout f04c76d -- models/artrackv2_b_256.pth.tar
   git restore --staged models/artrackv2_b_256.pth.tar
   ```

   这种方式会消耗 GitHub LFS 流量，只建议在官方链接不可用时使用。

## 校验

本项目开发和评测使用的文件 SHA-256：

```text
a99b7f8086e4827ecfe32ec8a9d32ad41c1ca9ff3cac551b62ec95576ca01d05
```

```bash
sha256sum models/artrackv2_b_256.pth.tar
```

Windows PowerShell 可以使用 `Get-FileHash models\artrackv2_b_256.pth.tar`。从官方链接下载的文件尚未与该哈希核对；如果不一致，请先确认下载的是 ARTrackV2-B-256（而不是 GOT 版本），并把结果记录到 [评测记录](../docs/evaluation-log.md)。

文件中应包含 `net` 状态字典，代码会严格加载 ViT-B、搜索尺寸 256 对应的全部参数。

## 分数校准（可选）

ARTrackV2 输出的分数不是校准过的概率。如果需要，可以把与本权重绑定的校准 JSON 路径写入 `configs/default.yaml` 的 `scoring.calibrationArtifact`。加载时会校验产物中记录的权重 SHA-256，以及阈值是否与 YAML 一致。没有校准产物时，运行时直接使用原始分数。

## 计划

按照 [V2Plan](../docs/V2Plan.md)，之后会把权重发布到 Hugging Face Hub 或 GitHub Releases，并提供 `track360 download` 命令自动下载和校验。
