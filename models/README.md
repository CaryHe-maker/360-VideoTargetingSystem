# 模型权重

默认配置读取本目录下的 `artrackv2_b_256.pth.tar`，即官方 [ARTrackV2-B-256](https://github.com/MIV-XJTU/ARTrack) checkpoint（约 1.6 GB）。这是官方的**序列级**权重（训练 40 个 epoch，带轨迹位置嵌入和外观解码器，共 303 组参数），默认按序列级模型加载，见 [Backends](../docs/modules/backends.md#这份权重是序列级模型)。权重**不随仓库分发**，需要单独下载后放到本目录，并重命名为 `artrackv2_b_256.pth.tar`。本目录中除本说明外的文件都被 `.gitignore` 忽略，不会被提交。

## 获取方式

1. **命令（推荐）**：

   ```bash
   track360 download
   ```

   从作者发布在 Google Drive 上的文件下载到本目录，并核对下面的 SHA-256；中断后再运行会接着下。`--url` 可以换来源，`--check` 只核对已有的文件。这个命令对真实的下载地址还没有完整验证过（下载需要 1.6 GB 流量），如果它报告“返回的是网页”，请用下面的方法。
2. **手动**：从 [ARTrack 仓库](https://github.com/MIV-XJTU/ARTrack) README 中的 ARTrackV2-B-256 链接下载（Google Drive，或者百度网盘），放到本目录并重命名为 `artrackv2_b_256.pth.tar`，再运行 `track360 download --check`。
3. **本仓库的历史版本**：提交 `f04c76d` 及更早的提交通过 Git LFS 保存了这个文件，可以这样取出：

   ```bash
   git lfs install
   git checkout f04c76d -- models/artrackv2_b_256.pth.tar
   git restore --staged models/artrackv2_b_256.pth.tar
   ```

   这种方式会消耗 GitHub LFS 流量，只建议在官方链接不可用时使用。

上游仓库的代码是 Apache-2.0；它的 README 同时写明“This project is not for commercial use”，权重没有单独的许可声明。本项目不再分发这份权重。

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

## 外观模型（可选）

丢失处理（`--preset loss_handling`）和外观探针（`tools/benchmark.py run --probe`）用到的小模型放在 `models/hub/`，由 `torch.hub` 在第一次使用时下载，同样不随仓库分发：

| 模型 | 来源 | 许可证 | 大小 |
|---|---|---|---:|
| DINOv2 ViT-S/14（默认的验证器） | `facebookresearch/dinov2` | Apache 2.0 | 约 85 MB |
| DINO ViT-S/16（只用于对比） | `facebookresearch/dino` | Apache 2.0 | 约 83 MB |
| ResNet-18（只用于对比） | torchvision 的 ImageNet 权重 | BSD-3 | 约 45 MB |

