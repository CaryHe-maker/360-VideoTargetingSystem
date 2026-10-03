# 模型权重

默认配置读取本目录下的 `artrackv2_b_256.pth.tar`：官方 [ARTrackV2-B-256](https://github.com/miv-xjtu/artrack) checkpoint，约 1.6 GB，通过 Git LFS 分发。

```bash
git lfs install
git lfs pull
```

文件中应包含 `net` 状态字典，代码会严格加载 ViT-B、搜索尺寸 256 对应的全部参数。

当前文件的 SHA-256：

```text
a99b7f8086e4827ecfe32ec8a9d32ad41c1ca9ff3cac551b62ec95576ca01d05
```

## 分数校准（可选）

ARTrackV2 输出的分数不是校准过的概率。如果需要，可以把与本权重绑定的校准 JSON 路径写入 `configs/default.yaml` 的 `scoring.calibrationArtifact`。加载时会校验产物中记录的权重 SHA-256 以及阈值是否与 YAML 一致。没有校准产物时，运行时直接使用原始分数。

## 计划

GitHub LFS 的免费流量有限。按照 [V2Plan](../docs/V2Plan.md)，权重之后会改为从 Hugging Face Hub 或 GitHub Releases 下载并校验，不再放在 Git 仓库中。
