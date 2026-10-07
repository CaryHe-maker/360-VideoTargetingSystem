"""
ARTrackV2 sequence-level model.

Derived from ARTrack (https://github.com/MIV-XJTU/ARTrack),
lib/models/artrackv2_seq/artrackv2_seq.py.  Reduced to what inference needs:

* the training branch of ``forward`` (appearance reconstruction against the ground
  truth crop) is removed;
* ``build_artrackv2_seq`` no longer loads a checkpoint from ``cfg.MODEL.PRETRAIN_PTH``
  (the caller loads the weights), and the unused decoder-head imports are dropped.
"""
import torch
from torch import nn
from timm.models.layers import trunc_normal_

from lib.models.artrackv2_seq.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.models.layers.mask_decoder import build_maskdecoder


class ARTrackV2Seq(nn.Module):

    def __init__(self, transformer,
                 cross_2_decoder,
                 score_mlp,
                 hidden_dim,
                 ):
        super().__init__()
        self.backbone = transformer
        self.score_mlp = score_mlp

        self.identity = torch.nn.Parameter(torch.zeros(1, 3, hidden_dim))
        self.identity = trunc_normal_(self.identity, std=.02)

        self.cross_2_decoder = cross_2_decoder

    def forward(self, template: torch.Tensor,
                dz_feat: torch.Tensor,
                search: torch.Tensor,
                seq_input=None,
                ):
        template_0 = template[:, 0]
        out, z_0_feat, z_1_feat, x_feat, score_feat = self.backbone(z_0=template_0, z_1_feat=dz_feat, x=search, identity=self.identity, seqs_input=seq_input)

        score = self.score_mlp(score_feat)
        out['score'] = score

        z_1_feat = z_1_feat.reshape(z_1_feat.shape[0], int(z_1_feat.shape[1] ** 0.5), int(z_1_feat.shape[1] ** 0.5),
                                    z_1_feat.shape[2]).permute(0, 3, 1, 2)
        update_feat = self.cross_2_decoder(z_1_feat, eval=True)
        update_feat = self.cross_2_decoder.patchify(update_feat)
        out['dz_feat'] = update_feat

        return out

class MlpScoreDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, bn=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        out_dim = 1 # score
        if bn:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])
        else:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Linear(n, k)
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])

    def forward(self, reg_tokens):
        """
        reg tokens shape: (b, 4, embed_dim)
        """
        x = self.layers(reg_tokens) # (b, 4, 1)
        x = x.mean(dim=1)   # (b, 1)
        return x

def build_score_decoder(cfg, hidden_dim):
    return MlpScoreDecoder(
        in_dim=hidden_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        bn=False
    )

def build_artrackv2_seq(cfg, training=True):
    if training:
        raise NotImplementedError("this copy of ARTrackV2Seq only supports inference")
    pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION, prenum=cfg.MODEL.PRENUM)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION, prenum=cfg.MODEL.PRENUM)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    cross_2_decoder = build_maskdecoder(cfg, hidden_dim)
    score_mlp = build_score_decoder(cfg, hidden_dim)

    model = ARTrackV2Seq(
        backbone,
        cross_2_decoder,
        score_mlp,
        hidden_dim,
    )
    return model
