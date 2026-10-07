# -*- coding:utf-8 -*-
# Derived from ARTrack (https://github.com/MIV-XJTU/ARTrack):
#   lib/models/layers/mask_decoder.py   author: Skye Song, Copyright (c) Skye-Song
#   lib/models/mask_decoder/{attention,block,mlp}.py
#
# Reduced to what inference needs.  The upstream module also crops search features
# with PrRoIPool (a compiled extension) and computes a reconstruction loss during
# training; neither runs at inference, so both are left out, and the transformer
# block it builds on is inlined here.  Parameter names are unchanged, so the
# released checkpoints load with ``strict=True``.

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, head, N, C//head)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, head, N, N)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MaskDecoder(nn.Module):
    def __init__(self, patch_size=16, num_patches=8 ** 2, embed_dim=1024, decoder_embed_dim=512,
                 decoder_depth=8, decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # Only used to mask tokens during training; kept so the checkpoint loads strictly.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Fixed sin-cos embedding, stored in the checkpoint.
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3, bias=True)  # decoder to patch

    def forward_decoder(self, x):
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

        return x

    def forward(self, x, eval=True):
        # input x = [B,C,H,W]; upstream: rearrange(x, 'b c h w -> b (h w) c')
        if not eval:
            raise NotImplementedError("this copy of MaskDecoder only supports inference")
        x = x.flatten(2).transpose(1, 2).contiguous()
        return self.unpatchify(self.forward_decoder(x))


def build_maskdecoder(cfg, hidden_dim):
    num_patches = (cfg.DATA.TEMPLATE.SIZE // cfg.MODEL.BACKBONE.PATCHSIZE) ** 2

    model = MaskDecoder(
        patch_size=cfg.MODEL.BACKBONE.PATCHSIZE,
        num_patches=num_patches,
        embed_dim=hidden_dim,
        decoder_embed_dim=cfg.MODEL.DECODER.EMBEDDIM,
        decoder_depth=cfg.MODEL.DECODER.DEPTH,
        decoder_num_heads=cfg.MODEL.DECODER.NUMHEADS,
        mlp_ratio=cfg.MODEL.DECODER.MLPRATIO,
        norm_layer=nn.LayerNorm)
    return model
