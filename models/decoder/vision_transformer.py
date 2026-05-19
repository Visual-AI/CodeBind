# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Model codes for CodeBind
# @Reference     : ImageBind, Meta Platforms, Inc. and affiliates.
#                  https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py ;
#                  https://github.com/facebookresearch/deit/blob/main/models.py
#                  https://github.com/facebookresearch/vissl/blob/main/vissl/models/trunks/vision_transformer.py
import math
from functools import partial
from typing import List, Optional

# import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.modules.utils import _ntuple
import pdb

from models.multimodal_preprocessors import get_sinusoid_encoding_table, _get_pos_embedding
from models.transformer import Mlp


class Block(nn.Module):
    def __init__(
        self,
        dim,
        attn_target,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale_type=None,  # from cait; possible values are None, "per_channel", "scalar"
        layer_scale_init_value=1e-4,  # from cait; float
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if isinstance(attn_target, nn.Module):
            self.attn = attn_target
        else:
            self.attn = attn_target(dim=dim)

        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale_type = layer_scale_type

        # Layerscale
        if self.layer_scale_type is not None:
            assert self.layer_scale_type in [
                "per_channel",
                "scalar",
            ], f"Found Layer scale type {self.layer_scale_type}"
            if self.layer_scale_type == "per_channel":
                # one gamma value per channel
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == "scalar":
                # single gamma value for all channels
                gamma_shape = [1, 1, 1]
            # two gammas: for each part of the fwd in the encoder
            self.layer_scale_gamma1 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )
            self.layer_scale_gamma2 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

    def forward(self, x):
        if self.layer_scale_type is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)) * self.layer_scale_gamma1)
            x = x + self.drop_path(self.mlp(self.norm2(x)) * self.layer_scale_gamma2)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        first_patch_idx,        # adding first_patch_idx since it is 1 if there is cls token, else 0
        patches_layout,         # patches layout on original image size (1, patches_num_h, patches_num_w)
        attn_target,            # class Attention(nn.Module)
        embed_dim,              # for build_pos_embedding
        decoder_embed_dim=512,  # dim in the attention block
        input_proj_dim=None,
        decoder_depth=8,        # num of stacked attention blocks
        drop_path_rate=0.0,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_norm_eps=1e-6,
        return_interim_layers=False,
        share_pos_embed=False,
        learnable_pos_embed=True,
        init_pos_embed_random=False,
        layer_scale_type=None,          # from cait; possible values are None, "per_channel", "scalar"
        layer_scale_init_value=1e-4,    # from cait; float
        final_projection=None,
        pos_sum_embed_only=False,
        **kwargs,
    ):
        super().__init__()
        self.patches_layout = patches_layout
        self.first_patch_idx = first_patch_idx     
        assert first_patch_idx == 0 or first_patch_idx == 1
        self.share_pos_embed = share_pos_embed

        self.build_pos_embedding(
            share_pos_embed=share_pos_embed,
            learnable_pos_embed=learnable_pos_embed,
            patches_layout=self.patches_layout,
            first_patch_idx=first_patch_idx,
            embed_dim=embed_dim,
            init_pos_embed_random=init_pos_embed_random,
        )
        self.pos_sum_embed_only = pos_sum_embed_only
        if pos_sum_embed_only:
            # another flag to catch if someone set this accidentally
            # recommended to use the `PosEmbedSumDecoder` class
            assert decoder_depth == -1, "Do not specify decoder_depth"
            return
        norm_layer = partial(nn.LayerNorm, eps=layer_norm_eps)
        self.norm = norm_layer(decoder_embed_dim)
        if input_proj_dim is not None:
            self.decoder_input_proj = nn.Linear(embed_dim, input_proj_dim, bias=True)                                    
            self.decoder_embed = nn.Linear(input_proj_dim, decoder_embed_dim, bias=True)
            self.input_proj_norm = norm_layer(input_proj_dim)
        else:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)
        ]  # stochastic depth decay rule

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(decoder_depth)
            ]
        )
        self.return_interim_layers = return_interim_layers
        self.final_projection = None
        # if final_projection is not None:
        #     self.final_projection = hydra.utils.instantiate(
        #         final_projection, _convert_="all", _recursive_=False
        #     )
        print("Build Decoder: decoder_depth =", decoder_depth)

    def build_pos_embedding(
        self,
        share_pos_embed,
        learnable_pos_embed,
        patches_layout,
        first_patch_idx,
        embed_dim,
        init_pos_embed_random,
    ):
        if share_pos_embed is True:
            # we expect pos_embed to be passed during `forward`
            # sharing nn.Parameter objects across modules is not recommended practice in PyTorch
            self.pos_embed = None
        elif learnable_pos_embed is True:
            self.pos_embed = nn.Parameter(
                # adding first_patch_idx since it is 1 if there is cls token, else 0
                torch.zeros(1, np.prod(patches_layout) + first_patch_idx, embed_dim)  # np.prod: the product of array elements
            )
            if init_pos_embed_random:
                trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_buffer(
                "pos_embed",
                get_sinusoid_encoding_table(
                    np.prod(patches_layout) + first_patch_idx, embed_dim
                ),
            )
    
    def get_pos_embedding(self, vision_input_shape, all_vision_tokens, input_pos_embed):
        pos_embed = _get_pos_embedding(
            all_vision_tokens.size(1) - self.first_patch_idx,
            pos_embed=input_pos_embed,
            patches_layout=self.patches_layout,
            input_shape=vision_input_shape,
            first_patch_idx=self.first_patch_idx,
        )
        return pos_embed

    def forward(
        self,
        x,
        orig_input_shape,
        input_pos_embed=None,
        input_trunk_embed=None,
        use_checkpoint=False,
    ):
        if self.share_pos_embed:
            assert input_pos_embed is not None, "Must specify encoder pos_embed when sharing pos_embed in decoder."
        if input_pos_embed is not None and self.first_patch_idx == 0:
                input_pos_embed = input_pos_embed[:, 1:, ...]
        curr_pos_embed = input_pos_embed if self.share_pos_embed else self.pos_embed
        pos_embed = self.get_pos_embedding(orig_input_shape, x, curr_pos_embed)
        
        if hasattr(self, 'decoder_input_proj'):
            x = self.decoder_input_proj(x)
            x = self.input_proj_norm(x)
            # adjust the batch size of input_trunk_embed to match x
            if input_trunk_embed.size(0) > x.size(0):
                pseudo_shape = [x.size(0), input_trunk_embed.size(0) // x.size(0)] + list(x.shape[1:])
                input_trunk_embed = input_trunk_embed.view(pseudo_shape).mean(dim=1)

            x = x + self.input_proj_norm(input_trunk_embed)
        x = x + pos_embed
        if self.pos_sum_embed_only:
            return x
        x = self.decoder_embed(x)
        interim = []
        for i, blk in enumerate(self.decoder_blocks):
            if use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if self.return_interim_layers or i == (len(self.decoder_blocks) - 1):
                # decoder trunk 的输出，去掉cls_token
                if i == (len(self.decoder_blocks) - 1) and self.first_patch_idx:
                    interim.append(x[:, 1:, ...])
                else:
                    interim.append(x)

        interim = [self.norm(el) for el in interim]
        if self.final_projection is not None:
            interim = [self.final_projection(el) for el in interim]
        # pdb.set_trace()
        if self.return_interim_layers:
            return interim
        return interim[-1]

