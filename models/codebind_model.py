# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Model codes for CodeBind
# @Reference     : ImageBind, Meta Platforms, Inc. and affiliates.
import logging
import os
from functools import partial
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn as nn

from models.helpers import (EinOpsRearrange, LearnableLogitScaling, Normalize,
                            SelectElement, SelectEOSAndProject)
from models.multimodal_preprocessors import (AudioPreprocessor,
                                             IMUPreprocessor, PadIm2Video,
                                             PatchEmbedGeneric,
                                             RGBDTPreprocessor,
                                             SpatioTemporalPosEmbeddingHelper,
                                             TextPreprocessor,
                                             ThermalPreprocessor,
                                             TactilePreprocessor,
                                             EegPreprocessor)
from models.transformer import MultiheadAttention, SimpleTransformer

import pdb
from icecream import ic


ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
    TACTILE='tactile',
    EEG="eeg",
)

class CodeBindModel(nn.Module):
    def __init__(
        self,
        video_frames=2,
        kernel_size=(2, 14, 14),
        audio_kernel_size=16,
        audio_stride=10,
        out_embed_dim=768,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_num_mel_bins=128,
        audio_target_len=204,
        audio_drop_path=0.1,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        depth_embed_dim=384,
        depth_kernel_size=16,
        depth_num_blocks=12,
        depth_num_heads=8,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_kernel_size=8,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
        tactile_embed_dim=768,
        tactile_kernel_size=16,
        tactile_num_blocks=12,
        tactile_num_heads=12,
        tactile_drop_path=0.0,
        #
        eeg_embed_dim=768,
        eeg_data_len=512,
        eeg_patch_size=1,
        eeg_num_blocks=6,
        eeg_num_heads=6,
        eeg_drop_path=0.0,
        cfg_encoder=None,
    ):
        super().__init__()
        # 通过外部来设定
        # self.use_mvq = None  
        self.use_both_head = cfg_encoder.get("use_both_head", False)
        self.add_new_output_head = cfg_encoder.get("add_new_output_head", False)
        self.use_postprocessors_outencoder = cfg_encoder.get("use_postprocessors_outencoder", False)
        self.learnable_postprocessors = cfg_encoder.get("learnable_postprocessors", False)

        self.modality_preprocessors = self._create_modality_preprocessors(
            video_frames,
            vision_embed_dim,
            kernel_size,
            text_embed_dim,
            audio_embed_dim,
            audio_kernel_size,
            audio_stride,
            audio_num_mel_bins,
            audio_target_len,
            depth_embed_dim,
            depth_kernel_size,
            thermal_embed_dim,
            thermal_kernel_size,
            imu_embed_dim,
            tactile_embed_dim,
            tactile_kernel_size,
            eeg_embed_dim,
            eeg_data_len,
            eeg_patch_size,
        )

        self.modality_trunks = self._create_modality_trunks(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            audio_drop_path,
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            depth_drop_path,
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            thermal_drop_path,
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            imu_drop_path,
            tactile_embed_dim,
            tactile_num_blocks,
            tactile_num_heads,
            tactile_drop_path,
            # eeg
            eeg_embed_dim,
            eeg_num_blocks,
            eeg_num_heads,
            eeg_drop_path,
        )

        # modality_heads
        self.modality_heads = self._create_modality_heads(
                    out_embed_dim,
                    vision_embed_dim,
                    text_embed_dim,
                    audio_embed_dim,
                    depth_embed_dim,
                    thermal_embed_dim,
                    imu_embed_dim,
                    tactile_embed_dim,
                    eeg_embed_dim,
                    select_element_idx=-1 # bypass SelectElement to get feature for all tokens
                )
        if self.add_new_output_head:
            _embed_dim = cfg_encoder.get("output_embed_dim")
            self.modality_heads_mvq = self._create_modality_heads_mvq(
                vision_embed_dim,
                text_embed_dim,
                audio_embed_dim,
                depth_embed_dim,
                thermal_embed_dim,
                imu_embed_dim,
                tactile_embed_dim,
                eeg_embed_dim,
                common_embed_dim=0 if self.use_both_head else _embed_dim.get("common"),  # out_embed_dim
                specific_embed_dim_vision=_embed_dim.get("vision"),
                specific_embed_dim_text=_embed_dim.get("text", 0),
                specific_embed_dim_audio=_embed_dim.get("audio"),
                specific_embed_dim_depth=_embed_dim.get("depth"),
                specific_embed_dim_thermal=_embed_dim.get("thermal"),
                specific_embed_dim_imu=_embed_dim.get("imu"),
                specific_embed_dim_tactile=_embed_dim.get("tactile", 1024),
                specific_embed_dim_eeg=_embed_dim.get("eeg", 1024),
            )

        self.modality_postprocessors = self._create_modality_postprocessors(learnable=self.learnable_postprocessors)
        self.modality_embed_norm = Normalize(dim=-1)

        print("CodeBindModel init success.")

    def _create_modality_preprocessors(
        self,
        video_frames=2,
        vision_embed_dim=1024,
        kernel_size=(2, 14, 14),
        text_embed_dim=768,
        audio_embed_dim=768,
        audio_kernel_size=16,
        audio_stride=10,
        audio_num_mel_bins=128,
        audio_target_len=204,
        depth_embed_dim=768,
        depth_kernel_size=16,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        imu_embed_dim=512,
        tactile_embed_dim=768,
        tactile_kernel_size=16,
        eeg_embed_dim=768,
        eeg_data_len=16,
        eeg_patch_size=1,
    ):
        rgbt_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=2),
                nn.Conv3d(
                    in_channels=3,
                    kernel_size=kernel_size,
                    out_channels=vision_embed_dim,
                    stride=kernel_size,
                    bias=False,
                ),
            ]
        )
        rgbt_preprocessor = RGBDTPreprocessor(
            img_size=[3, video_frames, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=rgbt_stem,
            depth_stem=None,
        )

        text_preprocessor = TextPreprocessor(
            context_length=77,
            vocab_size=49408,
            embed_dim=text_embed_dim,
            causal_masking=True,
        )

        audio_stem = PatchEmbedGeneric(
            proj_stem=[
                nn.Conv2d(
                    in_channels=1,
                    kernel_size=audio_kernel_size,
                    stride=audio_stride,
                    out_channels=audio_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim),
        )
        audio_preprocessor = AudioPreprocessor(
            img_size=[1, audio_num_mel_bins, audio_target_len],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            audio_stem=audio_stem,
        )

        depth_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=depth_kernel_size,
                    in_channels=1,
                    out_channels=depth_embed_dim,
                    stride=depth_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=depth_embed_dim),
        )

        depth_preprocessor = RGBDTPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=None,
            depth_stem=depth_stem,
        )

        thermal_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=thermal_kernel_size,
                    in_channels=1,
                    out_channels=thermal_embed_dim,
                    stride=thermal_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=thermal_embed_dim),
        )
        thermal_preprocessor = ThermalPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            thermal_stem=thermal_stem,
        )

        imu_stem = PatchEmbedGeneric(
            [
                nn.Linear(
                    in_features=48,
                    out_features=imu_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=imu_embed_dim),
        )

        imu_preprocessor = IMUPreprocessor(
            img_size=[6, 2000],
            num_cls_tokens=1,
            kernel_size=8,
            embed_dim=imu_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            imu_stem=imu_stem,
        )


        tactile_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=tactile_kernel_size,
                    in_channels=3,
                    out_channels=tactile_embed_dim,
                    stride=tactile_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=tactile_embed_dim),
        )
        tactile_preprocessor = TactilePreprocessor(
            img_size=[3, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            tactile_stem=tactile_stem,
        )
        # pdb.set_trace()
        in_chans = 128
        # eeg_patch_size = 1
        num_patches = int(eeg_data_len / eeg_patch_size)
        eeg_stem = PatchEmbedGeneric(
            [
                nn.Conv1d(in_chans, eeg_embed_dim, kernel_size=eeg_patch_size, stride=eeg_patch_size),
                # nn.Linear(in_features=in_chans, out_features=eeg_embed_dim, bias=False),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=eeg_embed_dim),
        )

        eeg_preprocessor = EegPreprocessor(
            num_cls_tokens=1,
            num_patches=num_patches,
            embed_dim=eeg_embed_dim,
            # pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            eeg_stem=eeg_stem,
        )

        modality_preprocessors = {
            ModalityType.VISION: rgbt_preprocessor,
            ModalityType.TEXT: text_preprocessor,
            ModalityType.AUDIO: audio_preprocessor,
            ModalityType.DEPTH: depth_preprocessor,
            ModalityType.THERMAL: thermal_preprocessor,
            ModalityType.IMU: imu_preprocessor,
            ModalityType.TACTILE: tactile_preprocessor,
            ModalityType.EEG: eeg_preprocessor,
        }

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_trunks(
        self,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_drop_path=0.0,
        depth_embed_dim=768,
        depth_num_blocks=12,
        depth_num_heads=12,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
        tactile_embed_dim=768,
        tactile_num_blocks=12,
        tactile_num_heads=12,
        tactile_drop_path=0.0,
        eeg_embed_dim=768,
        eeg_num_blocks=6,
        eeg_num_heads=6,
        eeg_drop_path=0.0,
    ):
        def instantiate_trunk(
            embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        modality_trunks = {}
        modality_trunks[ModalityType.VISION] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.TEXT] = instantiate_trunk(
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=audio_drop_path,
        )
        modality_trunks[ModalityType.DEPTH] = instantiate_trunk(
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=depth_drop_path,
        )
        modality_trunks[ModalityType.THERMAL] = instantiate_trunk(
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=thermal_drop_path,
        )
        modality_trunks[ModalityType.IMU] = instantiate_trunk(
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=imu_drop_path,
        )
        modality_trunks[ModalityType.TACTILE] = instantiate_trunk(
            tactile_embed_dim,
            tactile_num_blocks,
            tactile_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=tactile_drop_path,
        )

        modality_trunks[ModalityType.EEG] = instantiate_trunk(
            eeg_embed_dim,
            eeg_num_blocks,
            eeg_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=eeg_drop_path,
        )

        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads(
        self,
        out_embed_dim,
        vision_embed_dim,
        text_embed_dim,
        audio_embed_dim,
        depth_embed_dim,
        thermal_embed_dim,
        imu_embed_dim,
        tactile_embed_dim,
        eeg_embed_dim,
        select_element_idx=0
    ):
        modality_heads = {}

        modality_heads[ModalityType.VISION] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=select_element_idx),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
            proj=nn.Sequential(
                nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                nn.Linear(text_embed_dim, out_embed_dim, bias=False),
            ), only_proj=False
        )

        modality_heads[ModalityType.AUDIO] = nn.Sequential(
            nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
            SelectElement(index=select_element_idx),
            nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.DEPTH] = nn.Sequential(
            nn.LayerNorm(normalized_shape=depth_embed_dim, eps=1e-6),
            SelectElement(index=select_element_idx),
            nn.Linear(depth_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.THERMAL] = nn.Sequential(
            nn.LayerNorm(normalized_shape=thermal_embed_dim, eps=1e-6),
            SelectElement(index=select_element_idx),
            nn.Linear(thermal_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.IMU] = nn.Sequential(
            nn.LayerNorm(normalized_shape=imu_embed_dim, eps=1e-6),
            SelectElement(index=select_element_idx),
            nn.Dropout(p=0.5),
            nn.Linear(imu_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.TACTILE] = nn.Sequential(
            nn.LayerNorm(normalized_shape=tactile_embed_dim, eps=1e-6),
            SelectElement(index=select_element_idx),
            nn.Linear(tactile_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.EEG] = nn.Sequential(
            nn.LayerNorm(normalized_shape=eeg_embed_dim, eps=1e-6),
            SelectElement(index=select_element_idx),
            nn.Linear(eeg_embed_dim, out_embed_dim, bias=False),
        )
        

        return nn.ModuleDict(modality_heads)

    def _create_modality_heads_mvq(
        self,
        vision_embed_dim,
        text_embed_dim,
        audio_embed_dim,
        depth_embed_dim,
        thermal_embed_dim,
        imu_embed_dim,
        tactile_embed_dim,
        eeg_embed_dim,
        common_embed_dim,
        specific_embed_dim_vision,
        specific_embed_dim_text,
        specific_embed_dim_audio,
        specific_embed_dim_depth,
        specific_embed_dim_thermal,
        specific_embed_dim_imu,
        specific_embed_dim_tactile,
        specific_embed_dim_eeg,
    ):
        modality_heads = {}

        modality_heads[ModalityType.VISION] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            # SelectElement(index=0),
            nn.Linear(vision_embed_dim, common_embed_dim + specific_embed_dim_vision, bias=False),
        )

        modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
            proj=nn.Sequential(
                nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                nn.Linear(text_embed_dim, common_embed_dim + specific_embed_dim_text, bias=False),
            ), only_proj=False
        )

        modality_heads[ModalityType.AUDIO] = nn.Sequential(
            nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
            # SelectElement(index=0),
            nn.Linear(audio_embed_dim, common_embed_dim + specific_embed_dim_audio, bias=False),
        )

        modality_heads[ModalityType.DEPTH] = nn.Sequential(
            nn.LayerNorm(normalized_shape=depth_embed_dim, eps=1e-6),
            # SelectElement(index=0),
            nn.Linear(depth_embed_dim, common_embed_dim + specific_embed_dim_depth, bias=False),
        )

        modality_heads[ModalityType.THERMAL] = nn.Sequential(
            nn.LayerNorm(normalized_shape=thermal_embed_dim, eps=1e-6),
            # SelectElement(index=0),
            nn.Linear(thermal_embed_dim, common_embed_dim + specific_embed_dim_thermal, bias=False),
        )

        modality_heads[ModalityType.IMU] = nn.Sequential(
            nn.LayerNorm(normalized_shape=imu_embed_dim, eps=1e-6),
            # SelectElement(index=0),
            nn.Dropout(p=0.5),
            nn.Linear(imu_embed_dim, common_embed_dim + specific_embed_dim_imu, bias=False),
        )

        modality_heads[ModalityType.TACTILE] = nn.Sequential(
            nn.LayerNorm(normalized_shape=tactile_embed_dim, eps=1e-6),
            # SelectElement(index=0),
            nn.Linear(tactile_embed_dim, common_embed_dim + specific_embed_dim_tactile, bias=False),
        )

        modality_heads[ModalityType.EEG] = nn.Sequential(
            nn.LayerNorm(normalized_shape=eeg_embed_dim, eps=1e-6),
            # SelectElement(index=0),
            nn.Linear(eeg_embed_dim, common_embed_dim + specific_embed_dim_eeg, bias=False),
        )

        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self, learnable=False):
        modality_postprocessors = {}

        modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
        )
        modality_postprocessors[ModalityType.AUDIO] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=20.0, learnable=learnable),
        )
        modality_postprocessors[ModalityType.DEPTH] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=learnable),
        )
        modality_postprocessors[ModalityType.THERMAL] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=10.0, learnable=learnable),
        )
        modality_postprocessors[ModalityType.IMU] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=learnable),
        )
        modality_postprocessors[ModalityType.TACTILE] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=learnable),
        ) 
        modality_postprocessors[ModalityType.EEG] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=learnable),
        )
        return nn.ModuleDict(modality_postprocessors)

    def forward(self, inputs, output_intermediate=False, skip_norm=False):
        # print("forward inputs:", type(inputs), len(inputs), list(inputs.keys()))  # dict
        # pdb.set_trace()
        outputs = {}
        for modality_key, modality_value in inputs.items():

            reduce_list = (
                modality_value.ndim >= 5
            )  # Audio and Video inputs consist of multiple clips
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(B * S, *modality_value.shape[2:])

            if modality_value is not None:
                # --- step 1: preprocessor
                modality_value = self.modality_preprocessors[modality_key](**{modality_key: modality_value})
                # 
                trunk_inputs = modality_value["trunk"]  # {'tokens', 'attn_mask'}
                head_inputs = modality_value["head"]    # {'seq_len'}
                pos_embeds = modality_value.get('pos_embeds')
                # vision: trunk_inputs['tokens'].shape = storch.Size([bs, 257, 1280])
                # depth:  trunk_inputs['tokens'].shape = storch.Size([bs, 197, 384])
                # audio:  trunk_inputs['tokens'].shape = storch.Size([bs, 229, 768])
                if output_intermediate:
                    # trunk input token: Embedded Patches and Extralearnable [class] embedding
                    outputs.update({modality_key + '_trunk_input': trunk_inputs['tokens']})   
                # --- step 2: trunk
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)
                if output_intermediate:
                    outputs.update({modality_key + '_trunk_output': modality_value})  # trunk output feature
                    outputs.update({modality_key + '_pos_embed': pos_embeds})    # pos_embeds

                # --- step 3: head
                if modality_key == "text" or not self.add_new_output_head:
                    modality_value = self.modality_heads[modality_key](modality_value, **head_inputs)  # [bs, dim]
                elif self.use_both_head: 
                    modality_value_common = self.modality_heads[modality_key](modality_value, **head_inputs) # [bs, token_num, dim]
                    modality_value_specific = self.modality_heads_mvq[modality_key](modality_value, **head_inputs)
                    modality_value = torch.cat((modality_value_common, modality_value_specific), dim=2)
                else:
                    modality_value = self.modality_heads_mvq[modality_key](modality_value, **head_inputs)

                # --- step 4: postprocessor
                if self.use_postprocessors_outencoder:
                    if skip_norm:
                        pass
                    else:
                        modality_value = self.modality_embed_norm(modality_value)
                else:
                    modality_value = self.modality_postprocessors[modality_key](modality_value)  # [bs, token_num, 1024+256]
                if reduce_list: 
                    bs_shape = [B, S] + list(modality_value.shape[1:])
                    modality_value = torch.reshape(modality_value, bs_shape)
                    modality_value = modality_value.mean(dim=1)

                # --- step 5: cls_token & patch_tokens
                if modality_value.dim() == 3: # get cls_token if modality_value has shape [bs, token_num, dim]
                    outputs[modality_key] = modality_value[:, 0, ...]  
                else:
                    outputs[modality_key] = modality_value              
                if output_intermediate:
                    outputs.update({modality_key + '_all_tokens': modality_value})  # cls_token and patch_tokens

        return outputs


def imagebind_huge(pretrained=False, train_mode="lora", modality_train=None, cfg_encoder=None, cfg_eeg=None):
    model = CodeBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
        # 
        eeg_embed_dim= cfg_encoder.get('trunk_embed_dim').get('eeg', 768), 
        eeg_data_len=512 if cfg_eeg is None else cfg_eeg["data_len"],
        eeg_patch_size=1 if cfg_eeg is None else cfg_eeg["patch_size"],
        cfg_encoder=cfg_encoder,
    )

    if pretrained:
        if not os.path.exists(".checkpoints/imagebind_huge.pth"):
            print(
                "Downloading imagebind weights to .checkpoints/imagebind_huge.pth ..."
            )
            os.makedirs(".checkpoints", exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                ".checkpoints/imagebind_huge.pth",
                progress=True,
            )


        model.load_state_dict(torch.load(".checkpoints/imagebind_huge.pth"), strict=False)



    
    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    if train_mode is not None:
        if train_mode in ("lora", "headtune"):
            # in lora mode, freeze all preprocessors and trunks and enable cls token training of train modality
            # enable fine-tuning for head and postprocessors of train modality
            for m_name, modality_preprocessor in model.modality_preprocessors.named_children():
                if m_name in modality_train and hasattr(modality_preprocessor, "cls_token"):
                    modality_preprocessor.cls_token.requires_grad = True
            
        else:
            # in full model fine-tuning, we enable preprocessors, trunks, heads, postprocessors of train modality
                for m_name, modality_preprocessor in model.modality_preprocessors.named_children():
                    if m_name in modality_train:
                        modality_preprocessor.requires_grad_(True)
                for m_name, modality_trunk in model.modality_trunks.named_children():
                    if m_name in modality_train:
                        modality_trunk.requires_grad_(True)

        for m_name, modality_head in model.modality_heads.named_children():
            if m_name in modality_train:
                modality_head.requires_grad_(True)
        if cfg_encoder.get("add_new_output_head", False):
            for m_name, modality_head_mvq in model.modality_heads_mvq.named_children():
                if m_name in modality_train:
                    modality_head_mvq.requires_grad_(True)
        if not cfg_encoder.get('use_postprocessors_outencoder', False):
            for m_name, modality_postprocessor in model.modality_postprocessors.named_children():
                if m_name in modality_train:
                    modality_postprocessor.requires_grad_(True)
    
    # build modality postprocessors with learnable logit scaling outside encoder (i.e. imagebind original model structure)
    if cfg_encoder.get("use_postprocessors_outencoder", False):
        
        def _create_modality_postprocessors(learnable=False):
            modality_postprocessors = {}

            modality_postprocessors[ModalityType.VISION] = nn.Identity()
            modality_postprocessors[ModalityType.TEXT] = LearnableLogitScaling(learnable=True)
            modality_postprocessors[ModalityType.AUDIO] = LearnableLogitScaling(logit_scale_init=20.0, learnable=learnable)
            modality_postprocessors[ModalityType.DEPTH] = LearnableLogitScaling(logit_scale_init=5.0, learnable=learnable)
            modality_postprocessors[ModalityType.THERMAL] = LearnableLogitScaling(logit_scale_init=10.0, learnable=learnable)
            modality_postprocessors[ModalityType.IMU] = LearnableLogitScaling(logit_scale_init=5.0, learnable=learnable)
            modality_postprocessors[ModalityType.TACTILE] = LearnableLogitScaling(logit_scale_init=5.0, learnable=learnable)
            modality_postprocessors[ModalityType.EEG] = LearnableLogitScaling(logit_scale_init=5.0, learnable=learnable)

            return nn.ModuleDict(modality_postprocessors)
        
        model_postprocessors = _create_modality_postprocessors(learnable=cfg_encoder.get("learnable_postprocessors", False))
        # load pretrained parameters for modality postprocessors in imagebind model
        for param_name, param in model.modality_postprocessors.text.named_parameters():
            corresponding_param = getattr(model_postprocessors.text, param_name.split('.')[-1])   
            corresponding_param.data.copy_(param.data)
        # freeze parameters in new modality postprocessors
        for param in model_postprocessors.parameters():
            param.requires_grad = False
        if train_mode is not None:
            for m_name, modality_postprocessor in model_postprocessors.named_children():
                if m_name in modality_train:
                    modality_postprocessor.requires_grad_(True)
    
    else:
        model_postprocessors = None
        
    return model, model_postprocessors


def save_module(module_dict: nn.ModuleDict, modality_name = None, module_name: str = "",
                   checkpoint_dir: str = "./.checkpoints", postfix: str = "_last",
                   extension: str = "pth"):
    if module_name == "vector_quantise":
        torch.save(module_dict.state_dict(), 
                   os.path.join(checkpoint_dir, f"imagebind-{module_name}-{postfix}.{extension}"))
        logging.info(f"Saved parameters for module {module_name} to {checkpoint_dir}.")
        return
    
    if modality_name is None:
        # save the whole module_dict
        try:
            torch.save(module_dict.state_dict(),
                       os.path.join(checkpoint_dir, f"imagebind-{module_name}{postfix}.{extension}"))
            logging.info(f"Saved parameters for module {module_name} to {checkpoint_dir}.")
        except FileNotFoundError:
            logging.warning(f"Could not save module parameters for {module_name} to {checkpoint_dir}.")
    else:
        # save part of module_dict based on modality name
        for sub_modality_name in modality_name:
            try:
                for sub_module_name, sub_module_dict in module_dict.items():
                    if sub_module_name == sub_modality_name:
                        torch.save(sub_module_dict.state_dict(),
                                os.path.join(checkpoint_dir, f"imagebind-{module_name}-{sub_modality_name}{postfix}.{extension}"))
                        logging.info(f"Saved parameters for module {module_name} in {sub_modality_name} to {checkpoint_dir}.")
            except FileNotFoundError:
                logging.warning(f"Could not save module parameters for {module_name} in {sub_modality_name} to {checkpoint_dir}.")

def load_module(module_dict: nn.ModuleDict, modality_name = None, module_name: str = "",
                checkpoint_dir: str = "./.checkpoints", postfix: str = "_last",
                extension: str = "pth"):
    success_flag = True
    if module_name == "vector_quantise":
        file_path = os.path.join(checkpoint_dir, f"imagebind-{module_name}-{postfix}.{extension}")
        try:
            module_dict.load_state_dict(
                torch.load(file_path, map_location='cpu'),
                strict=False)
            logging.info(f"Loaded [{postfix[1:]}] parameters for module {module_name} from {checkpoint_dir}.")
        except FileNotFoundError:
            logging.warning(f"Could not load module parameters for {module_name} from {checkpoint_dir}. File not found: {file_path}")
            success_flag = False
        return success_flag
    
    if modality_name is None:
        # load the whole module_dict
        file_path = os.path.join(checkpoint_dir, f"imagebind-{module_name}{postfix}.{extension}")
        try:
            module_dict.load_state_dict(
                torch.load(file_path, map_location='cpu'), strict=False)
            logging.info(f"Loaded [{postfix[1:]}] parameters for module {module_name} from {checkpoint_dir}.")
        except FileNotFoundError:
            logging.warning(f"Could not load module parameters for {module_name} from {checkpoint_dir}. File not found: {file_path}")
            success_flag = False
    else:
        # load part of module_dict based on modality name
        for sub_modality_name in modality_name:
            try:
                for sub_module_name, sub_module_dict in module_dict.items():
                    if sub_module_name == sub_modality_name:
                        file_path = os.path.join(checkpoint_dir, f"imagebind-{module_name}-{sub_modality_name}{postfix}.{extension}")
                        logging.info(f"Try to load {file_path}")
                        sub_module_dict.load_state_dict(
                            torch.load(file_path, map_location='cpu'), strict=True)
                        logging.info(f"Loaded [{postfix[1:]}] parameters for module {module_name} in {sub_modality_name} from {checkpoint_dir}.")
            except FileNotFoundError:
                logging.warning(f"Could not load module parameters for {module_name} in {sub_modality_name} from {checkpoint_dir}.")
                success_flag = False

    return success_flag