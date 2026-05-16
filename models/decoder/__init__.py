import torch
import torch.nn as nn

# from types import SimpleNamespace
from models.codebind_model import ModalityType
from models.transformer import Attention
from models.decoder.vision_transformer import Decoder
from models.decoder.modules import DecoderHead
import pdb



def build_decoder(cfg, modality_reconstruction=None):
    # --- 参数配置
    input_embed_dim = cfg.get("input_embed_dim")
    trunk_embed_dim = cfg.get("trunk_embed_dim")
    input_projection = cfg.get('input_projection', False)
    encoder_trunk_embed_dim = cfg.get("encoder_trunk_embed_dim") if input_projection else None
    # 由于decoder trunk和encoder trunk的 input_dim不同，而无法使用share_pos_embed
    # 此时加入first_projection 将input_dim统一后使用share_pos_embed
    # share_pos_embed = cfg.get('input_type') == 'encoder_trunk_feature'
    share_pos_embed = input_projection
    with_cls_token = cfg.get("with_cls_token", False)  # 将cls token 加入decoder inputs
    first_patch_idx = 1 if with_cls_token else 0

    decoder_trunks = {}
    decoder_heads = {}


    decoder_trunks[ModalityType.VISION] = Decoder(first_patch_idx=first_patch_idx,
                patches_layout=(1, 16, 16),   
                attn_target=Attention,      
                embed_dim=input_embed_dim.get(ModalityType.VISION),         
                decoder_embed_dim=trunk_embed_dim.get(ModalityType.VISION, 512),
                input_proj_dim = encoder_trunk_embed_dim.get(ModalityType.VISION) if encoder_trunk_embed_dim is not None else None,
                learnable_pos_embed=True,  
                share_pos_embed=share_pos_embed,
                decoder_depth=cfg.get("decoder_depth", 8)
                )
    decoder_trunks[ModalityType.DEPTH] = Decoder(first_patch_idx=first_patch_idx,
                        patches_layout=(14, 14),   
                        attn_target=Attention,    
                        embed_dim=input_embed_dim.get(ModalityType.DEPTH),   # 384           
                        decoder_embed_dim=trunk_embed_dim.get(ModalityType.DEPTH, 512),   
                        input_proj_dim = encoder_trunk_embed_dim.get(ModalityType.DEPTH) if encoder_trunk_embed_dim is not None else None, 
                        learnable_pos_embed=True,  
                        share_pos_embed=share_pos_embed,
                        decoder_depth=cfg.get("decoder_depth", 8),        # num of stacked attention blocks
                        )
    
    decoder_trunks[ModalityType.THERMAL] = Decoder(first_patch_idx=first_patch_idx,
                    patches_layout=(14, 14),   
                    attn_target=Attention,    
                    embed_dim=input_embed_dim.get(ModalityType.THERMAL),              
                    decoder_embed_dim=trunk_embed_dim.get(ModalityType.THERMAL, 512),
                    input_proj_dim = encoder_trunk_embed_dim.get(ModalityType.THERMAL) if encoder_trunk_embed_dim is not None else None,    
                    learnable_pos_embed=True,  
                    share_pos_embed=share_pos_embed,
                    decoder_depth=cfg.get("decoder_depth", 8),        # num of stacked attention blocks
                    )
    
    decoder_trunks[ModalityType.AUDIO] = Decoder(first_patch_idx=first_patch_idx,
                        patches_layout=(12, 19),   
                        attn_target=Attention,    
                        embed_dim=input_embed_dim.get(ModalityType.AUDIO),    # 768          
                        decoder_embed_dim=trunk_embed_dim.get(ModalityType.AUDIO, 512),
                        input_proj_dim = encoder_trunk_embed_dim.get(ModalityType.AUDIO) if encoder_trunk_embed_dim is not None else None,    
                        learnable_pos_embed=True,  
                        share_pos_embed=share_pos_embed,
                        decoder_depth=cfg.get("decoder_depth", 8),        # num of stacked attention blocks
                        )
    decoder_trunks[ModalityType.TACTILE] = Decoder(first_patch_idx=first_patch_idx,
                        patches_layout=(14, 14),   
                        attn_target=Attention,    
                        embed_dim=input_embed_dim.get(ModalityType.TACTILE, 384),   # 384           
                        decoder_embed_dim=trunk_embed_dim.get(ModalityType.TACTILE, 512),   
                        input_proj_dim = encoder_trunk_embed_dim.get(ModalityType.TACTILE) if encoder_trunk_embed_dim is not None else None, 
                        learnable_pos_embed=True,  
                        share_pos_embed=share_pos_embed,
                        decoder_depth=cfg.get("decoder_depth", 8),        # num of stacked attention blocks
                        )
    
    
    decoder_heads[ModalityType.VISION] = DecoderHead(kernel_size=(2,7,7), stride=(2, 7, 7), patches_layout=(1, 16, 16), scale=2, 
                                                    in_channels=trunk_embed_dim.get(ModalityType.VISION, 512), 
                                                    hidden_channels=128, out_channels=3)  
    
    decoder_heads[ModalityType.DEPTH] = DecoderHead(kernel_size=8, stride=8, patches_layout=(14, 14), scale=2, 
                                                     in_channels=trunk_embed_dim.get(ModalityType.DEPTH, 512), 
                                                     hidden_channels=128, out_channels=1)
    
    decoder_heads[ModalityType.THERMAL] = DecoderHead(kernel_size=8, stride=8, patches_layout=(14, 14), scale=2, 
                                                      in_channels=trunk_embed_dim.get(ModalityType.THERMAL, 512), 
                                                      hidden_channels=128, out_channels=1)
    
    decoder_heads[ModalityType.AUDIO] = DecoderHead(kernel_size=8, stride=5, patches_layout=(12, 19), scale=2, 
                                                    in_channels=trunk_embed_dim.get(ModalityType.AUDIO, 512), 
                                                    hidden_channels=128, out_channels=1)
    decoder_heads[ModalityType.TACTILE] = DecoderHead(kernel_size=8, stride=8, patches_layout=(14, 14), scale=2, 
                                                     in_channels=trunk_embed_dim.get(ModalityType.TACTILE, 512), 
                                                     hidden_channels=128, out_channels=3)

    decoder_trunks, decoder_heads = nn.ModuleDict(decoder_trunks), nn.ModuleDict(decoder_heads)
    # freeze decoder parameters
    for param in decoder_trunks.parameters():
        param.requires_grad = False
    for param in decoder_heads.parameters():
        param.requires_grad = False

    for m_name, modality_decoder_trunk in decoder_trunks.named_children():
        if m_name in modality_reconstruction:
            modality_decoder_trunk.requires_grad_(True)

    for m_name, modality_decoder_head in decoder_heads.named_children():
        if m_name in modality_reconstruction:
            modality_decoder_head.requires_grad_(True)
            
    return decoder_trunks, decoder_heads