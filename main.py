# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Main codes for CodeBind

import logging
import os
import datetime
from typing import Any
# log
try:
    import wandb
except ImportError:
    wandb = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    logging.warning("Matplotlib not installed. This is not needed if you run this script as --headless")

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch import loggers as pl_loggers

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from models import codebind_model
from models import lora as LoRA
from models.codebind_model import ModalityType, load_module, save_module
from application.emergent_zero_shot_classification import EmergentZeroShotClassifier, map_calculation
from application.cross_modality_retrieval import CrossModalityRetrieval
from application.load_sd import imagebind_huge_sd

from config import load_cfg
from datasets import get_dataset
from models.VQ import build_vq
from models.decoder import build_decoder


from util import save_image

import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, force=True)

import pdb
from icecream import ic

# Logging settings
LOG_ON_STEP = True
LOG_ON_EPOCH = True
SAVE_CKP_LORA = True  # True 则手动保存， False 则使用lightning callback 自动保存checkpoint

MODALITIES = ["vision", "text", "audio", "thermal", "depth", "imu", "tactile", "eeg"]
from datasets.data_transform_rgbd import inv_normalize

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)


class CodeBind(L.LightningModule):
    def __init__(self, args):
        super().__init__()

        # --- load args and set default valus
        self.args = args
        self.train_mode = args.get("train_mode", "lora") # train_mode in lora, fulltune, headtune
        self.checkpoint_postfix = args.get("checkpoint_postfix", "_best")
        self.checkpoint_dir = self.args.get("checkpoint_dir")
        self.log_img_dir = self.args.get("log_img_dir")

        self.modality_pair = args.modality_pair  # 经过vq的模态
        self.modality_eval = args.modality_eval  # 待评估的目标模态
        self.modality_train = [args.modality_train] if isinstance(args.modality_train, str) else args.modality_train # 训练encoder的模态
        if self.modality_pair == ['vision', 'text'] or self.modality_pair == ['text', 'vision']:
            self.modality_anchor, self.modality_nonanchor = ['vision'], 'text'
        else:
            self.modality_anchor = list(set(self.modality_pair) & set(['vision', 'text']))
            self.modality_nonanchor = list(set(self.modality_pair) - set(self.modality_anchor))[0]
        self.intra_anchor_align = args.get('intra_anchor_align', False) and 'vision' in self.modality_anchor and 'text' in self.modality_anchor
        self.inter_anchor_align = [args.get('inter_anchor_align', None)] if isinstance(args.get('inter_anchor_align', None), str) else args.get('inter_anchor_align', None)
        self.inter_anchor_align = self.inter_anchor_align if self.inter_anchor_align is not None else self.modality_anchor

        self.cfg_encoder = self.args.get('encoder')
        self.cfg_decoder = self.args.get('decoder')
    
        # set up datasets
        self.train_dataset, self.test_dataset = get_dataset(args)

        self.set_validation_metric(init=True)
        
        self.save_hyperparameters()

        self.init_network()        
     
        # ---
        self.ezs_classfier = None
        self.retrieval = None  
        self.best_acc = 0.
        self.best_epoch = 0
        self.flag_update_text_model = True if ModalityType.TEXT in self.modality_train else False
        self.flag_update_retrieval_model = True if self.modality_eval in self.modality_train else False
        print(f"self.flag_update_retrieval_model={self.flag_update_retrieval_model}")

        if self.args.get('load_checkpoint_dir') is not None:
            self.load_flag = self.load_checkpoint(postfix=self.checkpoint_postfix, checkpoint_dir=self.args.get('load_checkpoint_dir'))
            print(f"load_checkpoint all success = {self.load_flag} from {self.args.get('load_checkpoint_dir')}")

    def init_network(self):
        # --- Load full pretrained ImageBind model
        if self.args.get("sd_version", False):
            self.model, self.model_postprocessors = imagebind_huge_sd(pretrained=True, train_mode=self.train_mode, 
                                                                      modality_train=self.modality_train, cfg_encoder=self.cfg_encoder, 
                                                                      cfg_eeg=self.args.get('eeg_conf'))
        else:
            self.model, self.model_postprocessors = codebind_model.imagebind_huge(pretrained=True, train_mode=self.train_mode, 
                                                                                modality_train=self.modality_train, cfg_encoder=self.cfg_encoder, 
                                                                                cfg_eeg=self.args.get('eeg_conf'))
        logging.info(f"Enable {self.train_mode} for {self.modality_train}.")
        if self.train_mode == "lora":
            lora_layer_idxs = {}
            for modality_name in self.args.lora_modality_names:
                if modality_name in MODALITIES:
                    lora_layer_idxs[modality_name] = getattr(self.args, f'lora_layer_idxs_{modality_name}', None)
                    if not lora_layer_idxs[modality_name]:
                        lora_layer_idxs[modality_name] = getattr(self.args, f'lora_layer_idxs', None)
                else:
                    raise ValueError(f"Unknown modality name: {modality_name}")
            logging.info("lora_layer_idxs:", lora_layer_idxs)
            lora_module_mode = self.args.get("lora_module_mode", "attn_out")
            logging.info(f"LoRA module mode: {lora_module_mode}")

            # add lora weights in trunks
            lora_model = LoRA.apply_lora_modality_trunks(self.model.modality_trunks, 
                                                         rank=self.args.get("lora_rank", 4),
                                                         layer_idxs=lora_layer_idxs,
                                                         modality_names=self.args.lora_modality_names,
                                                         lora_module_mode=lora_module_mode)
            
            self.model.modality_trunks.update(lora_model)

        # Decoder for reconstruction
        self.use_decoder = self.cfg_decoder.get('use_decoder', True)
        self.modality_reconstruction = self.args.get("modality_reconstruction", [])
        self.modality_reconstruction = [self.modality_reconstruction] if isinstance(self.modality_reconstruction, str) else self.modality_reconstruction
        self.modality_reconstruction = list(set(self.modality_pair).intersection(set(self.modality_reconstruction)))
        if self.use_decoder:
            assert self.modality_reconstruction != [], f"reconstruction modality should be specified if decoder is used"
            self.decoder_trunks, self.decoder_heads = build_decoder(self.cfg_decoder, self.modality_reconstruction)
            logging.info(f"Using reconstruction decoder for modality: {', '.join(self.modality_reconstruction)}")
        else:
            logging.info("Using reconstruction decoder for modality: None")

        # multimodel vector quantiser
        self.use_vq = self.args.get("vector_quantise", None) and (self.args.get("vector_quantise", None) is not None)
        logging.info(f"Using vector quantise: {self.use_vq}")
        if self.use_vq:
            self.cfg_vq = self.args.get('vector_quantise')
            self.cfg_vq.update({'modality_pair': self.args.get('modality_pair')})
            # pdb.set_trace()
            self.modality_vq = build_vq(self.cfg_vq)  
        else:
            self.modality_vq = None 
    
    def set_validation_metric(self, init=False):
        if init:
            dataset_name = self.test_dataset[0].dataset_name if isinstance(self.test_dataset, list) else self.test_dataset.dataset_name
        else:
            dataset_name = self.trainer.val_dataloaders.dataset[0].dataset_name if isinstance(self.trainer.val_dataloaders.dataset, list) else self.trainer.val_dataloaders.dataset.dataset_name
        
        if dataset_name == 'asa':
            self.validation_metric = 'map'
        elif dataset_name in ['msrvtt', 'audiocaps', 'clotho']:
            self.validation_metric = 'recall'
        else:
            self.validation_metric = 'acc'
        print(f"set_validation_metric '{self.validation_metric}' for {dataset_name}")

    def configure_optimizers(self):
        parameter_list = [
            {"params": self.model.modality_preprocessors.parameters()},
            {"params": self.model.modality_trunks.parameters()},
            {"params": self.model.modality_heads.parameters()},
            {"params": self.model.modality_postprocessors.parameters()},
            ]
        lr_new = self.args.get('lr_new', self.args.lr)
        # if self.args.get("sd_version", False):
        #     parameter_list.append({"params": self.model.mvq_adapter.parameters(), "lr": lr_new})
        if self.cfg_encoder.get('add_new_output_head'):
            parameter_list.append({"params": self.model.modality_heads_mvq.parameters(), "lr": lr_new})
        if self.cfg_encoder.get('use_postprocessors_outencoder'):
            parameter_list.append({"params": self.model_postprocessors.parameters(), "lr": lr_new})
        if self.use_vq:
            parameter_list.append({"params": self.modality_vq.parameters()})
        if self.use_decoder:
            parameter_list.append({"params": self.decoder_trunks.parameters(), "lr": lr_new})
            parameter_list.append({"params": self.decoder_heads.parameters(), "lr": lr_new})

        optimizer = optim.AdamW(
            parameter_list, 
            lr=self.args.lr, 
            weight_decay=self.args.weight_decay, 
            betas=self.args.momentum_betas
            )
        
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.args.max_epochs, 
            eta_min=self.args.lr / 50
            )
        
        # load optimizer
        # try:
        #     optimizer.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"imagebind-optimizer{self.checkpoint_postfix}.pth"), map_location='cpu'))
        # except FileNotFoundError:
        #     logging.warning(f"Could not load parameters for optimizer from {self.checkpoint_dir}.")

        return [optimizer], [lr_scheduler]
    
    def train_dataloader(self):
        if isinstance(self.train_dataset, list) and len(self.train_dataset) == 2:
            train_loader = DataLoader(
                self.train_dataset[self.current_epoch % 2],
                batch_size=self.args.batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=False,
                num_workers=self.args.num_workers)
        else:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=False,
                num_workers=self.args.num_workers,
            )
        self.train_loader = train_loader
        return train_loader
    
    def val_dataloader(self):
        if isinstance(self.test_dataset, list):
            val_loader = DataLoader(
                self.test_dataset[self.current_epoch % len(self.test_dataset)],
                batch_size=self.args.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=False,
                num_workers=self.args.num_workers)
        else:
            val_loader = DataLoader(
                self.test_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=False,
                num_workers=self.args.num_workers)
        self.val_loader = val_loader
        return val_loader
    
    def forward(self, batch, output_intermediate=False, modality_list=None):
        if modality_list is None:
            modality_list = self.modality_pair 
        in_dict = {}
        for class_i in modality_list:
            if batch.get(class_i) is not None:
                in_dict.update({class_i: batch.get(class_i)})
        out_dict = self.model(in_dict, output_intermediate)

        return out_dict

    def calculate_loss(self, out_dict, batch, batch_idx, mode):
        total_loss = 0
        # mode_vq = 'train' if mode == 'train' or self.args.get('val_loss', False) else 'val'
        if mode == 'train' or self.args.get('val_loss', False):
            mode_vq = 'train'
            if 'text' in self.modality_pair:
                vq_modality_list = [x for x in self.modality_pair if x != 'text']
                vq_modality_list.insert(0, 'text')
            else:
                vq_modality_list = self.modality_pair
        else:
            mode_vq = 'val'
            vq_modality_list = [self.modality_eval]

        feats_anchor = {modality_name: out_dict.get(modality_name) for modality_name in self.modality_anchor}
        feats_b = {self.modality_nonanchor: out_dict.get(self.modality_nonanchor)}

        if self.use_vq:
            vq_dict = {}
            if self.cfg_vq.get('vq_all_token'):  # mvq with cls_token and patch_tokens
                for modality_name in vq_modality_list:
                    if modality_name == 'text':
                        vq_dict_unimodal = self.modality_vq(out_dict.get(modality_name), modality_name, mode=mode_vq)
                    else:
                        vq_dict_unimodal = self.modality_vq(out_dict.get(modality_name+'_all_tokens'), modality_name, mode=mode_vq)
                    vq_dict.update({modality_name: vq_dict_unimodal})
                    out_dict.update({modality_name+'_vq': vq_dict_unimodal.get('concat')})
                    # get common feature for alignment    text 仅有cls_token, 无patch_token
                    if modality_name in self.modality_anchor:
                        feats_a = vq_dict_unimodal.get('common')[:, 0, ...] if modality_name != 'text' else vq_dict_unimodal.get('common')
                        feats_anchor.update({modality_name: feats_a})
                    else:
                        feats_b[modality_name] = vq_dict_unimodal.get('common')[:, 0, ...] if modality_name != 'text' else vq_dict_unimodal.get('common')

            else:  # mvq with only cls_token
                for modality_name in vq_modality_list:
                    vq_dict_unimodal = self.modality_vq(out_dict.get(modality_name), modality_name, mode=mode_vq)
                    vq_dict.update({modality_name: vq_dict_unimodal})
                    if modality_name in self.modality_anchor:
                        feats_anchor.update({modality_name: vq_dict_unimodal.get('common')})
                    else:
                        feats_b[modality_name] = vq_dict_unimodal.get('common')
            # if self.cfg_encoder.get('add_new_output_head'):
            ### same as above ###
            # else:
            #     if self.use_decoder and self.cfg_vq.get('vq_all_token'):  # single vq with cls_token and patch_tokens
            #         for modality_name in vq_modality_list:
            #             vq_dict_unimodal = self.modality_vq(out_dict.get(modality_name+'_all_tokens'))
            #             vq_dict.update({modality_name: vq_dict_unimodal})
            #             out_dict.update({modality_name+'_vq': vq_dict_unimodal.get('q')})
            #             if modality_name in self.modality_anchor:
            #                 feats_a = vq_dict_unimodal.get('q')[:, 0, ...] if modality_name != 'text' else vq_dict_unimodal.get('q')
            #                 feats_anchor.update({modality_name: feats_a})
            #             else:
            #                 feats_b[modality_name] = vq_dict_unimodal.get('q')[:, 0, ...] if modality_name != 'text' else vq_dict_unimodal.get('q')

            #     else:  # single vq with only cls_token
            #         for modality_name in vq_modality_list:
            #             vq_dict_unimodal = self.modality_vq(out_dict.get(modality_name))
            #             vq_dict.update({modality_name: vq_dict_unimodal})
            #             if modality_name in self.modality_anchor:
            #                 feats_anchor.update({modality_name: vq_dict_unimodal.get('q')})
            #             else:
            #                 feats_b[modality_name] = vq_dict_unimodal.get('q')

            loss_vq = self.vq_loss(vq_dict, mode=mode)
            total_loss += loss_vq

        elif self.cfg_encoder.get('add_new_output_head'):
            common_dim = self.cfg_encoder.get('output_embed_dim').get('common')
            feats_anchor = {key: value[:, :common_dim] for key, value in feats_anchor.items()}  
            feats_b = {key: value[:, :common_dim] for key, value in feats_b.items()}  

        dual_nll = self.info_nce_loss(feats_anchor, feats_b, mode=mode)
        total_loss += dual_nll

        if self.use_decoder:
            loss_recons = self.rec_loss(out_dict, batch, batch_idx, mode=mode)
            total_loss += loss_recons


        return total_loss
    

    def info_nce_loss(self, feats_anchor, feats_b, mode="train"):

        if mode == "val" and self.args.get('val_loss', False) is False:
            return 0
        
        # apply modality postprocessors if use it outside the encoder
        if self.cfg_encoder.get('use_postprocessors_outencoder', False):
            assert self.model_postprocessors is not None
            for modality_name, feats in feats_anchor.items():
                # feats = F.normalize(feats, dim=-1)
                feats_anchor.update({modality_name: self.model_postprocessors[modality_name](feats)})
            for modality_name, feats in feats_b.items():
                # feats = F.normalize(feats, dim=-1)
                feats_b.update({modality_name: self.model_postprocessors[modality_name](feats)})
        
        if self.args.self_contrast and 'vision' in self.inter_anchor_align:  # only modality vision has self_contrast
            # TODO # both modality 'a' and 'b' have self_contrast
            feats_tensors = [feats_anchor.get('vision').chunk(2)] + \
                            [(feats_a.chunk(2)[0], feats_b[self.modality_nonanchor]) for modalname, feats_a in feats_anchor.items() if modalname in self.inter_anchor_align]
            temperatures = [1] + [self.args.temperature for _ in range(len(self.inter_anchor_align))]
            contrast = ["self"] + [f"cross_{modality_name}" for modality_name in self.inter_anchor_align]
        else:
            feats_tensors = [(feats_a, feats_b[self.modality_nonanchor]) for modalname, feats_a in feats_anchor.items() if modalname in self.inter_anchor_align]
            temperatures = [self.args.temperature for _ in range(len(self.inter_anchor_align))]
            contrast = [f"cross_{modality_name}" for modality_name in self.inter_anchor_align]
        
        # intra anchor alignment when vision and text are all anchor modalities
        if self.intra_anchor_align:
            feats_tensors += [(feats_anchor.get('vision'), feats_anchor.get('text'))]
            temperatures += [self.args.temperature]
            contrast += ["intra"]

        # Accumulate self-contrastive loss for image and its augmentation, and modailty with image
        total_nll = 0
        for feats_idx, (_feats_a_tensor, _feats_b_tensor) in enumerate(feats_tensors):
            cos_sim = F.cosine_similarity(_feats_a_tensor[:, None, :], _feats_b_tensor[None, :, :], dim=-1)
            cos_sim = cos_sim / temperatures[feats_idx]
            nll_col = -torch.log(torch.diag(torch.softmax(cos_sim, dim=0)))
            nll_row = -torch.log(torch.diag(torch.softmax(cos_sim, dim=1)))
            nll = (nll_col + nll_row).mean()

            total_nll += nll

            # Logging loss 
            flag_acc = False if mode == 'train' else True  # 训练时，acc_top1基本都为1，因此关闭以减少训练时间
            if flag_acc and "cross" in contrast[feats_idx]:
                # Get ranking position of positive example
                pos_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
                comb_sim = torch.cat(
                    [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position (first column) is positive example
                    dim=1,
                )
                sim_argsort_1 = comb_sim.argsort(dim=1, descending=True).argmin(dim=1) # find the similarity rank of positive example within the row of similarity matrix
                comb_sim = torch.cat(
                    [cos_sim[pos_mask][None, :], cos_sim.masked_fill(pos_mask, -9e15)],  # First position (first row) is positive example
                    dim=0,
                )
                sim_argsort_2 = comb_sim.argsort(dim=0, descending=True).argmin(dim=0) # find the similarity rank of positive example within the column of similarity matrix

                # Logging ranking metrics
                self.log(mode + contrast[feats_idx] + "_acc_top1", ((sim_argsort_1 == 0).float().mean() + (sim_argsort_2 == 0).float().mean()) / 2,
                     prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,
                     sync_dist=True)
                self.log(mode + contrast[feats_idx] + "_acc_top5", ((sim_argsort_1 < 5).float().mean() + (sim_argsort_2 < 5).float().mean()) / 2,
                     prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,
                     sync_dist=True)
        total_nll /= len(contrast)
        self.log(mode + "_loss_cross", total_nll, prog_bar=True,
            on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,
            sync_dist=True,  # sync logging across all GPU workers,
            #  It is recommended to use 'sync_dist=True' when logging on epoch level in distributed setting to accumulate the metric across devices.
            )
        return total_nll
    
    def cmcm_loss(self, vq_dict):

        loss_cmcm = 0
        modality_pair_list = [(modal_anchor, self.modality_nonanchor) for modal_anchor in self.modality_anchor]
        if self.intra_anchor_align:
            modality_pair_list += [('vision', 'text')]
        for (modal_a, modal_b) in modality_pair_list:
            dist_a = vq_dict.get(modal_a).get('res_common').get('dist')
            dist_b = vq_dict.get(modal_b).get('res_common').get('dist') # shape: vq_all_token: [bs, token_num, orig_feature_dim/common_embed_dim, common_embed_num]
                                                                        # only cls token: [bs, orig_feature_dim/common_embed_dim, common_embed_num]
        
            # 受限于显存，仅对cls_token 计算cmcm loss, 若注释，则对所有token的平均计算cmcm loss
            dist_a = dist_a[:, 0, ...] if dist_a.dim() == 4 else dist_a
            dist_b = dist_b[:, 0, ...] if dist_b.dim() == 4 else dist_b
            # dist.dim() == 4: [bs, token_num, feature_num, embed_num] -> [token_num, bs, feature_num, embed_num]
            # dist.dim() == 3: [bs, feature_num, embed_num] -> [bs, feature_num, embed_num]
            dist_a = dist_a.permute(1,0,2,3) if dist_a.dim() == 4 else dist_a
            dist_b = dist_b.permute(1,0,2,3) if dist_b.dim() == 4 else dist_b
            
            # cmcm loss to align codebook across modalities
            # probability distribution of each code measured by the distance of the code to every feature
            if torch.any(dist_a < 0) or torch.any(dist_b < 0):
                code_prob_a = F.softmax(dist_a, dim=-1)  # [bs, feature_num, embed_num] or [token_num, bs, feature_num, embed_num]
                code_prob_b = F.softmax(dist_b, dim=-1)  
            else:
                code_prob_a = F.softmax(torch.sqrt(dist_a), dim=1) 
                code_prob_b = F.softmax(torch.sqrt(dist_b), dim=1)

            # average for cls+patch tokens code prob
            code_prob_a = code_prob_a.mean(0) if code_prob_a.dim() == 4 else code_prob_a
            code_prob_b = code_prob_b.mean(0) if code_prob_b.dim() == 4 else code_prob_b

            # [bs, feature_num, embed_num] * 2 -> [feature_num, bs, embed_num] * 2 -> [feature_num, bs, bs]
            code_prob_a = code_prob_a.permute(1,0,2)
            code_prob_b = code_prob_b.permute(1,0,2)
            code_similarity = torch.einsum('imd,ind->imn', code_prob_a, torch.log(code_prob_b + 1e-10)) + torch.einsum('imd,ind->imn', torch.log(code_prob_a + 1e-10), code_prob_b)
            code_similarity = code_similarity + torch.max(-code_similarity)
            target_idx =  torch.range(0, code_similarity.size(1)-1, dtype=torch.long, device=code_similarity.device).repeat(code_similarity.size(0))
            loss_cmcm += F.cross_entropy(code_similarity.view(-1, code_similarity.size(-1)), target_idx)

        loss_cmcm /= len(modality_pair_list)
        return loss_cmcm

    
    def modal_decomp_loss(self, vq_dict):
        # pdb.set_trace()
        loss_modal_decomp = 0
        cnt = 0
        # vq_specific = []
        for modality_name, vq_dict_unimodal in vq_dict.items():
            common_unimodal = vq_dict_unimodal.get('common')
            specific_unimodal = vq_dict_unimodal.get('specific')
            if modality_name != 'text':
                # perform normalization
                if not self.cfg_vq.get('norm_code', False):
                    common_unimodal = F.normalize(common_unimodal, dim=-1)
                    specific_unimodal = F.normalize(specific_unimodal, dim=-1)
                # vq_specific.append(specific_unimodal)
                loss_modal_decomp += torch.norm(torch.einsum('bnd, bmd->bnm', common_unimodal, specific_unimodal), dim=(1,2), p='fro').mean()
                cnt += 1
        # if len(vq_specific) == 2:
        #     loss_modal_decomp += torch.norm(torch.einsum('bnd, bmd->bnm', vq_specific[0], vq_specific[1]), dim=(1,2), p='fro').mean()
        #     cnt += 1
        loss_modal_decomp /= cnt

        return loss_modal_decomp
    
    def uniform_loss(self, vq_dict):
        loss_uniform = 0
        cnt = 0
        for modality_name, vq_dict_unimodal in vq_dict.items():
            specific_unimodal = vq_dict_unimodal.get('specific')
            if modality_name != 'text':
                # only apply to cls token
                specific_unimodal = specific_unimodal[:,0,...]
                # perform normalization
                if not self.cfg_vq.get('norm_code', False):
                    specific_unimodal = F.normalize(specific_unimodal, dim=-1)
                # l2 distances among specific embeds
                loss_uniform += torch.pdist(specific_unimodal,p=2).pow(2).mul(-1).exp().mean().log()
                cnt += 1
            
        loss_uniform /= cnt

        return loss_uniform
        

        
    def vq_loss(self, vq_dict, mode="train"):

        if mode == "val" and self.args.get('val_loss', False) is False:
            return 0
        
        total_loss_vq = 0
        loss_vq, loss_vq_contra, loss_vq_uni = 0, 0, 0
        for vq_dict_unimodal in vq_dict.values():
            loss_vq += vq_dict_unimodal.get('loss_codebook')
            loss_vq_contra += vq_dict_unimodal.get('loss_contra')
            loss_vq_uni += vq_dict_unimodal.get('loss_uni')
        total_loss_vq += self.args.get('loss_vq_weight', 1.0) * loss_vq
        total_loss_vq += self.args.get('loss_vq_reg_weight', 1.0) * (loss_vq_contra + loss_vq_uni)

        self.log(mode + "_loss_vq", loss_vq, prog_bar=True, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, sync_dist=True)
        if self.args.get('loss_vq_reg_weight', 1.0) != 0.0:
            self.log(mode + "_loss_vq_contra", loss_vq_contra, prog_bar=True, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, sync_dist=True)
            self.log(mode + "_loss_vq_uni", loss_vq_uni, prog_bar=True, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, sync_dist=True)
        
        if self.args.get('loss_cmcm_weight', 1.0) != 0.0:
            loss_cmcm = self.cmcm_loss(vq_dict)
            total_loss_vq += self.args.get('loss_cmcm_weight', 1.0) * loss_cmcm
            self.log(mode + "_loss_cmcm", loss_cmcm, prog_bar=True, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, sync_dist=True)
        
        if self.args.get('loss_modal_decomp_weight', 1.0) != 0.0 and self.cfg_vq.get('codebook_num') > 1:
            loss_modal_decomp = self.modal_decomp_loss(vq_dict)
            total_loss_vq += self.args.get('loss_modal_decomp_weight', 1.0) * loss_modal_decomp
            self.log(mode + "_loss_modal_decomp", loss_modal_decomp, prog_bar=True, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, sync_dist=True)

        if self.args.get('loss_uniform_weight', 1.0) != 0.0 and self.cfg_vq.get('codebook_num') > 1:
            loss_uniform = self.uniform_loss(vq_dict)
            total_loss_vq += self.args.get('loss_uniform_weight', 1.0) * loss_uniform
            self.log(mode + "_loss_uniform", loss_uniform, prog_bar=True, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, sync_dist=True)
        return total_loss_vq

    def rec_loss(self, encoder_output, batch, batch_idx, mode='train'):
        if mode == "val" and (not self.args.get('val_loss', False)) and batch_idx != 0:
            return 0
        
        loss_piexl_recons = 0
        for class_decoder in self.modality_reconstruction:
            # class_encoder = 'vision' if class_decoder in ('image', 'video') else class_decoder    
            ori_data = batch.get(class_decoder)
            # encoder_trunk_input = encoder_output.get(class_encoder+"_trunk_input")
            encoder_trunk_output = encoder_output.get(class_decoder+"_trunk_output")
            encoder_pos_embed = encoder_output.get(class_decoder+"_pos_embed")
            # 若有经过VQ的数据，则优先使用；没有则用encoder最终的输出
            if encoder_output.get(class_decoder+"_vq") is not None:
                encoder_embedding = encoder_output.get(class_decoder+"_vq")
                # if not self.cfg_vq.get('norm_code', False):
                #     encoder_embedding = F.normalize(encoder_embedding, dim=-1)
            else:
                encoder_embedding = encoder_output.get(class_decoder+"_all_tokens")

            decoder_input = encoder_embedding if self.cfg_decoder.get('with_cls_token') else encoder_embedding[:, 1:, ...]
            # encoder_pos_embed = None  # encoder trunk 和 decoder trunk 的 input_dim 不同

            rec_token = self.decoder_trunks[class_decoder](
                decoder_input,
                orig_input_shape=None, 
                input_pos_embed=encoder_pos_embed, 
                input_trunk_embed=encoder_trunk_output,
                use_checkpoint=False,
                )
            # pdb.set_trace()
            # ori_token = encoder_trunk_input[:, 1:, ...].transpose(1, 2)
            rec_token = rec_token.transpose(1, 2)  # channel first: [bs, token_num, token_dim] --> [bs, token_dim, token_num].  [bs, num=256, dim=512]
            rec_tensor, ori_tensor = self.decoder_heads[class_decoder](rec_token, ori_data)  # rec_img_h = ori_img_h / 2
      
            # -- pixel-level reconstruction loss
            if mode == 'train' or (mode == 'val' and self.args.get('val_loss', False)):
                class_decoder = 'vision' if class_decoder in ['image', 'video'] else class_decoder
                loss_piexl_recons_x = F.mse_loss(rec_tensor, ori_tensor)      
                self.log(mode + "_loss_piexl_recons_"+class_decoder, loss_piexl_recons_x, prog_bar=True, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, sync_dist=True)
                
                loss_piexl_recons += loss_piexl_recons_x
                
                # -- feature-level reconstruction loss
                # loss_feature_recons_x = F.mse_loss(rec_token_x, ori_token_x)  
                # self.log(mode + "_loss_feature_recons_"+class_x, loss_feature_recons_x, prog_bar=True,
                #          on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.args.batch_size,
                #          sync_dist=True,  # sync logging across all GPU workers,
                #          #  It is recommended to use 'sync_dist=True' when logging on epoch level in distributed setting to accumulate the metric across devices.
                #          )

            # visualize reconstruction results for some data in the first batch in both train and validation mode
            if batch_idx == 0:  # if mode == "val" and batch_idx == 0:
                if ori_tensor.ndim == 5:  # vision: only visualise frame 0 in video data
                    bs_ori = inv_normalize(class_decoder, ori_tensor[:, :, 0, ...])  # value_range=(0, 1)
                    bs_rec = inv_normalize(class_decoder, rec_tensor[:, :, 0, ...].clone().detach())
                else:
                    bs_ori = inv_normalize(class_decoder, ori_tensor)  # value_range=(0, 1)
                    bs_rec = inv_normalize(class_decoder, rec_tensor.clone().detach())
                max_images = min(bs_ori.size(0), 8)
                img_pair = torch.cat((bs_ori[0:max_images,...], bs_rec[0:max_images,...]), dim=0)
                # log image to disk
                input_grid = make_grid(img_pair, nrow=max_images, value_range=(0, 1), normalize=True)  # range -> value_range
                _save_img_dir = os.path.join(self.log_img_dir, mode)
                os.makedirs(_save_img_dir, exist_ok=True)
                log_img_path = os.path.join(_save_img_dir, f'ori_rec_{class_decoder}_epoch{self.current_epoch}.jpg')
                save_image(255 * input_grid.cpu().numpy(), image_path=log_img_path)
                # log image to wandb
                # if mode == "val" and self.args.get("loggers", False):
                #     images = wandb.Image(img_pair, caption="Top: Input, Bottom: Output")
                #     _key = f"reconstruction - {class_decoder}"
                #     wandb.log({_key: images})

        return loss_piexl_recons # loss_feature_recons_x + 
    
    def on_train_epoch_start(self) -> None:
        # self.batch_size = self.trainer.train_dataloader.batch_size
        # current_lr = [param_group['lr'] for param_group in self.optimizers().optimizer.param_groups]
        # self.log("lr_new", current_lr[-1], prog_bar=True,
        #                 on_step=False, on_epoch=LOG_ON_EPOCH, batch_size=self.args.batch_size,
        #                 sync_dist=True,
        #                 )
        # self.log("lr", current_lr[0], prog_bar=True,
        #         on_step=False, on_epoch=LOG_ON_EPOCH, batch_size=self.args.batch_size,
        #         sync_dist=True, 
        #         )
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        # mode = "train"
        # total_loss = 0

        output_intermediate = True # if self.use_decoder else False
        res_dict = self.forward(batch, output_intermediate)
        total_loss = self.calculate_loss(res_dict, batch, batch_idx, mode="train")

        return total_loss
        
    def on_train_epoch_end(self):
        # TODO [1] only run on rank 0, [2] move to a more sutible place. ----
        # self.save_checkpoint(postfix="_last")
        pass


    def validation_step(self, batch, batch_idx):
        
        if self.validation_metric == 'map':
            gt_name = batch.get('label')
            data_b = batch.get(self.modality_eval)
            class_b = self.modality_eval
            audio_outputs, labels = self.ezs_classfier.get_batch_classification_map(data_b, class_b, gt_name, vector_quantiser=self.modality_vq)
            self.validation_step_outputs['audio_outputs'].append(audio_outputs)
            self.validation_step_outputs['labels'].append(labels)

            self.number_of_examples += data_b.size(0)

        elif self.validation_metric == 'recall':
            gt_name = batch.get('retrieval_id')
            bs_recall = self.retrieval.get_batch_recall(batch.get('text'), 'text', gt_name, vector_quantiser=self.modality_vq)
            self.validation_step_outputs.append(bs_recall)

            self.number_of_examples += batch.get('text').size(0)
        else:
            gt_name = batch.get('label')
            data_b = batch.get(self.modality_eval)
            class_b = self.modality_eval
            # pdb.set_trace()
            bs_acc = self.ezs_classfier.get_batch_classification_acc(data_b, class_b, gt_name, vector_quantiser=self.modality_vq)
            self.validation_step_outputs.append(bs_acc)
            
            self.number_of_examples += data_b.size(0)

        if self.args.get('val_loss', False) or (self.use_decoder and batch_idx==0):
            # pdb.set_trace()
            output_intermediate = True # if self.use_decoder else False
            res_dict = self.forward(batch, output_intermediate)
            total_loss = self.calculate_loss(res_dict, batch, batch_idx, mode="val")

    def on_validation_epoch_start(self) -> None:
        self.set_validation_metric()
        # self.batch_size = self.trainer.val_dataloaders.batch_size
        if self.validation_metric == 'recall':
            if self.flag_update_retrieval_model or self.retrieval is None:
                print(f"Prepare Cross Madality Retrieval for {self.modality_eval} ...")
                self.retrieval = CrossModalityRetrieval(classifier_model=self, retrieval_modality_type=self.modality_eval)
        else:
            # 如果训练集中包含 TEXT，则 classifer 需要更新，否则不用更新
            if self.use_vq or self.flag_update_text_model or self.ezs_classfier is None:
                print(f"Prepare Emergent ZeroShot Classifier for {self.modality_eval} ... ")
                self.ezs_classfier = EmergentZeroShotClassifier(classifier_model=self, modality_type=self.modality_eval, text_template=self.args.get('text_template'))
                
        if self.validation_metric == 'map':
            self.validation_step_outputs = {'audio_outputs': [], 'labels': []}         
        else:
            self.validation_step_outputs = []
        self.number_of_examples = 0
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self):  
        # -- for data in each rank:
        
        # -- for data in all ranks:
        all_val_out = self.all_gather(self.validation_step_outputs)
        all_number_of_examples = self.all_gather(self.number_of_examples)

        if self.trainer.is_global_zero:
            # merge output and process
            if self.validation_metric == 'map':
                # use map as evaluation metric
                all_num = sum(all_number_of_examples)
                audio_outputs = []
                for batch_val_out in all_val_out['audio_outputs']:
                    if batch_val_out.dim() == 3:
                        audio_outputs.append(batch_val_out.reshape(-1, batch_val_out.size(-1)))
                    else:
                        assert batch_val_out.dim() == 2
                        audio_outputs.append(batch_val_out)
                audio_outputs = torch.cat(audio_outputs).to('cpu').detach()
                labels = []
                for batch_val_out in all_val_out['labels']:  # label: GT index of all classes 
                    if batch_val_out.dim() == 3:
                        labels.append(batch_val_out.reshape(-1, batch_val_out.size(-1)))
                    else:
                        assert batch_val_out.dim() == 2
                        labels.append(batch_val_out)
                labels = torch.cat(labels).to('cpu').detach()
                all_acc = [map_calculation(audio_outputs, labels)]  #  scale to list, to be compatible with other metrics

            else:
                # use accuracy or recall as evaluation metric
                all_num = sum(all_number_of_examples)   # list of int
                all_val_out = sum(all_val_out)          # list of list --> flattend list
                all_acc = sum(all_val_out) / all_num * 100.

            if self.best_acc < all_acc[0]:
                self.best_acc = all_acc[0]
                self.best_epoch = self.current_epoch
                if SAVE_CKP_LORA:
                    self.save_checkpoint(postfix='_best')
    
            logging.info(f"number_of_examples={all_num}, top 1 accuracy = {all_acc[0]:.2f}%, best_accuracy = {self.best_acc:.2f}% at epoch={self.best_epoch}")
            [print(f"Accuracy {i} {acc:.3f}%", end="  |") for i, acc in enumerate(all_acc)]

            self.log("val_acc", all_acc[0], on_epoch=LOG_ON_EPOCH, sync_dist=True, rank_zero_only=True)
            
        self.validation_step_outputs.clear()  # free memory
        self.trainer.strategy.barrier() # to let other cards to wait
        if SAVE_CKP_LORA:
            self.save_checkpoint(postfix='_last')
        
    def load_checkpoint(self, postfix='_best', checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir

        load_flag = []
        # Load existing preprocessors & heads & postprocessors
        load_flag.append(load_module(self.model.modality_preprocessors, modality_name=self.modality_pair,
                                        module_name="preprocessors", checkpoint_dir=checkpoint_dir, postfix=postfix))
        load_flag.append(load_module(self.model.modality_heads, modality_name=self.modality_pair,
                                module_name="heads", checkpoint_dir=checkpoint_dir, postfix=postfix))
        load_flag.append(load_module(self.model.modality_postprocessors, modality_name=self.modality_pair,
                                     module_name="postprocessors", checkpoint_dir=checkpoint_dir, postfix=postfix))
        if self.model_postprocessors is not None:
            load_flag.append(load_module(self.model_postprocessors, modality_name=self.modality_pair,
                                        module_name="postprocessors_new", checkpoint_dir=checkpoint_dir, postfix=postfix))

        if hasattr(self.model, 'modality_heads_mvq'):
            load_flag.append(load_module(self.model.modality_heads_mvq, modality_name=self.modality_pair,
                                         module_name="heads_mvq", checkpoint_dir=checkpoint_dir, postfix=postfix))
        if hasattr(self.model, 'mvq_adapter'):
            load_flag.append(load_module(self.model.mvq_adapter, modality_name=['vision', 'text'],
                                         module_name="mvq_adapter", checkpoint_dir=checkpoint_dir, postfix=postfix))
        if self.train_mode == "lora":
            # use fixed trunks with loaded lora parameters
            load_flag.append(LoRA.load_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=checkpoint_dir))
        elif self.train_mode == "fulltune":
            # Load existing trunks
            load_flag.append(load_module(self.model.modality_trunks, modality_name=self.modality_pair,
                                        module_name="trunks", checkpoint_dir=checkpoint_dir, postfix=postfix))
        
        # load codebook
        if self.use_vq:
            load_flag.append(load_module(self.modality_vq, module_name="vector_quantise", checkpoint_dir=checkpoint_dir, postfix=postfix))

        # load decoder
        if self.use_decoder and self.args.get('train', False):  # test 时，不需要 decoder
            load_flag.append(load_module(self.decoder_trunks, modality_name=self.modality_reconstruction,
                                         module_name="decoder_trunk", checkpoint_dir=checkpoint_dir, postfix=postfix))
            load_flag.append(load_module(self.decoder_heads, modality_name=self.modality_reconstruction,
                                         module_name="decoder_head", checkpoint_dir=checkpoint_dir, postfix=postfix))
        
        if all(load_flag):
            return True
        else:
            print(f"load_flag status = {load_flag}")
            return False

    def save_checkpoint(self, postfix='_best'):

        # Save preprocessors & heads & postprocessors
        save_module(self.model.modality_preprocessors, modality_name=self.modality_train,
                        module_name="preprocessors", checkpoint_dir=self.checkpoint_dir, postfix=postfix)
        save_module(self.model.modality_postprocessors, modality_name=self.modality_train,
                    module_name="postprocessors", checkpoint_dir=self.checkpoint_dir, postfix=postfix)
        save_module(self.model.modality_heads, modality_name=self.modality_train,
                    module_name="heads", checkpoint_dir=self.checkpoint_dir, postfix=postfix)
        if self.model_postprocessors is not None:
            save_module(self.model_postprocessors, modality_name=self.modality_train,
                        module_name="postprocessors_new", checkpoint_dir=self.checkpoint_dir, postfix=postfix)
        if hasattr(self.model, 'modality_heads_mvq'):
            save_module(self.model.modality_heads_mvq, modality_name=self.modality_train,
                        module_name="heads_mvq", checkpoint_dir=self.checkpoint_dir, postfix=postfix)
        if hasattr(self.model, 'mvq_adapter'):
            save_module(self.model.mvq_adapter, modality_name=['vision', 'text'],
                        module_name="mvq_adapter", checkpoint_dir=self.checkpoint_dir, postfix=postfix)

        if self.train_mode == "lora":
            LoRA.save_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=self.checkpoint_dir, postfix=postfix)
        elif self.train_mode == "fulltune":
            # Save preprocessors, trunks, heads, postprocessors in whole model fine-tuning situation
            save_module(self.model.modality_trunks, modality_name=self.modality_train,
                        module_name="trunks", checkpoint_dir=self.checkpoint_dir, postfix=postfix)
        
        # save Vector Quantise
        if self.use_vq:
            save_module(self.modality_vq, module_name="vector_quantise", checkpoint_dir=self.checkpoint_dir, postfix=postfix)
        
        # save decoder
        if self.use_decoder:
            save_module(self.decoder_trunks, modality_name=self.modality_reconstruction,  
                        module_name="decoder_trunk", checkpoint_dir=self.checkpoint_dir, postfix=postfix)
            save_module(self.decoder_heads, modality_name=self.modality_reconstruction, 
                        module_name="decoder_head", checkpoint_dir=self.checkpoint_dir, postfix=postfix)
        
        # save optimizer
        # try:
        #     torch.save(self.optimizers().optimizer.state_dict(), 
        #             os.path.join(self.checkpoint_dir, f"imagebind-optimizer{postfix}.pth"))
        #     logging.info(f"Saved parameters optimizer to {self.checkpoint_dir}.")
        # except FileNotFoundError:
        #     logging.warning(f"Could not save parameters for optimizer to {self.checkpoint_dir}.")
        

def set_logger(args):
    # Create logger
    # for logger in args.loggers if args.loggers is not None else []:
    logger = args.get("loggers", None)
    # pdb.set_trace()
    if logger == "wandb":
        save_dir = args.get('loggers_dir', f"./exp/{args.get('expname')}/log")
        os.makedirs(save_dir, exist_ok=True)
        wandb.init(
            project="imagebind",
            entity="image_bind", 
            name=args.get('expname'),  # display name for this run
            dir=save_dir,
            config=dict(args))
        wandb_logger = pl_loggers.WandbLogger(
            save_dir=save_dir,
            name="imagebind")
        logger = wandb_logger
    elif logger == "tensorboard":
        tensorboard_logger = pl_loggers.TensorBoardLogger(
            save_dir=args.loggers_dir,
            name="imagebind")
        logger = tensorboard_logger
    elif logger == "comet":
        comet_logger = pl_loggers.CometLogger(
            save_dir=args.loggers_dir,
            api_key=os.environ["COMET_API_KEY"],
            workspace=os.environ["COMET_WORKSPACE"],
            project_name=os.environ["COMET_PROJECT_NAME"],
            experiment_name=os.environ.get("COMET_EXPERIMENT_NAME", None),
        )
        logger = comet_logger
    elif logger == "mlflow":
        mlflow_logger = pl_loggers.MLFlowLogger(
            save_dir=args.loggers_dir,
            experiment_name=os.environ["MLFLOW_EXPERIMENT_NAME"],
            tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            run_name="imagebind"
        )
        logger = mlflow_logger
    
    print("Create logger:", args.get("loggers", None))
    return logger


def main(args):
    # Set experiment properties
    seed_everything(args.seed, workers=True)
    # torch.backends.cudnn.determinstic = True

    # train_dataset, test_dataset = get_dataset(args)

    # if train_dataset is not None:
    #     train_loader = DataLoader(
    #         train_dataset,
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #         drop_last=True,
    #         pin_memory=False,
    #         num_workers=args.num_workers,
    #     )
    # if test_dataset is not None:
    #     val_loader = DataLoader(
    #         test_dataset,
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         drop_last=False,
    #         pin_memory=False,
    #         num_workers=args.num_workers,
    #     )

    model = CodeBind(args=args)
    # load existing checkpoint
    if (not args.get('train', False)) and (not SAVE_CKP_LORA):
        save_checkpoint_dir = os.path.join(args.get('lightning_log', f"./exp/{args.get('expname')}/lightning_log"), "example_best.ckpt")
        if os.path.exists(save_checkpoint_dir):
            model = CodeBind.load_from_checkpoint(save_checkpoint_dir)
            
        
    
    # =======================
    # train(args, model)
    if args.get('train', False):
        logging.info("# # # # # ==> Running training progress ...")        
        train(args, model)
    else:
        logging.info("# # # # # ==> Running testing progress ...")
        test(args, model)
        

def train(args, model):
    device_name = args.device  # "cuda:0" if torch.cuda.is_available() else "cpu"
    if "cpu" == device_name[0]:
        devices = 1
        accelerator = "cpu"
    else:
        devices = [int(device_name.split(":")[1])] if isinstance(device_name, str) else [int(i.split(":")[1]) for i in device_name] 
        devices.sort()
        accelerator = "gpu"
    print("devices =", devices)

    # whether to reload dataloaders between epochs based on number of validation datasets
    if isinstance(model.test_dataset, list):
        reload_dataloaders_epoch_num = 1
    else:
        reload_dataloaders_epoch_num = 0

    logger = set_logger(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.get('lightning_log', f"./exp/{args.get('expname')}/lightning_log"),
        filename='example_best',
        save_top_k=1,
        verbose=True,
        monitor='val_acc',
        mode='max',
    )

    logging.info("Prepare trainer")
    trainer = Trainer(accelerator=accelerator,
                      devices=devices,
                      strategy='ddp_find_unused_parameters_true',
                      deterministic=True,
                      max_epochs=args.max_epochs, 
                      gradient_clip_val=args.gradient_clip_val, #   callbacks=[lr_monitor]
                      check_val_every_n_epoch=1,   # validation frequency
                      reload_dataloaders_every_n_epochs=reload_dataloaders_epoch_num,
                      limit_train_batches=args.get('limit_train_batches', None),  # debug, run through only 10% of the training set each epoch
                      limit_val_batches=args.get('limit_val_batches', None),
                      logger=logger if logger else None, 
                      default_root_dir=args.get('lightning_log', f"./exp/{args.get('expname')}/lightning_log"),
                      enable_checkpointing=False if SAVE_CKP_LORA else True,
                      num_sanity_val_steps=0,
                      callbacks=[checkpoint_callback] if not SAVE_CKP_LORA else None
                    #   **checkpointing
                      )
    

    # TODO Resume : continue training from checkpoint
    # 需要记录 optimizer, epoch, learning rate
    # automatically restores model, epoch, step, LR schedulers, etc...
    # ckpt_path = None  # "some/path/to/my_checkpoint.ckpt",  
    # if args.full_model_checkpoint_dir
    # model = MyLightingModule.load_from_checkpoint(PATH)
    # 覆盖ckpt文件中的超参数，in_dim=128, out_dim=10
    # model = LitModel.load_from_checkpoint(PATH, in_dim=128, out_dim=10)
    if args.get('train', False):
        logging.info("# # # # # ==> Running training progress ...")     
        # logging.info("Start fit")
        trainer.fit(model,
                    # ckpt_path=ckpt_path,
                    )
        if not SAVE_CKP_LORA:
            root_dir = args.get('lightning_log', f"./exp/{args.get('expname')}/lightning_log")
            trainer.save_checkpoint(os.path.join(root_dir, "example_last.ckpt"))

    else:
        logging.info("# # # # # ==> Running testing progress ...")
        # eval(model, val_loader)
        trainer.validate(model)


# @rank_zero_only
# def eval(model, val_loader):
#     # --------------- eval ---------------------
#     # 
#     if model.validation_metric == 'recall':
#         logging.info('Start evaluate Cross Modality Retrieval')
#         device = model.retrieval.device
#         model.to(device)    
#         model.freeze()     
#         retrieval = model.retrieval
#         retrieval.get_retrieval_recall(val_loader)
#     else:
#         logging.info('Start evaluate Emergent Zero-Shot Classifion')
#         device = model.ezs_classfier.device
#         model.to(device)    
#         model.freeze()                          
#         ezs_classfier = model.ezs_classfier

#         if model.validation_metric == 'acc':
#             ezs_classfier.get_classification_acc(val_loader)
#         else:
#             ezs_classfier.get_classification_map(val_loader)


def test(args, model):
    device_name = args.device
    if "cpu" == device_name[0]:
        devices = 1
    else:
        devices = [int(device_name.split(":")[1])] if isinstance(device_name, str) else [int(i.split(":")[1]) for i in device_name]
        devices.sort()
    print("devices =", devices)
    device = torch.device("cpu" if "cpu" == device_name[0] else f"cuda:{devices[0]}")

    # -----------------------------------
    logging.info('Start evaluate Emergent Zero-Shot Classification')
    model.val_dataloader()
    model.to(device)
    model.freeze()


    if model.validation_metric == 'recall':
        retrieval = CrossModalityRetrieval(classifier_model=model,
                                           retrieval_modality_type=args.modality_eval,
                                           retrieval_data=model.test_dataset.data_paths)
        retrieval.get_retrieval_recall(model.val_loader)
    else:
        ezs_classfier = EmergentZeroShotClassifier(classnames=model.test_dataset.class_names, 
                                                   classifier_model=model,
                                                   dataset_name=args.datasets,
                                                   modality_type=args.modality_eval,
                                                   text_template=args.get('text_template'))
        
        # if args.get("intramodal_classifier", False):
        #     # ezs_classfier.set_classifier(train_loader, intramodal_vector='common_and_specific')
        #     # ezs_classfier.set_classifier(val_loader, intramodal_vector='common')
        #     ezs_classfier.set_classifier(model.val_loader, intramodal_vector='specific')
        #     # ezs_classfier.set_classifier(val_loader, intramodal_vector='common_and_specific')

        if model.validation_metric == 'acc':
            ezs_classfier.get_classification_acc(model.val_loader)
        else:
            ezs_classfier.get_classification_map(model.val_loader)


if __name__ == "__main__":
    time_start = datetime.datetime.now()
    cfg_all = load_cfg()
    main(cfg_all)
    time_end=datetime.datetime.now()
    print('Running time: %s'%(time_end - time_start))
