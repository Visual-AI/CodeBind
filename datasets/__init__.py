# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision import transforms
import pdb


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]
    

# def collate_fn(batch):
#   pass
#   return {
#       'vision': torch.stack([x.get('vision') for x in batch]),
#       'text': torch.stack([x.get('text') for x in batch]),
#     #   'label': [x.get('label') for x in batch]
# }



def get_dataset(args):

    train_datasets = []
    test_datasets = []

    arg_dataset_list = [args.datasets] if isinstance(args.datasets, str) else args.datasets
    print("arg_dataset_list:", arg_dataset_list)

    # Load datasets
    # validation dataloader
    _val_loss = args.get('val_loss', False)  # val_loss needs modality_pair
    _res_pair = False
    for i in args.get('modality_reconstruction', []):
        if i != args.modality_eval:
            _res_pair = True

    dataloader_output_modality = args.modality_pair if _val_loss or _res_pair else [args.modality_eval]

    if "imagenet" in arg_dataset_list:
        from datasets.imagenet import IN1KDataset
        train_datasets.append(IN1KDataset(root_dir=args.imagenet_root, dataset_name='imagenet', split="train", modality_pair=args.modality_pair,
                                          scale_factor=args.scale_factor, mode='train'))
        test_datasets.append(IN1KDataset(root_dir=args.imagenet_root, dataset_name='imagenet', split="test",
                                             modality_pair=dataloader_output_modality, mode='test'))
    if "places365" in arg_dataset_list:
        from datasets.places365 import Places365Dataset
        train_datasets.append(Places365Dataset(root_dir=args.places365_root, dataset_name='places365', split="train", modality_pair=args.modality_pair,
                                          scale_factor=args.scale_factor, mode='train'))
        test_datasets.append(Places365Dataset(root_dir=args.places365_root, dataset_name='places365', split="test",
                                             modality_pair=dataloader_output_modality, mode='test'))
        
    if 'msrvtt' in arg_dataset_list:
        from datasets.msrvtt import MSRVTTDataset
        train_datasets.append(MSRVTTDataset(root_dir=args.msrvtt_root, dataset_name='msrvtt', split="train", modality_pair=args.modality_pair,
                                          scale_factor=args.scale_factor, mode='train'))

        test_datasets.append(MSRVTTDataset(root_dir=args.msrvtt_root, dataset_name='msrvtt', split="test",
                                             modality_pair=dataloader_output_modality, mode='test'))
    
    if 'k400' in arg_dataset_list:
        from datasets.k400 import KineticsDataset
        train_datasets.append(KineticsDataset(root_dir=args.k400_root, dataset_name='k400', split="train", modality_pair=args.modality_pair,
                                          scale_factor=args.scale_factor, mode='train'))

        test_datasets.append(KineticsDataset(root_dir=args.k400_root, dataset_name='k400', split="test",
                                             modality_pair=dataloader_output_modality, mode='test'))
            
    if "nyu" in arg_dataset_list:
        from datasets.nyu import NYURGBD_Dataset
        args.self_contrast = False  # TODO : True not implemented
        train_datasets.append(NYURGBD_Dataset(root_dir=args.nyu_root, dataset_name='nyu', split="train", modality_pair=args.modality_pair,
                                              scale_factor=args.scale_factor, mode='train', text_template=args.get('text_template')))
        
        test_datasets.append(NYURGBD_Dataset(root_dir=args.nyu_root, dataset_name='nyu', split="test",
                                             modality_pair=dataloader_output_modality, # [args.modality_eval],  # - Only load the modality to eval
                                             mode='test', text_template=args.get('text_template')))

    
    if 'sun' in arg_dataset_list:
        from datasets.sun import SUNRGBD_Dataset
        args.self_contrast = False  # TODO : True not implemented
        print('data_version =', args.get('data_version', ''))
        train_datasets.append(SUNRGBD_Dataset(root_dir=args.sun_root, dataset_name='sun', split="train", modality_pair=args.modality_pair,
                                              scale_factor=args.scale_factor, mode='train', text_template=args.get('text_template'),
                                              data_version=args.get('data_version', '')
                                              ))

        test_datasets.append(SUNRGBD_Dataset(root_dir=args.sun_root, dataset_name='sun', split="test",
                                             modality_pair=dataloader_output_modality, mode='test', text_template=args.get('text_template'),
                                             data_version=args.get('data_version', '')
                                             ))


    if 'sun_evalnyu' in arg_dataset_list:
        from datasets.nyu import NYURGBD_Dataset
        from datasets.sun import SUNRGBD_Dataset
        train_datasets.append(SUNRGBD_Dataset(root_dir=args.sun_root, dataset_name='sun', split="train", modality_pair=args.modality_pair,
                                              scale_factor=args.scale_factor, mode='train', text_template=args.get('text_template'),
                                              data_version=args.get('data_version', '')
                                              ))
        test_datasets.append(NYURGBD_Dataset(root_dir=args.nyu_root, dataset_name='sun', split="test", modality_pair=dataloader_output_modality,
                                             mode='test', text_template=args.get('text_template'),
                                             ))

    if 'asa' in arg_dataset_list:
        from datasets.asa import AudiosetDataset
        train_datasets.append(AudiosetDataset(root_dir=args.asa_root, dataset_name='asa', split="train", modality_pair=args.modality_pair,
                                              scale_factor=args.scale_factor, dense_text=args.get("dense_text", False), mode='train'))

        test_datasets.append(AudiosetDataset(root_dir=args.asa_root, dataset_name='asa', split="test",
                                             modality_pair=dataloader_output_modality, dense_text=args.get("dense_text", False), mode='test'))

    if 'vggs' in arg_dataset_list:
        from datasets.vggsound import VGGSoundDataset
        train_datasets.append(VGGSoundDataset(root_dir=args.vggs_root, dataset_name='vggs', split="train", modality_pair=args.modality_pair,
                                              scale_factor=args.scale_factor, mode='train'))

        test_datasets.append(VGGSoundDataset(root_dir=args.vggs_root, dataset_name='vggs', split="test",
                                             modality_pair=dataloader_output_modality, mode='test'))
    if 'audiocaps' in arg_dataset_list:
        from datasets.audiocaps import AudioCapsDataset
        train_datasets.append(AudioCapsDataset(root_dir=args.audiocaps_root, dataset_name='audiocaps', split="train", modality_pair=args.modality_pair,
                                              scale_factor=args.scale_factor, mode='train'))

        test_datasets.append(AudioCapsDataset(root_dir=args.audiocaps_root, dataset_name='audiocaps', split="test",
                                             modality_pair=dataloader_output_modality, mode='test'))

    if 'esc' in arg_dataset_list:
        from datasets.esc import EscDataset
        fold_index = args.get("fold_index")
        print(f"dataset init: args.get('fold_index') = {fold_index}")
        # train_datasets.append(EscDataset(root_dir=args.esc_root, split="train", modality_pair=args.modality_pair,
        #                                       scale_factor=args.scale_factor, mode='train'))
        test_datasets.append(EscDataset(root_dir=args.esc_root, dataset_name='esc', split="test",
                                             modality_pair=[args.modality_eval], mode='test', fold_index=fold_index))

    if 'ave' in arg_dataset_list:
        from datasets.ave import AVEDataset
        train_datasets.append(AVEDataset(root_dir=args.ave_root, dataset_name='ave', split="train", modality_pair=args.modality_pair,
                                          scale_factor=args.scale_factor, dense_text=args.get("dense_text", False), mode='train'))
        test_datasets.append(AVEDataset(root_dir=args.ave_root, dataset_name='ave', split="test",
                                             modality_pair=dataloader_output_modality, dense_text=args.get("dense_text", False), mode='test'))                     
    if 'clotho' in arg_dataset_list:
        from datasets.clotho import ClothoDataset
        test_datasets.append(ClothoDataset(root_dir=args.clotho_root, dataset_name='clotho', split="test",
                                             modality_pair=['text'], mode='test'))
    if 'llvip' in arg_dataset_list:
        from datasets.llvip import LLVIPDataset
        train_datasets.append(LLVIPDataset(root_dir=args.llvip_root, dataset_name='llvip', split="train", modality_pair=args.modality_pair,
                                           scale_factor=args.scale_factor, mode='train'))
        test_datasets.append(LLVIPDataset(root_dir=args.llvip_root, dataset_name='llvip', split="test",
                                          modality_pair=dataloader_output_modality, mode='test'))
    
    if 'flir' in arg_dataset_list:
        from datasets.flir import FLIRDataset
        flir_root = args.flir_root if args.get('flir_cache', None) is None else args.get('flir_cache') 
        print(f"flir_root = {flir_root}")
        train_datasets.append(FLIRDataset(root_dir=flir_root, dataset_name='flir', split="train", modality_pair=args.modality_pair,
                                           scale_factor=args.scale_factor, mode='train'))
        test_datasets.append(FLIRDataset(root_dir=flir_root, dataset_name='flir', split="test",
                                          modality_pair=dataloader_output_modality, mode='test'))

    if 'tag_m' in arg_dataset_list:  # touch_and_go
        from datasets.tag import TAG_Dataset
        train_datasets.append(TAG_Dataset(root_dir=args.tag_root, dataset_name='tag_m', split="train_material", modality_pair=args.modality_pair,
                                           scale_factor=args.scale_factor, mode='train'))
        test_datasets.append(TAG_Dataset(root_dir=args.tag_root, dataset_name='tag_m', split="test_material",
                                          modality_pair=dataloader_output_modality, mode='test'))
    
    if 'tag_h' in arg_dataset_list:  # touch_and_go
        from datasets.tag import TAG_Dataset
        train_datasets.append(TAG_Dataset(root_dir=args.tag_root, dataset_name='tag_h', split="train_hard", modality_pair=args.modality_pair,
                                           scale_factor=args.scale_factor, mode='train'))
        test_datasets.append(TAG_Dataset(root_dir=args.tag_root, dataset_name='tag_h', split="test_hard",
                                          modality_pair=dataloader_output_modality, mode='test'))
    if 'tag_r' in arg_dataset_list:  # touch_and_go
        from datasets.tag import TAG_Dataset
        train_datasets.append(TAG_Dataset(root_dir=args.tag_root, dataset_name='tag_r', split="train_rough", modality_pair=args.modality_pair,
                                           scale_factor=args.scale_factor, mode='train'))
        test_datasets.append(TAG_Dataset(root_dir=args.tag_root, dataset_name='tag_r', split="test_rough",
                                          modality_pair=dataloader_output_modality, mode='test'))
    
    if 'eeg' in arg_dataset_list:
        from datasets.eeg import EEGDataset
        train_datasets.append(EEGDataset(root_dir=args.eeg_root, dataset_name='eeg', split="train", modality_pair=args.modality_pair,
                                           scale_factor=args.scale_factor, mode='train', eeg_cfg=args.get('eeg_conf')))
        test_datasets.append(EEGDataset(root_dir=args.eeg_root, dataset_name='eeg', split="test",
                                          modality_pair=dataloader_output_modality, mode='test', eeg_cfg=args.get('eeg_conf')))

    if ('esc' not in arg_dataset_list) and ('clotho' not in arg_dataset_list):
        assert len(arg_dataset_list) == len(train_datasets)
    assert len(arg_dataset_list) == len(test_datasets)

    print(f"Get {len(arg_dataset_list)} datasets.")

    if len(arg_dataset_list) == 1:
        train_dataset = train_datasets[0] if len(train_datasets) else None
        test_dataset = test_datasets[0]
    else:
        if args.get("concat_datasets", False):
            train_dataset = ConcatDataset(train_datasets)
            test_dataset = test_datasets
        else:
            # vision data combine training
            if arg_dataset_list == ["imagenet", "places365", "k400"]:
                train_dataset = [ConcatDataset([train_datasets[0], train_datasets[1]]), train_datasets[2]]
                test_dataset = test_datasets
            else:
                assert len(arg_dataset_list) == 2, "only allow two datasets for alternative training if disable concat_datasets"
                train_dataset = train_datasets
                test_dataset = test_datasets
    
    return train_dataset, test_dataset
 