# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind
# @Reference     : ImageBind, Meta Platforms, Inc. and affiliates.

import os
import random
import json
import logging

import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from typing import Iterable
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from models.codebind_model import ModalityType
from datasets import data

# import pdb


def load_and_transform_eeg_data(eeg, time_low=20, time_high=460, data_len=512):
    eeg = eeg.float().t()  # [channel x time] -> [time x channel]
    eeg = eeg[time_low : time_high, :]

    eeg = np.array(eeg.transpose(0, 1))
    x = np.linspace(0, 1, eeg.shape[-1])
    x2 = np.linspace(0, 1, data_len)
    f = interp1d(x, eeg)
    eeg = f(x2)
    eeg = torch.from_numpy(eeg).float()
    return eeg


class EEGDataset(Dataset):
    def __init__(self, root_dir: str,
                 dataset_name: str,
                 modality_pair: list = ['vision', 'eeg'],
                 mode: str = None,  # mode is used for different data augmentation in train and test
                 split: str = 'train',
                 scale_factor=1,
                 device: str = 'cpu',
                 eeg_cfg = None):

        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.device = device
        self.split = split

        if eeg_cfg is None:
            self.eeg_cfg =  {
                "time_low": 20,
                "time_high": 460,
                "data_len": 512,
            }
        else:
            self.eeg_cfg = eeg_cfg
        print("eeg_cfg: ", self.eeg_cfg)

        split_num = 0
        self.modality_pair = modality_pair
        self.scale_factor = scale_factor
        
        for m in self.modality_pair:
            assert m in ['vision', 'eeg', 'text'], f"Get '{m}'"
        self.has_vision = True if 'vision' in self.modality_pair else False
        self.has_eeg = True if 'eeg' in self.modality_pair else False
        self.has_text = True if 'text' in self.modality_pair else False

        self.img_data_root=f"{root_dir}/imageNet_images"
        anno_path={
            "data": f"{root_dir}/eeg_5_95_std.pth",
            "split_info": f"{root_dir}/block_splits_by_image_all.pth",
        }

        data = torch.load(anno_path["data"])
        self.dataset = data["dataset"]
        self.synset_labels = data["labels"]
        self.image_list = data["images"]


        self.split_info = torch.load(anno_path["split_info"])
        self.imagenet_mapping = json.load(
            open(f"{root_dir}/imagenet_cls_mapping.json", "r")
        )

        self.split_indices = self.split_info["splits"][split_num][self.split]

        # filter data
        self.split_indices = [
            i
            for i in self.split_indices
            if 450 <= self.dataset[i]["eeg"].size(1) <= 600
        ]  # 7959

        if self.scale_factor > 1 and split != 'test':
            self.split_indices = self.split_indices * int(self.scale_factor)

        self.init_labels()
        self.class_names = self.idx2label

    def init_labels(self):
        self.idx2label = [self.imagenet_mapping[i][0] for i in self.synset_labels]
        self.label2idx = {self.idx2label[i]: i for i in range(len(self.idx2label))}

    def __len__(self):
        return len(self.split_indices)

    def __getitem__(self, index):
        # pdb.set_trace()
        fetch_idx = self.split_indices[index]
        datum = self.dataset[fetch_idx]

        label = datum["label"]
        synset_label = self.synset_labels[label]
        cls_name = None
        if self.split in ["train", "pretrain"]:
            cls_name = random.choice(self.imagenet_mapping[synset_label])
        else:
            cls_name = self.imagenet_mapping[synset_label][0]
        output_dict = {'label': cls_name}

        if self.has_vision: # image
            image_name = self.image_list[datum["image"]]
            image_path = os.path.join(self.img_data_root, image_name.split("_")[0], image_name + ".JPEG")
            _image = data.load_and_transform_vision_data(image_paths=[image_path], device=self.device)
            if _image is None:
                return self.__getitem__(index+1)
            else:
                output_dict.update({ModalityType.VISION: _image[0]})
            # image = self.vis_processor(image_path)

        if self.has_text:
            caption = "an image of {}.".format(cls_name)
            _text = data.load_and_transform_text([caption], self.device)
            output_dict.update({ModalityType.TEXT: _text[0]})

        if self.has_eeg:
            eeg_signal = datum["eeg"]  # <class 'torch.Tensor'>  # torch.Size([128, 534])
            # eeg = self.eeg_processor(eeg_signal)
            eeg = load_and_transform_eeg_data(eeg_signal, time_low=self.eeg_cfg["time_low"], time_high=self.eeg_cfg["time_high"], data_len=self.eeg_cfg["data_len"])
            output_dict.update({ModalityType.EEG: eeg})
     
        return output_dict


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    eeg_dataset_train= EEGDataset(
        root_dir='/home/vislab/jieli23/dataset/EEG', 
        dataset_name='eeg', 
        split="train", 
        modality_pair=['vision', 'eeg'], 
        scale_factor=1, 
        mode='train'
        )
    
    train_loader = DataLoader(
                eeg_dataset_train,
                batch_size=1,
                shuffle=True,
                drop_last=True,
                pin_memory=False,
                num_workers=0,
                # collate_fn= collate_fn if self.args.concat_datasets else None,
            )
    

    for i, data in enumerate(train_loader):
        print(f"i={i}")
