# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind
# @Reference     : ImageBind, Meta Platforms, Inc. and affiliates.
import os
import csv
import random
import imageio
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn

from typing import Optional, Callable
from torch.utils.data import Dataset
from torchvision import transforms
from models.codebind_model import ModalityType
from datasets import data
# from datasets.data_transform_rgbd import load_and_transform_rgbd_data


# from types import SimpleNamespace
# ModalityType = SimpleNamespace(
#     VISION="vision",
#     TEXT="text",
#     AUDIO="audio",
#     THERMAL="thermal",
#     DEPTH="depth",
#     IMU="imu",
# )


# debug
# def load_and_transform_rgbd_data(image_paths=None, depth_paths=None,
#                                  camera_types=None, focal_lengths=None, device=None, mode=None):
#     if depth_paths is None and image_paths is None:
#         return None
#     return torch.tensor([[2, 3.3, 4, 5, 255], [1, 2, 3, 4, 5,]])

# def load_and_transform_text(caption, device=None):
#     return [1]


class TAG_Dataset(Dataset):
    def __init__(self, root_dir: str, 
                 dataset_name: str,
                 mode: str = None, # mode is used for different data augmentation in train and test
                 modality_pair: list = ['vision', 'tactile'],
                 split: str = 'train', 
                 scale_factor = 1,
                 device: str = 'cpu', 
                 text_template = None):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.device = device
        self.split = split
        self.scale_factor = scale_factor
        self.modality_pair = modality_pair
        text_template_name = text_template if text_template is not None else 'imagenet'
        # self.text_template = data.text_template_dict.get(text_template_name)

        TACTILE_META_DATA_DIR = os.path.join(root_dir, 'meta')
        anno_path={
            "pretrain": f"{TACTILE_META_DATA_DIR}/pretrain.json",
            "train_material": f"{TACTILE_META_DATA_DIR}/train.json",
            "test_material": f"{TACTILE_META_DATA_DIR}/test.json",
            "train_hard": f"{TACTILE_META_DATA_DIR}/train.json",
            "test_hard": f"{TACTILE_META_DATA_DIR}/test.json",
            "train_rough": f"{TACTILE_META_DATA_DIR}/train_rough.json",
            "test_rough": f"{TACTILE_META_DATA_DIR}/test_rough.json",
            "pretrain_exclude_others": f"{TACTILE_META_DATA_DIR}/pretrain_exclude_others.json",
            "test_material_exclude_others": f"{TACTILE_META_DATA_DIR}/test_exclude_others.json",
        }

        for m in self.modality_pair:
            assert m in ['vision', 'tactile', 'text']
        self.has_vision = True if 'vision' in self.modality_pair else False
        self.has_tactile = True if 'tactile' in self.modality_pair else False
        self.has_text = True if 'text' in self.modality_pair else False

        self.annotation = json.load(open(anno_path[split], "r"))

        if self.scale_factor > 1 and 'train' in split:
            self.annotation = self.annotation * int(self.scale_factor)

        self.init_labels()
        # self.material_name_list = []
        self.material_name_list = self.idx2label
        self.class_names = self.idx2label
        # get csv file
        with open(os.path.join(root_dir, f'tag_{split}.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            # row = ['image_path', 'gel_path', 'material_name', 'sr_label', 'hs_label']
            for ann in self.annotation:
                image_path = os.path.join(self.root_dir, 'dataset', ann['image_path'])
                gel_path = os.path.join(self.root_dir, 'dataset', ann['gel_path'])
                writer.writerow([image_path, gel_path, ann['material_name'], ann['sr_label'], ann['hs_label']])


        print(f'TAG_Dataset, {self.modality_pair}, split = {split}, length = {len(self.annotation)}, text_template={text_template_name}')

    def __len__(self):
        return len(self.annotation)
    
    def init_labels(self):
        self.idx2label = None
        self.label2idx = None
        if "material" in self.split:
            self.idx2label = [
                "concrete",
                "plastic",
                "glass",
                "wood",
                "metal",
                "brick",
                "tile",
                "leather",
                "synthetic fabric",
                # "natural fabric",  # train_material test_material 中均不存在"natural fabric"?
                "ruber",
                "paper",
                "tree",
                "grass",
                "soil",
                "rock",
                "gravel",
                "sand",
                "plants",
                "others",
            ]
            if "exclude_others" in self.split:
                self.idx2label = self.idx2label[:-1]  # exclude "others" class
            self.label2idx = {self.idx2label[i]: i for i in range(len(self.idx2label))}

        elif "hard" in self.split:
            self.idx2label = ["hard", "soft"]
            self.label2idx = {"hard": 0, "soft": 1}

        elif "rough" in self.split:
            self.idx2label = ["smooth", "rough"]
            self.label2idx = {"smooth": 0, "rough": 1}

        else:
            pass
    
    def get_classnames(self):
        return self.material_name_list
        
        # return self.class_names
    
    def __getitem__(self, index):
        """
          {
            "gel_path": "20220601_182052/gelsight_frame/0000033833.jpg",
            "image_path": "20220601_182052/video_frame/0000033833.jpg",
            "material_label": 3,
            "material_name": "wood",
            "sr_label": null,
            "sr_name": null,
            "hs_label": 0,
            "hs_name": "hard"
        }
        """
        # pdb.set_trace()
        ann = self.annotation[index]
        img_path = os.path.join(self.root_dir, 'dataset', ann["image_path"])
        gel_path = os.path.join(self.root_dir, 'dataset', ann["gel_path"])

        # material
        m_label = ann["material_label"]
        
        label = None
        if "rough" in self.split:
            label = ann["sr_label"]
            material_name = "rough" if label else "smooth" # tag_r
        elif "hard" in self.split:
            label = ann["hs_label"]
            material_name = "soft" if label else "hard" # tag_h
        else:
            label = ann["material_label"]
            material_name = ann["material_name"] # tag_m

        output_dict = {'label': material_name if material_name is not None else ''}

        if self.has_vision:
                _rgb = data.load_and_transform_vision_data(image_paths=[img_path], device=self.device, to_tensor=True)
                output_dict.update({ModalityType.VISION: _rgb[0]})

        if self.has_tactile:
                _tac = data.load_and_transform_vision_data(image_paths=[gel_path], device=self.device, to_tensor=True)
                output_dict.update({ModalityType.TACTILE: _tac[0]})

        if self.has_text:
            if material_name is not None:
                caption = "an image of {}.".format(material_name)
                # caption = self.text_processor(caption)
            else:
                caption = "an image showing a material."  # dummy
            _text = data.load_and_transform_text([caption], self.device)
            output_dict.update({ModalityType.TEXT: _text[0]})

        return output_dict


if __name__ == "__main__":
    import pdb
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    m_dset = TAG_Dataset(root_dir='/home/vislab/jieli23/dataset/touch_and_go',
                         dataset_name='tag',
                        #  split='train_material',
                        #  split='test_material',
                        #  split="pretrain",
            # split="train_hard",
            # split="test_hard",
            # split="train_rough",
            split="test_rough",
            # split="pretrain_exclude_others",
            # split="test_material_exclude_others",
                         )
    m_dloader = DataLoader(m_dset,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=False,
                    num_workers=0)
    for i_data in m_dloader:
        pass
        # pdb.set_trace()
        # print('ddd')
    # m_dset.get_classnames()

    idx2label = [
                "concrete",
                "plastic",
                "glass",
                "wood",
                "metal",
                "brick",
                "tile",
                "leather",
                "synthetic fabric",
                "natural fabric",  # train_material test_material 中均不存在"natural fabric"?
                "ruber",
                "paper",
                "tree",
                "grass",
                "soil",
                "rock",
                "gravel",
                "sand",
                "plants",
                "others",
            ]
    
    # material_name_list = ['wood', 'concrete', 'synthetic fabric', 'metal', 'brick', 
    #                       'plastic', 'tile', 'leather', 'paper', 'others', 
    #                       'grass', 'rock', 'soil', 'gravel', 'tree', 
    #                       'plants', 'glass', 'ruber', 'sand']
    material_name_list = m_dset.get_classnames()
    print(material_name_list)
    material_name_list.sort()
    idx2label.sort()

    print(len(material_name_list), len(idx2label))

    print(material_name_list == idx2label)
    
    print('done')

    
