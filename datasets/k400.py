# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind
# @Reference     : ImageBind, Meta Platforms, Inc. and affiliates.
import os
import csv
import json
import re
from torch.utils.data import Dataset
from tqdm import tqdm

from models.codebind_model import ModalityType
from datasets import data
from util import unavailable_video_detect
import pdb

# obtain unreadable video list
filter_video_list_path = '.datasets/K400/unreadable_video.txt'
# filter_video_list_path = '.datasets/k400_unreadable_video.txt'
if not os.path.exists(filter_video_list_path):
    unavailable_video_detect('.datasets/K400/train', '.datasets/K400/val', 
                             save_dir=filter_video_list_path, second_dir_exist=True)
with open(filter_video_list_path, 'r') as f:
    lines = f.readlines()
unavailable_video = [line.strip() for line in list(lines)]

class KineticsDataset(Dataset):
    def __init__(self, root_dir: str,
                 dataset_name: str,
                 modality_pair: list = ['vision', 'text'],
                 mode: str = None,  # mode is used for different data augmentation in train and test
                 split: str = 'train',
                 scale_factor=1,
                 device: str = 'cpu'):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.device = device
        self.split = split
        self.mode = mode
        self.scale_factor = scale_factor
        self.modality_pair = modality_pair

        for m in self.modality_pair:
            assert m in ['vision', 'text'], f"Get '{m}'"
        self.has_vision = True if 'vision' in self.modality_pair else False
        self.has_text = True if 'text' in self.modality_pair else False

        if mode == 'test':
            self.has_vision, self.has_text = True, True
        self.meta_list = list()
        if split in ['train', 'all']:
            self.parse_csv(".datasets/K400/train.list", ".datasets/K400/train")
        if split in ['test', 'all']:
            self.parse_csv(".datasets/K400/val.list", ".datasets/K400/val")
        
        # get classnames from file
        label_file = open(".datasets/K400/Kinetics-400_labels.txt", "r") 
        label_data = label_file.read()
        self.class_names = label_data.split("\n")  
        label_file.close()

        # 放大N倍进行训练，将self.parse_csv repeat N份
        if self.scale_factor > 1 and split != 'test':
            self.meta_list = self.meta_list * int(self.scale_factor)
        print(f'KineticsDataset, {self.modality_pair}, split = {split}, length = {len(self.meta_list)}')

    def __len__(self):
        return len(self.meta_list)

    def get_classnames(self):
        return self.class_names

    def parse_csv(self, csv_path, dataroot):
        self.class_names = []
        self.video_paths = []
        with open(csv_path, 'r', encoding="utf-8") as csv_data:
            for i, row in enumerate(csv.reader(csv_data, delimiter=' ')):
                video_path = row[0]  # video path
                video_label_id = int(row[1])  # video id
                video_label_text = video_path.split('/')[0].replace('_', ' ')  # Human Action
                video_fullpath = os.path.join(dataroot, video_path)

                # filter out unavailable video
                if video_fullpath in unavailable_video:
                    continue

                # 已验证，所有文件均存在
                # if not os.path.exists(video_fullpath):
                #     print(f"Warning, file not exist: {video_fullpath}")
                #     continue
                self.meta_list.append([video_label_id, video_fullpath, video_label_text])

    def __getitem__(self, index):
        output_dict = {}
        activity_name = self.meta_list[index][2]  # 同 self.class_names[video_label_id] # 
        output_dict.update({'label': activity_name})

        if self.has_vision:
            _video = data.load_and_transform_video_data(video_paths=[self.meta_list[index][1]], device=self.device, 
                                                        clips_per_video=1)
            assert _video is not None
            # if _video is None:
            #     # index+1可能越界, (1)已经在write_to_csv()中进行过滤掉路径不存在的数据（2）读取失败时，load_and_transform_video_data直接抛出异常
            #     return self.__getitem__(index+1)  
            # else:
            output_dict.update({ModalityType.VISION: _video[0]}) # _video[0].size() [clips per video(15), 3, clip duration(2), 224, 224]

        if self.has_text:
            _text = data.load_and_transform_text([activity_name], self.device)
            output_dict.update({ModalityType.TEXT: _text[0]})
        return output_dict
    

