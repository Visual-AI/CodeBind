# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind

import os
from torch.utils.data import Dataset
import random

from models.codebind_model import ModalityType
from datasets import data


# load ave class names (28 classes in total)
with open('.datasets/AVE_Dataset/Annotations.txt', 'r') as f:
    lines = f.readlines()
categories = set()
for line in lines[1:]:
    category = line.strip().split("&")[0]
    categories.add(category)
ave_scene_names = list(categories)


class AVEDataset(Dataset):
    def __init__(self, root_dir: str,
                 dataset_name: str,
                 modality_pair: list = ['vision', 'audio'],
                 mode: str = None,  # mode is used for different data augmentation in train and test
                 split: str = 'train',
                 scale_factor=1,
                 dense_text=False,
                 device: str = 'cpu'):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.device = device
        self.split = split
        self.mode = mode
        self.class_names = ave_scene_names
        self.scale_factor = scale_factor
        self.modality_pair = modality_pair
        self.dense_text = dense_text

        for m in self.modality_pair:
            assert m in ['vision', 'audio', 'text'], f"Get '{m}'"
        self.has_vision = True if 'vision' in self.modality_pair else False
        self.has_audio = True if 'audio' in self.modality_pair else False
        self.has_text = True if 'text' in self.modality_pair else False

        self.path_list = list()
        if split in ['train', 'all']:
            if dense_text:
                csv_path = os.path.join('.datasets/AVE_Dataset', 'trainSet_vlmcaption.txt')
            else:
                csv_path = os.path.join('.datasets/AVE_Dataset', 'trainSet.txt')
            self.parse_csv(csv_path)
        if split in ['test', 'all']:
            if dense_text:
                csv_path = os.path.join('.datasets/AVE_Dataset', 'valSet_vlmcaption.txt')
            else:
                csv_path = os.path.join('.datasets/AVE_Dataset', 'valSet.txt')
            self.parse_csv(csv_path)

        # 放大N倍进行训练，将self.parse_csv repeat N份
        if self.scale_factor > 1 and split != 'test':
            self.path_list = self.path_list * int(self.scale_factor)
        print(f'AVEDataset, {self.modality_pair}, split = {split}, length = {len(self.path_list)}')

    def __len__(self):
        return len(self.path_list)

    def get_classnames(self):
        return self.class_names

    def parse_csv(self, csv_path):

        with open(csv_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            row = line.strip()
            if not line:
                continue
            if self.dense_text:
                row = line.split("&", 5)
                if len(row) != 6:
                    print(f"Warning: Skipping invalid format line: {line}")
                    continue
            else:
                row = line.split("&")
            video_path = os.path.join(self.root_dir, "AVE", f"{row[1]}.mp4")  # video
            audio_path = os.path.join(self.root_dir, "audio", f"{row[1]}.wav")  # audio
            scene_name = row[0]  # scene
            if self.dense_text:
                scene_text = row[-1]  # last column is the dense text
                self.path_list.append([video_path, audio_path, scene_name, scene_text])
            else:
                self.path_list.append([video_path, audio_path, scene_name])

    def __getitem__(self, index):
        scene_name = self.path_list[index][2]
        output_dict = {'label': scene_name}
        if self.has_vision:
            _video = data.load_and_transform_video_data(video_paths=[self.path_list[index][0]], device=self.device)
            if _video is None:
                return self.__getitem__(index+1)
            else:
                output_dict.update({ModalityType.VISION: _video[0]})

        if self.has_audio:
            _audio = data.load_and_transform_audio_data(audio_paths=[self.path_list[index][1]], device=self.device,
                                                        mode=self.mode)
            if _audio is None:
                return self.__getitem__(index+1)
            else:
                output_dict.update({ModalityType.AUDIO: _audio[0]})  # _audio: size [1, 3, 1, 128, 204] 3: number of clips

        if self.has_text:
            if self.dense_text:
                scene_text = self.path_list[index][-1]
                _text = data.load_and_transform_text([scene_text], self.device)
                output_dict.update({ModalityType.TEXT: _text[0]})
            else:
                t = random.choice(data.imagenet_templates)
                class_text = t.format(scene_name)
                _text = data.load_and_transform_text([class_text], self.device)
                output_dict.update({ModalityType.TEXT: _text[0]})
        return output_dict

