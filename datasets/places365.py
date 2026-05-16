# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind
# @Reference     : ImageBind, Meta Platforms, Inc. and affiliates.
import os
import csv
from torch.utils.data import Dataset
import random
from tqdm import tqdm

from models.codebind_model import ModalityType
from datasets import data
import pdb


# load places365 class names
def get_scene_names(root_dir):
    dirname_ls = os.listdir(root_dir)
    for dirname in dirname_ls:
        if os.path.isfile(os.path.join(root_dir, dirname)):
            dirname_ls.remove(dirname)
    return dirname_ls

places365_scene_names = get_scene_names('.datasets/Places365/train')


class Places365Dataset(Dataset):
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
        self.class_names = places365_scene_names
        self.scale_factor = scale_factor
        self.modality_pair = modality_pair

        for m in self.modality_pair:
            assert m in ['vision', 'text'], f"Get '{m}'"
        self.has_vision = True if 'vision' in self.modality_pair else False
        self.has_text = True if 'text' in self.modality_pair else False

        self.path_list = list()
        if split in ['train', 'all']:
            csv_path = os.path.join('.datasets/Places365', 'train.txt')
            self.parse_csv(csv_path)
        if split in ['test', 'all']:
            csv_path = os.path.join('.datasets/Places365', 'val.txt')
            self.parse_csv(csv_path)

        # 放大N倍进行训练，将self.parse_csv repeat N份
        if self.scale_factor > 1 and split != 'test':
            self.path_list = self.path_list * int(self.scale_factor)
        print(f'Places365Dataset, {self.modality_pair}, split = {split}, length = {len(self.path_list)}')

    def __len__(self):
        return len(self.path_list)

    def get_classnames(self):
        return self.class_names

    def parse_csv(self, csv_path):
        with open(csv_path, 'r', encoding="utf-8") as csv_data:
            for row in csv.reader(csv_data):
                image_path = os.path.join('./.datasets/Places365/', row[0])
                scene_name = row[0].split('/')[1]  # scene
                self.path_list.append([image_path, scene_name])

    def __getitem__(self, index):
        scene_name = self.path_list[index][1]
        output_dict = {'label': scene_name}
        if self.has_vision:
            _image = data.load_and_transform_vision_data(image_paths=[self.path_list[index][0]], device=self.device)
            if _image is None:
                return self.__getitem__(index+1)
            else:
                output_dict.update({ModalityType.VISION: _image[0]})

        if self.has_text:
            t = random.choice(data.imagenet_templates)
            class_text = t.format(scene_name)
            _text = data.load_and_transform_text([class_text], self.device)
            output_dict.update({ModalityType.TEXT: _text[0]})
        return output_dict
    
