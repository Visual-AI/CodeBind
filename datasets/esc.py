# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind
# @Reference     : ImageBind, Meta Platforms, Inc. and affiliates.

import os
import csv
import pandas as pd
from torch.utils.data import Dataset
import random

from models.codebind_model import ModalityType
from datasets import data

# load esc class names (50 classes in total)
with open('.datasets/ESC-50/meta/esc50.csv', 'r') as f:
    lines = f.readlines()
esc_scene_names = list(set([line.split(',')[3].replace('_', ' ') for line in lines[1:]]))

class EscDataset(Dataset):
    def __init__(self, root_dir: str,
                 dataset_name: str,
                 modality_pair: list = ['vision', 'audio'],
                 mode: str = None,  # mode is used for different data augmentation in train and test
                 split: str = 'train',
                 scale_factor=1,
                 fold_index=0,  # 0 is for all fold, 1-5 for a specific fold
                 device: str = 'cpu'):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.device = device
        self.split = split
        self.mode = mode
        self.class_names = esc_scene_names
        self.scale_factor = scale_factor
        self.modality_pair = modality_pair

        for m in self.modality_pair:
            assert m in ['vision', 'audio', 'text'], f"Get '{m}'"
        self.has_vision = True if 'vision' in self.modality_pair else False
        self.has_audio = True if 'audio' in self.modality_pair else False
        self.has_text = True if 'text' in self.modality_pair else False

        self.path_list = list()
        assert fold_index in [0, 1, 2, 3, 4, 5]  # 0 is for all csv
        if split in ['train', 'all']:
            pass
        if split in ['test', 'all']:
            fold_list = [
                'esc_audio_test_fold1.csv',
                'esc_audio_test_fold2.csv',
                'esc_audio_test_fold3.csv',
                'esc_audio_test_fold4.csv',
                'esc_audio_test_fold5.csv',
            ]
            if fold_index == 0:
                for _i_fold in fold_list:
                    csv_path = os.path.join(root_dir, _i_fold)
                    print(f"csv_path={csv_path}")
                    self.parse_csv(csv_path)
            else:
                csv_path = os.path.join(root_dir, fold_list[fold_index-1])
                print(f"csv_path={csv_path}")
                self.parse_csv(csv_path)

        # 放大N倍进行训练，将self.parse_csv repeat N份
        if self.scale_factor > 1 and split != 'test':
            self.path_list = self.path_list * int(self.scale_factor)
        print(f'EscDataset, {self.modality_pair}, split = {split}, length = {len(self.path_list)}')

    def __len__(self):
        return len(self.path_list)

    def get_classnames(self):
        return self.class_names

    def parse_csv(self, csv_path):
        # If the processed csv file for the Audioset does not exist,
        # then the dataset needs to be prepared from the raw data.
        if not os.path.exists(csv_path):
            prepare_esc_dataset(self.root_dir)

        with open(csv_path, 'r', encoding="utf-8") as csv_data:
            reader = csv.reader(csv_data, delimiter=',')
            next(reader)
            for row in reader:
                audio_path = os.path.join(self.root_dir, 'audio', row[0])  # audio
                scene_name = row[1]  # scene
                scene_name = scene_name.replace('_', ' ')

                self.path_list.append([audio_path, scene_name])

    def __getitem__(self, index):
        scene_name = self.path_list[index][1]
        output_dict = {'label': scene_name}
        if self.has_vision:
            raise Exception("Video is not available for ESC-50 dataset")

        if self.has_audio:
            _audio = data.load_and_transform_audio_data(audio_paths=[self.path_list[index][0]], device=self.device,
                                                        mode=self.mode)
            if _audio is None:
                return self.__getitem__(index+1)
            else:
                output_dict.update({ModalityType.AUDIO: _audio[0]})  # _audio: size [1, 3, 1, 128, 204] 3: number of clips

        if self.has_text:
            # t = random.choice(data.imagenet_templates)  # imagenet_templates 对结果有影响
            # class_text = t.format(scene_name)
            # class_text = f"The sound of {scene_name}"
            # class_text = f"A photo of {scene_name}"
            class_text = scene_name
            _text = data.load_and_transform_text([class_text], self.device)
            output_dict.update({ModalityType.TEXT: _text[0]})
        return output_dict



def prepare_esc_dataset(esc_data_root):

    print("Preparing ESC audio dataset csv files for evaluation.")
    # remove bad case

    # prepare validation data info
    all_data = pd.read_csv(os.path.join(esc_data_root, 'meta/esc50.csv'))
    data_class_1 = all_data.loc[all_data['fold'] == 1, ['filename', 'category', 'fold']]
    data_class_2 = all_data.loc[all_data['fold'] == 2, ['filename', 'category', 'fold']]
    data_class_3 = all_data.loc[all_data['fold'] == 3, ['filename', 'category', 'fold']]
    data_class_4 = all_data.loc[all_data['fold'] == 4, ['filename', 'category', 'fold']]
    data_class_5 = all_data.loc[all_data['fold'] == 5, ['filename', 'category', 'fold']]

    data_class_1.to_csv(os.path.join(esc_data_root, 'esc_audio_test_fold1.csv'), index=False)
    data_class_2.to_csv(os.path.join(esc_data_root, 'esc_audio_test_fold2.csv'), index=False)
    data_class_3.to_csv(os.path.join(esc_data_root, 'esc_audio_test_fold3.csv'), index=False)
    data_class_4.to_csv(os.path.join(esc_data_root, 'esc_audio_test_fold4.csv'), index=False)
    data_class_5.to_csv(os.path.join(esc_data_root, 'esc_audio_test_fold5.csv'), index=False)


