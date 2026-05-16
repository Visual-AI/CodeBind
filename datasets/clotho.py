# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind

import os
import csv
from torch.utils.data import Dataset

from models.codebind_model import ModalityType
from datasets import data


class ClothoDataset(Dataset):
    def __init__(self, root_dir: str,
                 dataset_name = 'clotho',
                 modality_pair: list = ['text', 'audio'],
                 mode: str = None,  # mode is used for different data augmentation in train and test
                 split: str = 'test',
                 scale_factor=1,
                 device: str = 'cpu'):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.device = device
        self.split = split
        self.mode = mode
        # self.class_names = esc_scene_names
        self.scale_factor = scale_factor
        self.modality_pair = modality_pair

        for m in self.modality_pair:
            assert m in ['audio', 'text'], f"Only 'audio' and 'text' is available for Clotho dataset, not '{m}'"
        self.has_audio = True if 'audio' in self.modality_pair else False
        self.has_text = True if 'text' in self.modality_pair else False

        self.path_list = list()
        self.data_paths = list()
        self.audio_id_list = list()
        self.data_path_dict = dict()
        self.caption_dict = dict()
        if split in ['train', 'all']:
            pass
        if split in ['test', 'all']:
            csv_path = os.path.join(root_dir, 'clotho_captions_evaluation.csv')
            self.parse_csv(csv_path)
        self.merge_caption = True

        # 放大N倍进行训练，将self.parse_csv repeat N份
        if self.scale_factor > 1 and split != 'test':
            self.path_list = self.path_list * int(self.scale_factor)
        print(f'ClothoDataset, {self.modality_pair}, split = {split}, length = {len(self.path_list)}')

    def __len__(self):
        if self.merge_caption:
            return len(self.audio_id_list)
        else:
            return len(self.path_list)

    # def get_classnames(self):
    #     return self.class_names

    def parse_csv(self, csv_path):
        # If the processed csv file for the Audioset does not exist,
        # then the dataset needs to be prepared from the raw data.
        # if not os.path.exists(csv_path):
        #     prepare_esc_dataset(self.root_dir)

        with open(csv_path, 'r', encoding="utf-8") as csv_data:
            reader = csv.reader(csv_data, delimiter=',')
            next(reader)
            audio_id = 0
            for row in reader:
                
                audio_path = os.path.join(self.root_dir, 'audio/evaluation', row[0])  # audio
                caption_1,caption_2,caption_3,caption_4,caption_5 = row[1:]

                # self.path_list.append([audio_path, caption_1,caption_2,caption_3,caption_4,caption_5])

                self.path_list.append([audio_path, caption_1])
                self.path_list.append([audio_path, caption_2])
                self.path_list.append([audio_path, caption_3])
                self.path_list.append([audio_path, caption_4])
                self.path_list.append([audio_path, caption_5])

                self.audio_id_list.append(audio_id)
                self.caption_dict.update({audio_id: [caption_1,caption_2,caption_3,caption_4,caption_5]})
                self.data_path_dict.update({audio_id: audio_path})

                if audio_path not in self.data_paths:
                        self.data_paths.append(audio_path)
                
                audio_id += 1

    def __getitem__(self, index):
        if self.merge_caption:
            retrieval_audio_id = index
            audio_path = self.data_path_dict.get(retrieval_audio_id)
            caption = ", ".join(self.caption_dict.get(retrieval_audio_id))

        else:
            audio_path = self.path_list[index][0]
            caption = self.path_list[index][1]
            retrieval_audio_id = self.data_paths.index(audio_path)
        
        output_dict = {'label': caption}

        if self.mode == 'test':
            # retrieval_audio_id = self.data_paths.index(audio_path)
            output_dict.update({'retrieval_id': retrieval_audio_id})
            # assert retrieval_audio_id == index  # 

        if self.has_audio:
            _audio = data.load_and_transform_audio_data(audio_paths=[audio_path], device=self.device,
                                                        mode=self.mode)
            if _audio is None:
                return self.__getitem__(index+1)
            else:
                output_dict.update({ModalityType.AUDIO: _audio[0]})  # _audio: size [1, 3, 1, 128, 204] 3: number of clips

        if self.has_text:
            class_text = caption
            _text = data.load_and_transform_text([class_text], self.device)
            output_dict.update({ModalityType.TEXT: _text[0]})
        return output_dict
