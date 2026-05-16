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
import pdb

import random


def random_select_at_least_one(my_list):
    num_to_select = random.randint(1, len(my_list))
    random_elements = random.sample(my_list, num_to_select)
    random.shuffle(random_elements)
    return ", ".join(random_elements)


class MSRVTTDataset(Dataset):
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
        
        self.merge_caption = False
        # self.merge_caption = True
        # self.merge_caption = True if split in ['train', 'all'] else False

        self.unique_caption_cnt_dict = dict()
        self.path_list = list()
        self.video_id_list = list()
        self.video_caption_dict = dict()
        self.video_path_dict = dict()
        self.class_names, self.data_paths = [], []  # for retrieval evaluation
        if split in ['train', 'all']:
            self.parse_csv(os.path.join('.datasets/MSRVTT', 'msrvtt_train.csv'))
            self.parse_csv(os.path.join('.datasets/MSRVTT', 'msrvtt_val.csv'))

        if split in ['test', 'all']:
            csv_path = os.path.join('.datasets/MSRVTT', 'msrvtt_val.csv')
            # csv_path = os.path.join('.datasets/MSRVTT', 'msrvtt_test.csv')
            self.parse_csv(csv_path)
        
        if self.merge_caption:
            # self.class_names = []
            self.data_paths = []
            self.video_id_list = []
            # self.video_caption_dict.update({video_id: exist_video_caption})
            for _k, _v in self.video_caption_dict.items():
                self.video_id_list.append(_k)
                # self.class_names.append(', '.join(_v)) # video_id, video_caption
                self.data_paths.append(self.video_path_dict.get(_k))  # video_id , video_path

        # 放大N倍进行训练，将self.parse_csv repeat N份
        if self.scale_factor > 1 and split != 'test':
            self.path_list = self.path_list * int(self.scale_factor)
        
        # self.data_paths = self.data_paths[0:100]  # debug
        if self.merge_caption:
            print(f'MSRVTTDataset, {self.modality_pair}, split = {split}, length = {len(self.video_id_list)}')
        else:
            print(f'MSRVTTDataset, {self.modality_pair}, split = {split}, length = {len(self.path_list)}')



    def __len__(self):
        if self.mode == 'train':
            return 1000  # debug
        if self.merge_caption:
            return len(self.video_id_list) 
        else:
            return len(self.path_list)

    # def get_classnames(self):
    #     return self.class_names

    def parse_csv(self, csv_path):
        # If the processed csv file for the MSRVTT does not exist,
        # then the dataset needs to be prepared from the raw data.
        if not os.path.exists(csv_path):
            prepare_msrvtt_dataset(self.root_dir)

        with open(csv_path, 'r', encoding="utf-8") as csv_data:
            csv_idx = 0
            for row in csv.reader(csv_data):
                csv_idx += 1
                video_id = row[0]  # video id
                video_path = row[1]  # video path
                video_caption = row[2]  # video caption

                cnt = self.unique_caption_cnt_dict.get(video_caption, 0)
                self.unique_caption_cnt_dict.update({video_caption: cnt+1})

                self.path_list.append([video_id, video_path, video_caption])
                # gather text descriptions for all videos and gather all video paths seperately  
                exist_video_caption = self.video_caption_dict.get(video_id, [])
                if True: # cnt == 0 or  (csv_idx%20==0 and len(exist_video_caption)==0):  # 剔除重复的 caption, 确保至少有一个 caption
                    exist_video_caption.append(video_caption)
                    self.video_caption_dict.update({video_id: exist_video_caption})
                
                self.video_path_dict.update({video_id: video_path})
                #
                if not self.merge_caption:  # and mode == 'test' 
                    # self.class_names.append(video_caption)
                    if video_path not in self.data_paths:
                        self.data_paths.append(video_path)
        

    def __getitem__(self, index):
        if self.mode == 'train':
            index = random.randint(0, len(self.path_list)-1)  # debug
        if self.merge_caption:
            # video_id = self.path_list[index][0]
            video_id = self.video_id_list[index]
            video_path = self.video_path_dict[video_id]
            video_caption_list = self.video_caption_dict[video_id]
            if self.split == 'train':
                video_caption = random_select_at_least_one(video_caption_list)
            else:
                # video_caption = ", ".join(video_caption_list)
                video_caption = video_caption_list[0]
        else:
            video_id = self.path_list[index][0]
            video_path = self.path_list[index][1]
            # assert self.data_paths[index//20] == video_path
            video_caption = self.path_list[index][2]

        output_dict = {'label': video_caption}
        if self.mode == 'test':
            # TODO 不同视频的 caption相同，因而正确的retrieval_id不止一个
            output_dict.update({'retrieval_id': int(video_id) if self.split == 'all' else int(video_id) - int(self.path_list[0][0])}) # label is video id for retrieval
        
        if self.has_vision:
            _video = data.load_and_transform_video_data(
                video_paths=[video_path], device=self.device,
                clips_per_video=1)
            assert _video is not None
            output_dict.update({ModalityType.VISION: _video[0]}) # _video[0].size() [clips per video(15), 3, clip duration(2), 224, 224]
        if self.has_text:
            _text = data.load_and_transform_text([video_caption], self.device)
            output_dict.update({ModalityType.TEXT: _text[0]})
        return output_dict
    

def write_to_csv(dataroot, sentences, filename):
    with open(os.path.join(dataroot, filename), 'w') as csvoutput:
            writer = csv.writer(csvoutput)
            for sentence in tqdm(sentences):
                video_caption = sentence['caption']
                video_path = os.path.join(dataroot, 'TrainValVideo', sentence['video_id']+'.mp4')
                if not os.path.exists(video_path):
                    print(f"Warning, File not exist: {video_path}")
                    continue
                video_id = int(re.findall('\d+', sentence['video_id'])[0])
                writer.writerow([video_id, video_path, video_caption])

def prepare_msrvtt_dataset(msrvtt_data_root):

    # Read the JSON file
    with open(os.path.join(msrvtt_data_root, 'train_val_videodatainfo.json'), 'r') as file:
        video_info = json.load(file)

    # Filter and sort sentences, then write to CSV files
    sentences = video_info['sentences']
    sentences_val = [sentence for sentence in sentences if int(re.findall('\d+', sentence['video_id'])[0]) > 6512]
    sentences_train = [sentence for sentence in sentences if int(re.findall('\d+', sentence['video_id'])[0]) <= 6512]

    sentences_train = sorted(sentences_train, key=lambda x: int(re.findall('\d+', x['video_id'])[0]))
    sentences_val = sorted(sentences_val, key=lambda x: int(re.findall('\d+', x['video_id'])[0]))

    print("Preparing MSRVTT video-text dataset csv files for training.")
    write_to_csv(msrvtt_data_root, sentences_train, 'msrvtt_train.csv')
    print("Preparing MSRVTT video-text dataset csv files for validation.")
    write_to_csv(msrvtt_data_root, sentences_val, 'msrvtt_val.csv')


    with open(os.path.join(msrvtt_data_root, 'test_videodatainfo.json'), 'r') as file:
        video_info = json.load(file)
    sentences = video_info['sentences']
    sentences_test = [sentence for sentence in sentences if int(re.findall('\d+', sentence['video_id'])[0]) > 6512+497]
    sentences_test = sorted(sentences_test, key=lambda x: int(re.findall('\d+', x['video_id'])[0]))
    print("Preparing MSRVTT video-text dataset csv files for test.")
    write_to_csv(msrvtt_data_root, sentences_val, 'msrvtt_test.csv')
