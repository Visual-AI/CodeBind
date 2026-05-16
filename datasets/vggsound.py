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

# load vggsound class names (309 classes in total)
vggs_scene_names = []
with open('.datasets/VGGSound/stat.csv', 'r') as f:
    for row in csv.reader(f):
        vggs_scene_names.append(row[0])

class VGGSoundDataset(Dataset):
    def __init__(self, root_dir: str,
                 dataset_name: str,
                 modality_pair: list = ['vision', 'audio'],
                 mode: str = None,  # mode is used for different data augmentation in train and test
                 split: str = 'train',
                 scale_factor=1,
                 device: str = 'cpu'):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.device = device
        self.split = split
        self.mode = mode
        self.class_names = vggs_scene_names
        self.scale_factor = scale_factor
        self.modality_pair = modality_pair

        for m in self.modality_pair:
            assert m in ['vision', 'audio', 'text'], f"Get '{m}'"
        self.has_vision = True if 'vision' in self.modality_pair else False
        self.has_audio = True if 'audio' in self.modality_pair else False
        self.has_text = True if 'text' in self.modality_pair else False

        self.path_list = list()
        if split in ['train', 'all']:
            csv_path = os.path.join('.datasets/VGGSound', 'vggs_audio_train.csv')
            # csv_path = os.path.join('.datasets/VGGSound', 'asa_audio_test_debug.csv')
            self.parse_csv(csv_path)
        if split in ['test', 'all']:
            csv_path = os.path.join('.datasets/VGGSound', 'vggs_audio_test.csv')
            # csv_path = os.path.join('.datasets/VGGSound', 'asa_audio_test_debug.csv')
            self.parse_csv(csv_path)

        # 放大N倍进行训练，将self.parse_csv repeat N份
        if self.scale_factor > 1 and split != 'test':
            self.path_list = self.path_list * int(self.scale_factor)
        print(f'VGGSoundDataset, {self.modality_pair}, split = {split}, length = {len(self.path_list)}')

    def __len__(self):
        return len(self.path_list)

    def get_classnames(self):
        return self.class_names

    def parse_csv(self, csv_path):
        # If the processed csv file for the VGGSound does not exist,
        # then the dataset needs to be prepared from the raw data.
        if not os.path.exists(csv_path):
            prepare_vggs_dataset(self.root_dir, self.split)

        with open(csv_path, 'r', encoding="utf-8") as csv_data:
            for row in csv.reader(csv_data):
                video_path = row[2]  # video
                audio_path = row[3]  # audio
                scene_name = row[4]  # scene

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
            t = random.choice(data.imagenet_templates)
            class_text = t.format(scene_name)
            _text = data.load_and_transform_text([class_text], self.device)
            output_dict.update({ModalityType.TEXT: _text[0]})
        return output_dict


def prepare_vggs_dataset(vggs_data_root, split='train'):
    assert split in ['train', 'test']

    # gather video & audio filename to lists
    video_file_names = os.listdir(os.path.join(vggs_data_root, f"video/{split}"))
    audio_file_names = os.listdir(os.path.join(vggs_data_root, f"audio/{split}"))

    # video_file_names_eval = os.listdir(os.path.join(vggs_data_root, 'video/test'))
    # audio_file_names_eval = os.listdir(os.path.join(vggs_data_root, 'audio/test'))

    # remove bad case

    # prepare training data info
    # vggs_audio_csv_filepath = f".datasets/VGGSound/vggs_audio_{split}.csv"
    # split_list_csv_filepath = f".datasets/VGGSound/{split}.csv"
    # scorrupted_list_csv_filepath = f".datasets/VGGSound/corrupted_{split}.csv"
    
    vggs_audio_csv_filepath = os.path.join(vggs_data_root, f"vggs_audio_{split}.csv")
    split_list_csv_filepath = os.path.join(vggs_data_root, f"{split}.csv")
    scorrupted_list_csv_filepath = os.path.join(vggs_data_root, f"corrupted_{split}.csv")
    
    corrupted_list = []
    if not os.path.exists(vggs_audio_csv_filepath):
        print(f"Preparing VGGSound audio-video dataset csv files for {split}.")
        with open(split_list_csv_filepath, 'r') as csv_file:
            lines = len(csv_file.readlines())
        with open(split_list_csv_filepath, 'r') as csvinput:
            with open(vggs_audio_csv_filepath, 'w') as csvoutput:
                writer = csv.writer(csvoutput)

                for row in tqdm(csv.reader(csvinput), total=lines):
                    # get video id and start seconds
                    video_info = row[0].rsplit('_', 1)
                    assert len(video_info) == 2
                    video_id = video_info[0]
                    start_seconds = video_info[1].split('.')[0].lstrip('0')

                    # find video path through video id
                    for video_name in video_file_names:
                        if video_id in video_name:
                            video_path = os.path.join(vggs_data_root, f"video/{split}", video_name)
                            #
                            video_size = os.path.getsize(video_path) // 1024  # k
                            if video_size < 10:
                                print("Corrupted video, size < 10k:", video_path)
                                corrupted_list.append(video_path)
                                video_path = None
                            break
                        else:
                            video_path = None
                    # find audio path through video id
                    for audio_name in audio_file_names:
                        if video_id in audio_name:
                            audio_path = os.path.join(vggs_data_root, f"audio/{split}", audio_name)
                            break
                        else:
                            audio_path = None
                    # get label
                    label = row[1]

                    if video_path is not None and audio_path is not None:
                        # cnt += 1
                        # print(f"video_path={video_path}, audio_path={audio_path}")
                        writer.writerow([video_id, start_seconds, video_path, audio_path, label])
                    
    print(f"Get {len(corrupted_list)} corrpted video:", corrupted_list)

    with open(scorrupted_list_csv_filepath, 'w') as f:
        for line in corrupted_list:
            f.write(f"{line}\n")
