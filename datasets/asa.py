# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind
# @Reference     : https://github.com/YuanGongND/ast/blob/master/src/dataloader.py

import os
import csv
from torch.utils.data import Dataset
import random

from models.codebind_model import ModalityType
from datasets import data

# load audioset class names (527 classes in total)
with open('.datasets/AudioSet/asa_scene_names.txt', 'r') as f:
    lines = f.readlines()
lines_list = list(lines)
asa_scene_names = [line.strip() for line in lines_list]

with open('.datasets/AudioSet/unreadable_video.txt', 'r') as f:
    lines = f.readlines()
lines_list = list(lines)
unavailable_video_list = [line.strip() for line in lines_list]

unavailable_audio_list = ['./.datasets/AudioSet/audio/eval/0N0C0Wbe6AI_30.000_40.000.wav']


class AudiosetDataset(Dataset):
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
        self.class_names = asa_scene_names
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
                if not os.path.exists(os.path.join(root_dir, 'asa_audio_train.csv')):
                    prepare_asa_dataset(self.root_dir)
                csv_path = os.path.join(root_dir, 'asa_audio_train_vlmcaption.csv')
            else:
                csv_path = os.path.join(root_dir, 'asa_audio_train.csv')
            # csv_path = os.path.join(root_dir, 'asa_audio_test_debug.csv')
            self.parse_csv(csv_path)
        if split in ['test', 'all']:
            if dense_text:
                if not os.path.exists(os.path.join(root_dir, 'asa_audio_test.csv')):
                    prepare_asa_dataset(self.root_dir)
                csv_path = os.path.join(root_dir, 'asa_audio_test_vlmcaption.csv')
            else:
                csv_path = os.path.join(root_dir, 'asa_audio_test.csv')
            # csv_path = os.path.join(root_dir, 'asa_audio_test_debug.csv')
            self.parse_csv(csv_path)

        # 放大N倍进行训练，将self.parse_csv repeat N份
        if self.scale_factor > 1 and split != 'test':
            self.path_list = self.path_list * int(self.scale_factor)
        print(f'ASADataset, {self.modality_pair}, split = {split}, length = {len(self.path_list)}')

    def __len__(self):
        return len(self.path_list)

    def get_classnames(self):
        return self.class_names

    def parse_csv(self, csv_path):
        # If the processed csv file for the Audioset does not exist,
        # then the dataset needs to be prepared from the raw data.
        if not os.path.exists(csv_path):
            prepare_asa_dataset(self.root_dir)

        with open(csv_path, 'r', encoding="utf-8") as csv_data:
            reader = csv.reader(csv_data, delimiter=',')
            next(reader)
            for row in reader:
                assert len(row) == 8 if self.dense_text else 7, f"Expected 8 or 7 columns in the CSV, but got {len(row)}: {row}"
                video_path = row[4]  # video
                audio_path = row[5]  # audio
                scene_name = row[6]  # scene

                # unavailable_file_list
                if video_path in unavailable_video_list or audio_path in unavailable_audio_list:
                    continue
                if ('.part' in video_path) or ('.ytdl' in video_path):
                    continue

                if self.dense_text:
                    scene_text = row[-1]
                    self.path_list.append([video_path, audio_path, scene_name, scene_text])
                else:
                    self.path_list.append([video_path, audio_path, scene_name])

    def __getitem__(self, index):
        scene_name = self.path_list[index][2]
        output_dict = {'label': scene_name}
        # output_dict = {'label': scene_name.split(';')}
        if self.has_vision:
            _video = data.load_and_transform_video_data(video_paths=[self.path_list[index][0]], device=self.device)
            if _video is None:
                return self.__getitem__(index+1)
            else:
                output_dict.update({ModalityType.VISION: _video[0]}) # _video: size [1, 15, 3, 2, 224, 224]

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
            else:
                t = random.choice(data.imagenet_templates)
                i_scene_name = random.choice(scene_name.split(';'))
                # class_text = [t.format(scene_name_i) for scene_name_i in scene_name.split(';')]
                # class_text = t.format(scene_name.split(';')[-1])
                class_text = t.format(i_scene_name)
                _text = data.load_and_transform_text([class_text], self.device)
            output_dict.update({ModalityType.TEXT: _text[0]})
        return output_dict



def prepare_asa_dataset(ASA_data_root):

    print("Preparing ASA audio-video dataset csv files for training and evaluation.")

    # gather video & audio filename to lists
    video_file_names_train = os.listdir(os.path.join(ASA_data_root, 'video/balanced_train'))
    audio_file_names_train = os.listdir(os.path.join(ASA_data_root, 'audio/balanced_train'))
    video_file_names_eval = os.listdir(os.path.join(ASA_data_root, 'video/eval'))
    audio_file_names_eval = os.listdir(os.path.join(ASA_data_root, 'audio/eval'))

    # remove bad case
    # unreadable_audio = os.path.join(ASA_data_root, 'audio/valid_wav/0N0C0Wbe6AI.wav')
    # audio_file_names_eval.remove(unreadable_audio)

    label_dict = {}  # label_id: scene name
    with open(os.path.join(ASA_data_root, 'class_labels_indices.csv'), 'r', encoding="utf-8") as csv_data:
        reader = csv.reader(csv_data, delimiter=',')
        next(reader)
        for row in reader:
            label_dict[row[1]] = row[2]

    # prepare training data info
    with open(os.path.join(ASA_data_root, 'audio/train.csv'), 'r') as csvinput:
        with open(os.path.join(ASA_data_root, 'asa_audio_train.csv'), 'w') as csvoutput:
            writer = csv.writer(csvoutput)

            for row in csv.reader(csvinput):
                if row[0] == "YTID":
                    writer.writerow(row + ['video_path', 'audio_path', 'scene_type'])
                else:
                    video_id = row[0]
                    # find video path through video id
                    for video_name in video_file_names_train:
                        if video_id in video_name:
                            video_path = os.path.join(ASA_data_root, 'video/balanced_train', video_name)
                            break
                        else:
                            video_path = None
                    # find audio path through video id
                    for audio_name in audio_file_names_train:
                        if video_id in audio_name:
                            audio_path = os.path.join(ASA_data_root, 'audio/balanced_train', audio_name)
                            break
                        else:
                            audio_path = None
                    # find scene type in label dict
                    label_ids = row[3].split(',')
                    scene_type = [label_dict[label_id] for label_id in label_ids]
                    scene_type = ';'.join(scene_type)

                    if video_path is not None and audio_path is not None:
                        writer.writerow(row + [video_path, audio_path, scene_type])

    # prepare validation data info
    with open(os.path.join(ASA_data_root, 'audio/valid.csv'), 'r') as csvinput:
        with open(os.path.join(ASA_data_root, 'asa_audio_test.csv'), 'w') as csvoutput:
            writer = csv.writer(csvoutput)

            for row in csv.reader(csvinput):
                if row[0] == "YTID":
                    writer.writerow(row + ['video_path', 'audio_path', 'scene_type'])
                else:
                    video_id = row[0]
                    # find video path through video id
                    for video_name in video_file_names_eval:
                        if video_id in video_name:
                            video_path = os.path.join(ASA_data_root, 'video/eval',
                                                      video_name)
                            break
                        else:
                            video_path = None
                    # find audio path through video id
                    for audio_name in audio_file_names_eval:
                        if video_id in audio_name:
                            audio_path = os.path.join(ASA_data_root, 'audio/eval', audio_name)
                            break
                        else:
                            audio_path = None
                    # find scene type in label dict
                    label_ids = row[3].split(',')
                    scene_type = [label_dict[label_id] for label_id in label_ids]
                    scene_type = ';'.join(scene_type)

                    if video_path is not None and audio_path is not None:
                        writer.writerow(row + [video_path, audio_path, scene_type])