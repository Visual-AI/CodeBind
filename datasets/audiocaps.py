# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind


import os
import csv
from torch.utils.data import Dataset
from tqdm import tqdm

from models.codebind_model import ModalityType
from datasets import data
import pdb


unavailable_video_list = []
unavailable_audio_list = []

class AudioCapsDataset(Dataset):
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
        # self.class_names = asa_scene_names
        self.scale_factor = scale_factor
        self.modality_pair = modality_pair

        for m in self.modality_pair:
            assert m in ['vision', 'audio', 'text'], f"Get '{m}'"
        self.has_vision = True if 'vision' in self.modality_pair else False
        self.has_audio = True if 'audio' in self.modality_pair else False
        self.has_text = True if 'text' in self.modality_pair else False

        self.path_list = list()
        self.class_names = []
        
        if split in ['train', 'all']:
            csv_path = os.path.join(root_dir, 'audiocaps_audio_train.csv')
            self.parse_csv(csv_path)
        if split in ['test', 'all']:
            csv_path = os.path.join(root_dir, 'audiocaps_audio_test.csv')
            self.parse_csv(csv_path)
        # pdb.set_trace()

        # 放大N倍进行训练，将self.parse_csv repeat N份
        if self.scale_factor > 1 and split != 'test':
            self.path_list = self.path_list * int(self.scale_factor)
        print(f'AudioCapsDataset, {self.modality_pair}, split = {split}, length = {len(self.path_list)}')

    def __len__(self):
        return len(self.path_list)

    def get_classnames(self):
        return self.class_names

    def parse_csv(self, csv_path):
        # If the processed csv file for the Audioset does not exist,
        # then the dataset needs to be prepared from the raw data.
        if not os.path.exists(csv_path):
            prepare_audiocaps_dataset(self.root_dir)
        # retrieval_audio_id = -1  # unique_id
        # self.audiocap_id_to_retrieval_audio_id_mapping = {}
        # self.youtube_id_to_retrieval_audio_id_mapping = {}
        self.data_paths = []  # 用于评估
        # audiocap_id_list = []
        # youtube_id_list = []
        with open(csv_path, 'r', encoding="utf-8") as csv_data:
            reader = csv.reader(csv_data, delimiter=',')
            next(reader)
            for row in reader:
                # pdb.set_trace()
                # row: audiocap_id,youtube_id,start_time,caption,video_path,audio_path
                audiocap_id = row[0]  # audiocap_id唯一
                video_path = row[4]  # video
                audio_path = row[5]  # audio
                audio_caption = row[3]  # scene
                # youtube_id = row[1] 

                # unavailable_file_list
                if video_path in unavailable_video_list or audio_path in unavailable_audio_list:
                    continue
                if ('.part' in video_path) or ('.ytdl' in video_path):
                    continue

                # #  测试集中，一个audio片段 （对应一个audiocap_id）有多个（不同但相似的）caption。这些caption 的 retrieval_audio_id应该一致。
                # # step 1: 先确定 retrieval_audio_id
                # # 由于一个 youtube_id 仅对应一个audio_path，故此处用youtube_id作为key
                # # if audio_path not in self.data_paths:
                # #     retrieval_audio_id += 1
                # #     self.data_paths.append(audio_path)  # NOTE: audio_path 必须与retrieval_audio_id对应
                # #     self.data_paths_to_retrieval_audio_id_mapping.update({audio_path: retrieval_audio_id})
                # if youtube_id not in youtube_id_list:
                #     youtube_id_list.append(youtube_id)
                #     if audio_path not in self.data_paths:
                #         self.data_paths.append(audio_path)
                #     else:
                #         # ERROR: 若进入该分支，则youtube_id 与audio_path 不是一对一的，代码有误。
                #         raise ValueError(f" audio_path = {audio_path} in self.data_paths")
                #     retrieval_audio_id += 1  # NOTE 此处累加，不能回头
                #     current_retrieval_audio_id = retrieval_audio_id
                #     self.youtube_id_to_retrieval_audio_id_mapping.update({youtube_id: retrieval_audio_id})
                # else: 
                #     # 获取已有的retrieval_audio_id
                #     current_retrieval_audio_id = self.youtube_id_to_retrieval_audio_id_mapping.get(youtube_id)
                #     # print(f"youtube_id={youtube_id} in youtube_id_list")
                # # step 2: 再建立映射
                # if audiocap_id not in audiocap_id_list:
                #     audiocap_id_list.append(audiocap_id)
                #     # retrieval_audio_id += 1
                #     self.audiocap_id_to_retrieval_audio_id_mapping.update({audiocap_id: current_retrieval_audio_id})
                #     #
                # else:  # audiocap_id唯一, 因此不会进去到该分支
                #     raise ValueError(f"audiocap_id={audiocap_id} in audiocap_id_list")

                if audio_path not in self.data_paths:
                        self.data_paths.append(audio_path)
                
                self.path_list.append([audiocap_id, video_path, audio_path, audio_caption])
                # gather text descriptions for all audios and gather all audio paths seperately
                self.class_names.append(audio_caption)


    def __getitem__(self, index):
        # audiocap_id = self.path_list[index][0]
        video_path = self.path_list[index][1]
        audio_path = self.path_list[index][2]
        audio_caption = self.path_list[index][3]

        output_dict = {'label': audio_caption}
        if self.mode == 'test':
            # retrieval_audio_id = self.audiocap_id_to_retrieval_audio_id_mapping.get(audiocap_id)
            # audio_path_idx = self.data_paths.index(audio_path)
            # assert retrieval_audio_id == audio_path_idx  # 
            # # if retrieval_audio_id != audio_path_idx:  # 
            # #     pdb.set_trace()
            # #     retrieval_audio_id = self.data_paths.index(audio_path)
            retrieval_audio_id = self.data_paths.index(audio_path)
            output_dict.update({'retrieval_id': retrieval_audio_id})

        # output_dict = {'label': scene_name.split(';')}
        if self.has_vision:
            _video = data.load_and_transform_video_data(video_paths=[video_path], device=self.device)
            if _video is None:
                return self.__getitem__(index+1)
            else:
                output_dict.update({ModalityType.VISION: _video[0]}) # _video: size [1, 15, 3, 2, 224, 224]

        if self.has_audio:
            _audio = data.load_and_transform_audio_data(audio_paths=[audio_path], device=self.device,
                                                        mode=self.mode)
            if _audio is None:
                return self.__getitem__(index+1)
            else:
                output_dict.update({ModalityType.AUDIO: _audio[0]})  # _audio: size [1, 3, 1, 128, 204] 3: number of clips

        if self.has_text:
            class_text = audio_caption
            _text = data.load_and_transform_text([class_text], self.device)
            output_dict.update({ModalityType.TEXT: _text[0]})
        return output_dict



def prepare_audiocaps_dataset(data_root):

    print("Preparing AudioCaps_retrieval_dataset csv files for training and evaluation.")

    # gather video & audio filename to lists
    video_file_names = os.listdir(os.path.join(data_root, 'video'))  # 'retrieval_val/video'
    audio_file_names = os.listdir(os.path.join(data_root, 'audio'))  # 'retrieval_val/audio'

    audiocap_id_list = []
    video_youtube_id_list = []

    # prepare training data info
    with open(os.path.join(data_root, 'AudioCaps_retrieval_dataset/retrieval_train.csv'), 'r') as csv_file:
        lines = len(csv_file.readlines())
    with open(os.path.join(data_root, 'AudioCaps_retrieval_dataset/retrieval_train.csv'), 'r') as csvinput:
        with open(os.path.join(data_root, 'audiocaps_audio_train.csv'), 'w') as csvoutput:
            writer = csv.writer(csvoutput)

            for row in tqdm(csv.reader(csvinput, delimiter=','), total=lines):
                if row[0] == "audiocap_id":
                    writer.writerow(row + ['video_path', 'audio_path'])
                else:
                    video_youtube_id = row[1]
                    # audiocap_id = row[0]
                    # find video path through video id
                    for video_name in video_file_names:
                        if video_youtube_id in video_name:
                            video_path = os.path.join(data_root, 'video', video_name)
                            break
                        else:
                            video_path = None
                    # find audio path through video id
                    for audio_name in audio_file_names:
                        if video_youtube_id in audio_name:  #TODO: 一个 youtube_id 有多段音频，开始时间不一样，caption也不同
                            audio_path = os.path.join(data_root, 'audio', audio_name)
                            break
                        else:
                            audio_path = None

                    if video_path is not None and audio_path is not None:
                        # # audiocap_id,youtube_id,start_time,caption,video_path,audio_path
                        # if audiocap_id in audiocap_id_list:
                        #     print(f"audiocap_id={audiocap_id} in audiocap_id_list")
                        #     pdb.set_trace()
                        # if video_youtube_id in video_youtube_id_list:
                        #     print(f"video_youtube_id={video_youtube_id} in video_youtube_id_list")
                        #     pdb.set_trace()
                        # video_youtube_id_list.append(video_youtube_id)
                        # audiocap_id_list.append(audiocap_id)
                        writer.writerow(row + [video_path, audio_path])

    # prepare validation data info
    with open(os.path.join(data_root, 'AudioCaps_retrieval_dataset/retrieval_test.csv'), 'r') as csv_file:
        lines = len(csv_file.readlines())
    with open(os.path.join(data_root, 'AudioCaps_retrieval_dataset/retrieval_test.csv'), 'r') as csvinput:
        with open(os.path.join(data_root, 'audiocaps_audio_test.csv'), 'w') as csvoutput:
            writer = csv.writer(csvoutput)

            for row in tqdm(csv.reader(csvinput, delimiter=','), total=lines):
                if row[0] == "audiocap_id":
                    writer.writerow(row + ['video_path', 'audio_path'])
                else:
                    video_youtube_id = row[1]
                    # audiocap_id = row[0]
                    # find video path through video id
                    for video_name in video_file_names:
                        if video_youtube_id in video_name:
                            video_path = os.path.join(data_root,  'video',  #'retrieval_val/video',
                                                      video_name)
                            break
                        else:
                            video_path = None
                    # find audio path through video id
                    for audio_name in audio_file_names:
                        if video_youtube_id in audio_name:
                            audio_path = os.path.join(data_root, 'audio',  # 'retrieval_val/audio', 
                                                      audio_name)
                            break
                        else:
                            audio_path = None

                    if video_path is not None and audio_path is not None:
                        # audiocap_id,youtube_id,start_time,caption,video_path,audio_path
                        # if audiocap_id in audiocap_id_list:
                        #     print(f"audiocap_id={audiocap_id} in audiocap_id_list")
                        #     pdb.set_trace()
                        # if video_youtube_id in video_youtube_id_list:
                        #     print(f"video_youtube_id={video_youtube_id} in video_youtube_id_list")
                        #     pdb.set_trace()
                        # video_youtube_id_list.append(video_youtube_id)
                        # audiocap_id_list.append(audiocap_id)
                        writer.writerow(row + [video_path, audio_path])