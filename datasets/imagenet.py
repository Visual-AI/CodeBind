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

# load imagenet1k class names
in1k_scene_names = []
scene_dict = {}
with open('.datasets/ImageNet/synset_words.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        key, value = line.split(' ', 1)
        scene_dict[key] = value.strip()
        in1k_scene_names.append(value.strip())

# in1k_scene_names = ['tench, Tinca tinca', 'goldfish, Carassius auratus', 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 
#                     'tiger shark, Galeocerdo cuvieri', 'hammerhead, hammerhead shark', 'electric ray, crampfish, numbfish, torpedo', 'stingray']

class IN1KDataset(Dataset):
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
        self.class_names = in1k_scene_names
        self.scale_factor = scale_factor
        self.modality_pair = modality_pair

        for m in self.modality_pair:
            assert m in ['vision', 'text'], f"Get '{m}'"
        self.has_vision = True if 'vision' in self.modality_pair else False
        self.has_text = True if 'text' in self.modality_pair else False

        self.path_list = list()
        if split in ['train', 'all']:
            csv_path = os.path.join('.datasets/ImageNet', 'imagenet_train.csv')
            # csv_path = os.path.join('.datasets/ImageNet', 'imagenet_debug.csv')
            self.parse_csv(csv_path)
        if split in ['test', 'all']:
            csv_path = os.path.join('.datasets/ImageNet', 'imagenet_test.csv')
            # csv_path = os.path.join('.datasets/ImageNet', 'imagenet_debug.csv')
            self.parse_csv(csv_path)

        # 放大N倍进行训练，将self.parse_csv repeat N份
        if self.scale_factor > 1 and split != 'test':
            self.path_list = self.path_list * int(self.scale_factor)
        print(f'IN1KDataset, {self.modality_pair}, split = {split}, length = {len(self.path_list)}')

    def __len__(self):
        return len(self.path_list)

    def get_classnames(self):
        return self.class_names

    def parse_csv(self, csv_path):
        # If the processed csv file for the ImageNet does not exist,
        # then the dataset needs to be prepared from the raw data.
        if not os.path.exists(csv_path):
            prepare_in1k_dataset(self.root_dir)

        with open(csv_path, 'r', encoding="utf-8") as csv_data:
            for row in csv.reader(csv_data):
                image_path = row[0]  # image
                scene_name = row[1]  # scene

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
    


def prepare_in1k_dataset(in1k_data_root):

    dirname_ls = os.listdir(os.path.join(in1k_data_root, 'train'))

    # prepare training data info
    if not os.path.exists('.datasets/ImageNet/imagenet_train.csv'):
        print("Preparing imagenet image-text dataset csv files for training.")
        with open('.datasets/ImageNet/imagenet_train.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput)
            for dirname in tqdm(dirname_ls):
                # get image label through dirname
                # scene_name = scene_dict[id_dirname_dict[dirname]]
                scene_name = scene_dict[dirname]
                image_names = os.listdir(os.path.join(in1k_data_root, 'train', dirname))
                for image_name in image_names:
                    image_path = os.path.join(in1k_data_root, 'train', dirname, image_name)

                    writer.writerow([image_path, scene_name])

    # prepare validation data info
    if not os.path.exists('.datasets/ImageNet/imagenet_test.csv'):
        print("Preparing imagenet image-text dataset csv files for validation.")
        with open('.datasets/ImageNet/imagenet_test.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput)
            for dirname in tqdm(dirname_ls):
                # get image label through dirname
                # scene_name = scene_dict[id_dirname_dict[dirname]]
                scene_name = scene_dict[dirname]
                image_names = os.listdir(os.path.join(in1k_data_root, 'val', dirname))
                for image_name in image_names:
                    image_path = os.path.join(in1k_data_root, 'val', dirname, image_name)

                    writer.writerow([image_path, scene_name])
