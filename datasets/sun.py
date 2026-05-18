# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind
# @Reference     : ImageBind, Meta Platforms, Inc. and affiliates.
import os
import csv
import random

from typing import Optional, Callable

import numpy as np
from glob import glob
from scipy.io import loadmat
from torch.utils.data import Dataset

from datasets import data
from models.codebind_model import ModalityType
from datasets.nyu import nyu_class_names
from datasets.data_transform_rgbd import load_and_transform_rgbd_data


scene_name_mapping_SUN2NYU = {
    # raw_nyu_scene_names_in_SUN --> nyu-depth-v2
    'cafeteria':        'cafe',
    'computer room':    'computer lab',
    'gym':              'exercise room',
    'laundromat':       'laundry room',    # 自助洗衣店
    'lobby':            'foyer',
    'storage room':     'home storage',
    # 'study space':      'study room',
    'recreation room':  'student lounge',   # 60  #  ?- >'lounge'
    # 
    'coffee room':      'cafe',
    'dining area':      'dining room',
    # 'office dining':    'dining room',
    'reception':        'reception room',
    'rest space':       'student lounge',   # 924
    'home':             'living room',      # 6
    'discussion area':  'conference room'   # 201
}

scene_name_mapping_eval = {
    'computer room':    'computer lab',
    # 'dining area':      'dining room',
}

# 44 class names
sun_raw_scene_names = [
    'basement',     'bathroom',     'bedroom',      'bookstore',        'cafe', 
    'classroom',    'coffee room',  'computer lab', 'conference room',  'corridor', 
    'dancing room', 'dinette',      'dining area',  'dining room',      'discussion area', 
    'exercise room','exhibition',   'foyer',        'furniture store',  'home', 
    'home office',  'home storage', 'hotel room',   'indoor balcony',   'kitchen', 
    'lab',          'laundry room', 'lecture theatre', 'library',       'living room', 
    'mail room',    'music room',   'office',       'office dining',    'office kitchen', 
    'playroom',     'printer room', 'reception',    'reception room',   'rest space', 
    'stairs',       'student lounge', 'study',      'study room']

# 19 class names
sun_19_scene_names = [
    "bathroom", "bedroom", "classroom", "computer room", "conference room", "corridor",
    "dining area", "dining room", "discussion area", "furniture store", "home office",
    "kitchen", "lab", "lecture theatre", "library", "living room", "office",
    "rest space", "study space"]



# sun_class_names_mapped = [scene_name_mapping_SUN2NYU.get(i, i) for i in sun_raw_scene_names]
# sun_class_names = list(set(sun_class_names_mapped))
sun_class_names = list(set(sun_19_scene_names))


class SUNRGBD_Dataset(Dataset):
    def __init__(self, root_dir: str, 
                 dataset_name: str,
                 mode: str = None, # mode is used for different data augmentation in train and test
                 modality_pair: list = ['vision', 'depth'],
                 split: str = 'train', 
                 scale_factor = 1,
                 device: str = 'cpu',
                 text_template = None,
                 data_version = ''):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.device = device
        self.split = split
        self.class_names = sun_class_names
        self.scale_factor = scale_factor
        self.modality_pair = modality_pair
        text_template_name = text_template if text_template is not None else 'imagenet'
        self.text_template = data.text_template_dict.get(text_template_name)

        for m in self.modality_pair:
            assert m in ['vision', 'depth', 'text'], f"Get '{m}'"
        self.has_vision = True if 'vision' in self.modality_pair else False
        self.has_depth = True if 'depth' in self.modality_pair else False
        self.has_text = True if 'text' in self.modality_pair else False

        self.path_list = list()
        if split in ['train', 'all']:
            csv_path = os.path.join(root_dir, f'sun_rgbd_train{data_version}.csv')
            # csv_path = os.path.join(root_dir, 'nyu_depth_v2_train.csv')
            # csv_path = os.path.join(root_dir, 'sun_debug.csv')
            self.parse_csv(csv_path)
        if split in ['test', 'all']:
            csv_path = os.path.join(root_dir, f'sun_rgbd_test{data_version}.csv')
            # csv_path = os.path.join(root_dir, 'nyu_depth_v2_test.csv')
            # csv_path = os.path.join(root_dir, 'nyu_depth_v2_test_sunpath.csv')
            # csv_path = os.path.join(root_dir, 'sun_debug.csv')
            self.parse_csv(csv_path)
        print(csv_path)
        # 放大N倍进行训练，将self.parse_csv repeat N份
        if self.scale_factor > 1 and split != 'test':
            self.path_list = self.path_list * int(self.scale_factor)
        print(f'SUNRGBD_Dataset, {self.modality_pair}, split = {split}, length = {len(self.path_list)}, text_template={text_template_name}')


    def __len__(self):
        return len(self.path_list)

    def get_classnames(self):
        return self.class_names
    
    def parse_csv(self, csv_path):
        # If the processed csv file for the NYU-depth-v2 dataset does not exist, 
        # then the dataset needs to be prepared from the raw data.
        if not os.path.exists(csv_path):
            prepare_sun_dataset(self.root_dir)
        
        with open(csv_path, 'r', encoding="utf-8") as csv_data:
            reader = csv.reader(csv_data, delimiter=',')
            for row in reader:
                rgb_path = row[0]  # RGB image
                dep_path = row[1]  # depth
                scene_name = row[2]  # scene
                camera_type = row[3]  # camera type in ['kv1', 'kv2', 'realsense', 'xtion']
                focal_length = row[4]  # camera focal length

                # mapping
                # scene_name = scene_name_mapping_SUN2NYU.get(scene_name, scene_name)

                self.path_list.append([rgb_path, dep_path, scene_name, camera_type, focal_length])
    
    def __getitem__(self, index):
        scene_name = self.path_list[index][2]
        output_dict = {'label': scene_name}
        if self.has_vision and self.has_depth:
            _rgbd = load_and_transform_rgbd_data(image_paths=[self.path_list[index][0]],
                                                 depth_paths=[self.path_list[index][1]],
                                                 camera_types=[self.path_list[index][3]],
                                                 focal_lengths=[self.path_list[index][4]],
                                                 device=self.device, mode=self.mode)
            output_dict.update({ModalityType.VISION: _rgbd[0, 0:3, ...]})
            output_dict.update({ModalityType.DEPTH: _rgbd[0, 3, ...].unsqueeze(0)})
        else:
            if self.has_vision:
                # _rgb = data.load_and_transform_vision_data([self.path_list[index][0]], self.device, to_tensor=True)
                _rgb = load_and_transform_rgbd_data(image_paths=[self.path_list[index][0]], device=self.device,
                                                    mode=self.mode)
                output_dict.update({ModalityType.VISION: _rgb[0]})

            if self.has_depth:
                # _dep = data.load_and_transform_depth_data([self.path_list[index][1]], self.device, to_tensor=True)
                _dep = load_and_transform_rgbd_data(depth_paths=[self.path_list[index][1]],
                                                    camera_types=[self.path_list[index][3]],
                                                    focal_lengths=[self.path_list[index][4]], device=self.device,
                                                    mode=self.mode)
                output_dict.update({ModalityType.DEPTH: _dep[0]})

        if self.has_text:
            t = random.choice(self.text_template)
            class_text = t.format(scene_name)
            _text = data.load_and_transform_text([class_text], self.device)
            output_dict.update({ModalityType.TEXT: _text[0]})
        return output_dict
    

def get_train_test_split(splits):
    # train/test split
    print("Loading", splits)
    tt_split = loadmat(splits)

    alltrain = tt_split.get("alltrain")
    alltrain = alltrain[0]   # ndarray of shape (1, 5285) --> (5285)
    alltrain = np.concatenate(alltrain, axis=0)

    alltest = tt_split.get("alltest")
    alltest = alltest[0]   # ndarray of shape (1, 5050) --> (5050)
    alltest = np.concatenate(alltest, axis=0)

    # trainvalsplit = tt_split.get("trainvalsplit")
    # trainvalsplit = trainvalsplit[0][0]
    # train = trainvalsplit[0]  # (2666, 1)
    # val = trainvalsplit[1]  # (2619, 1)
    # train = np.concatenate(train[:, 0], axis=0)
    # val = np.concatenate(val[:, 0], axis=0)

    # print(alltrain[0])
    # /n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/000065_2014-05-16_20-14-38_260595134347_rgbf000121-resize
    return alltrain, alltest


def get_path(SUN_data_root, raw_split, save_csv_path=None, mapping=None, scene_list=[]):
    image_depth_scene = []
    if mapping:
        map_list = list(mapping.keys())
    for s in raw_split:
        new_s = s.replace('/n/fs/sun3d/data', SUN_data_root)
        # get image path
        p_image = glob(os.path.join(new_s, 'image/*.jpg'))[0]
        # print('p_image:', p_image)
        # get depth path
        p_depth = glob(os.path.join(new_s, 'depth_bfx/*.png'))[0]
        # print('p_image:', p_depth)
        # get scene name
        p_scene = os.path.join(new_s, 'scene.txt') 
        f_scene = open(p_scene, "r")
        scene = f_scene.readlines()[0]
        scene = scene.replace("_", " ")

        # mapping
        if mapping and (scene in map_list):
            scene = mapping.get(scene)

        # filter scene within 19 classes
        if len(scene_list) and (scene not in scene_list):
            continue

        # get camera type and focal length in intrinsics
        if 'SUNRGBD/kv1' in new_s: camera_type = 'kv1'
        elif 'SUNRGBD/kv2' in new_s: camera_type = 'kv2'
        elif 'SUNRGBD/realsense' in new_s: camera_type = 'realsense'
        elif 'SUNRGBD/xtion' in new_s: camera_type = 'xtion'

        p_intrinsics = os.path.join(new_s, 'intrinsics.txt')
        with open(p_intrinsics, 'r') as fh:
            lines = fh.readlines()
            focal_length = float(lines[0].strip().split()[0])

        image_depth_scene.append([p_image, p_depth, scene, camera_type, focal_length])

    if save_csv_path:
        print(f"Get {len(image_depth_scene)} samples, saving to csv:", save_csv_path)
        with open(save_csv_path, 'w', newline='\n') as csv_of:
            spamwriter = csv.writer(csv_of, delimiter=',')
            for i in image_depth_scene:
                spamwriter.writerow(i)
    return image_depth_scene


def prepare_sun_dataset(SUN_data_root):
    print("Prepare SUN-RGBD ...")
    split_mat_path = os.path.join(SUN_data_root, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
    train_split, test_split = get_train_test_split(split_mat_path)

    sun_csv_train_path = os.path.join(SUN_data_root, 'sun_rgbd_train.csv')
    sun_csv_test_path = os.path.join(SUN_data_root, 'sun_rgbd_test.csv')

    get_path(SUN_data_root, raw_split=train_split, save_csv_path=sun_csv_train_path, mapping=None)
    get_path(SUN_data_root, raw_split=test_split, save_csv_path=sun_csv_test_path, mapping=None)

    # sun_csv_train_path = os.path.join(SUN_data_root, 'sun_rgbd_train_v2.csv')
    # sun_csv_test_path = os.path.join(SUN_data_root, 'sun_rgbd_test_v2.csv')

    # # mapping: scene_name_mapping_SUN2NYU
    # get_path(SUN_data_root, raw_split=train_split, save_csv_path=sun_csv_train_path, mapping=scene_name_mapping_SUN2NYU)
    # get_path(SUN_data_root, raw_split=test_split, save_csv_path=sun_csv_test_path, mapping=scene_name_mapping_eval, 
    #          scene_list=sun_19_scene_names)

    nyu_train_split = [i for i in train_split if "kv1/NYUdata" in i]
    nyu_test_split = [i for i in test_split if "kv1/NYUdata" in i]

    nyu_csv_train_path = os.path.join(SUN_data_root, 'nyu_depth_v2_train.csv')
    nyu_csv_test_path = os.path.join(SUN_data_root, 'nyu_depth_v2_test.csv')
    get_path(SUN_data_root, raw_split=nyu_train_split, save_csv_path=nyu_csv_train_path, mapping=None)
    get_path(SUN_data_root, raw_split=nyu_test_split, save_csv_path=nyu_csv_test_path, mapping=None)

