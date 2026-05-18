# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind
# @Reference     : ImageBind, Meta Platforms, Inc. and affiliates.
import os
import csv
import random
import imageio
import numpy as np

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn

from typing import Optional, Callable
from torch.utils.data import Dataset
from torchvision import transforms
from models.codebind_model import ModalityType
from datasets import data
from datasets.data import DepthNorm
from datasets.data_transform_rgbd import load_and_transform_rgbd_data


nyu_default_9_scene_class = [
    'bathroom',         'bedroom',      'bookstore',        'classroom',        'dining room',
    'home office',      'kitchen',      'living room',      'office']

nyu_other_18_semantic_class = [
    'basement',         'cafe',         'computer lab',     'conference room',  'dinette', 
    'exercise room',    'foyer',        'furniture store',  'home storage',     'indoor balcony', 
    'laundry room',     'office kitchen', 'playroom',       'printer room',     'reception room', 
    'student lounge',   'study',        'study room']

nyu_class_names = nyu_default_9_scene_class + nyu_other_18_semantic_class



class NYURGBD_Dataset(Dataset):
    def __init__(self, root_dir: str, 
                 dataset_name: str,
                 mode: str = None, # mode is used for different data augmentation in train and test
                 modality_pair: list = ['vision', 'depth'],
                 split: str = 'train', 
                 scale_factor = 1,
                 device: str = 'cpu', 
                 text_template = None):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.device = device
        self.split = split
        self.class_names = nyu_class_names
        self.scale_factor = scale_factor
        self.modality_pair = modality_pair
        text_template_name = text_template if text_template is not None else 'imagenet'
        self.text_template = data.text_template_dict.get(text_template_name)

        for m in self.modality_pair:
            assert m in ['vision', 'depth', 'text']
        self.has_vision = True if 'vision' in self.modality_pair else False
        self.has_depth = True if 'depth' in self.modality_pair else False
        self.has_text = True if 'text' in self.modality_pair else False

        self.path_list = list()
        if split in ['train', 'all']:
            csv_path = os.path.join(root_dir, 'nyu_depth_v2_train.csv')
            self.parse_csv(csv_path)
        if split in ['test', 'all']:
            # csv_path = os.path.join(root_dir, 'nyu_depth_v2_test.csv')
            csv_path = os.path.join(root_dir, 'nyu_depth_v2_test_sunpath.csv') # use in train sun and eval nyu (labels from nyu and data paths from sun)
            # csv_path = os.path.join(root_dir, 'sun_debug.csv')
            # csv_path = '.datasets/nyu_depth_v2_test_sunpath.csv'
            self.parse_csv(csv_path)
        
        # 放大N倍进行训练，将self.parse_csv repeat N份
        if self.scale_factor > 1 and split != 'test':
            self.path_list = self.path_list * int(self.scale_factor)
        print(f'NYURGBD_Dataset, {self.modality_pair}, split = {split}, length = {len(self.path_list)}, text_template={text_template_name}')


    def __len__(self):
        return len(self.path_list)
    
    def get_classnames(self):
        return self.class_names
    
    def parse_csv(self, csv_path):
        # If the processed csv file for the NYU-depth-v2 dataset does not exist, 
        # then the dataset needs to be prepared from the raw data.
        if not os.path.exists(csv_path):
            print("File not exists:", csv_path)
            prepare_nyu_dataset(self.root_dir, data_types = ['image', 'depth', 'csv'])
        
        with open(csv_path, 'r', encoding="utf-8-sig") as csv_data:
            reader = csv.reader(csv_data, delimiter=',')
            for row in reader:
                rgb_path = row[0]  # RGB image
                dep_path = row[1]  # depth
                # rgb_path = os.path.join(self.root_dir, row[0])  # RGB image
                # dep_path = os.path.join(self.root_dir, row[1])  # depth
                scene_name = row[2]
                self.path_list.append([rgb_path, dep_path, scene_name])
    
    def __getitem__(self, index):
        scene_name = self.path_list[index][2]
        output_dict = {'label': scene_name}
        if self.has_vision and self.has_depth:
            _rgbd = load_and_transform_rgbd_data(image_paths=[self.path_list[index][0]],
                                                 depth_paths=[self.path_list[index][1]], camera_types=['kv1'],
                                                 focal_lengths=[518.857901], device=self.device, mode=self.mode)
            output_dict.update({ModalityType.VISION: _rgbd[0, 0:3, ...]})
            output_dict.update({ModalityType.DEPTH: _rgbd[0, 3, ...].unsqueeze(0)})
        else:
            if self.has_vision:
                _rgb = load_and_transform_rgbd_data(image_paths=[self.path_list[index][0]], device=self.device,
                                                    mode=self.mode)
                # _rgb = data.load_and_transform_vision_data([self.path_list[index][0]], self.device, to_tensor=True)
                output_dict.update({ModalityType.VISION: _rgb[0]})

            if self.has_depth:
                _dep = load_and_transform_rgbd_data(depth_paths=[self.path_list[index][1]], camera_types=['kv1'],
                                                    focal_lengths=[518.857901], device=self.device, mode=self.mode)
                # _dep = data.load_and_transform_depth_data([self.path_list[index][1]], self.device, to_tensor=True)
                output_dict.update({ModalityType.DEPTH: _dep[0]})

        if self.has_text:
            # t = random.choice(data.imagenet_templates)
            t = random.choice(self.text_template)
            class_text = t.format(scene_name)
            _text = data.load_and_transform_text([class_text], self.device)
            output_dict.update({ModalityType.TEXT: _text[0]})
        return output_dict



def prepare_nyu_dataset(NYU_data_root, data_types: list = ['image', 'depth', 'scene_type']):
    """ Extract RGB-D images and scene types from the matlab mat file.
    """
    print("Prepare NYU-depth-v2 ...")
    nyu_depth_v2_labeled = os.path.join(NYU_data_root, "nyu_depth_v2_labeled.mat")
    splits = os.path.join(NYU_data_root, "splits.mat")

    assert len(data_types) > 0, "data_types is empty."
    for i in data_types:
        data_list = ['image', 'depth', 'scene_type', 'depth_gray', 'csv']
        assert i in data_list, f"data_types must in {data_list}"

    from scipy.io import loadmat
    # train/test split
    print("Loading", splits)
    tt_split = loadmat(splits)
    test_idx_list = tt_split.get("testNdxs")  # start form 1
    test_idx_list = test_idx_list.flatten().tolist()
    train_idx_list = tt_split.get("trainNdxs")
    train_idx_list = train_idx_list.flatten().tolist()

    import h5py
    print("Loading", nyu_depth_v2_labeled)
    f = h5py.File(nyu_depth_v2_labeled)

    img_path_converted='nyu_images'
    rgb_path_list = []
    if 'image' in data_types:
        print("loading images")
        images=f["images"]
        images=np.array(images) # shape (1449, 3, 640, 480)
        if not os.path.isdir(os.path.join(NYU_data_root, img_path_converted)):
            os.makedirs(os.path.join(NYU_data_root, img_path_converted))
        data_length = len(images)
    
    depth_path_converted = 'nyu_depths'
    depth_gray_path_converted= 'nyu_depths_gray'
    depth_path_list = []
    if ('depth' in data_types) or ('depth_gray' in data_types):
        print("loading depths")
        depths=f["depths"]
        depths=np.array(depths)  # shape (1449, 640, 480), d_type float32, depth in m
        depths = depths.transpose((0,2,1)) # shape (1449, 480, 640)

        # print(depths.shape)
        # print(depths.max())  # 9.99547
        # print(depths.min())  # 0.7132995

        if 'depth' in data_types:
            depths_mm = depths * 1000 * 8  # m to mm， shift 3
            if not os.path.isdir(os.path.join(NYU_data_root, depth_path_converted)):
                os.makedirs(os.path.join(NYU_data_root, depth_path_converted))

        if 'depth_gray' in data_types:
            max = depths.max()
            depths_gray = depths / max * 255  # uint8 255, uint16 65535
            if not os.path.isdir(os.path.join(NYU_data_root, depth_gray_path_converted)):
                os.makedirs(os.path.join(NYU_data_root, depth_gray_path_converted))

        data_length = len(depths)

    if 'scene_type' in data_types or 'csv' in data_types:
        print("loading sceneTypes")
        sceneTypes=f["sceneTypes"]
        sceneTypes=np.array(sceneTypes)[0]  # 1449
        scene_type_list = []
        data_length = len(sceneTypes)

    rgb_d_scene_list = []

    cnt = 0
    for i in tqdm(range(data_length)):
        cnt += 1
        basename = f"{i+1:04d}"
        # -------
        rgb_img_path = os.path.join(NYU_data_root, img_path_converted, basename+'.jpg')
        rgb_path_list.append(rgb_img_path)

        depth_img_path = os.path.join(NYU_data_root, depth_path_converted, basename+'.png')
        depth_path_list.append(depth_img_path)
        
        if 'image' in data_types:
            a = images[i]
            r = Image.fromarray(a[0]).convert('L')
            g = Image.fromarray(a[1]).convert('L')
            b = Image.fromarray(a[2]).convert('L')
            rgb_img = Image.merge("RGB", (r, g, b))
            rgb_img = rgb_img.transpose(Image.ROTATE_270)
            rgb_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)

            rgb_img.save(rgb_img_path, optimize=False)
        
        if 'depth' in data_types:
            dep_img = Image.fromarray(np.uint16(depths_mm[i]))  # uint16
            # dep_img = dep_img.transpose(Image.FLIP_LEFT_RIGHT)
            dep_img.save(depth_img_path, 'PNG', optimize=False)

        if 'depth_gray' in data_types:
            dep_gray= Image.fromarray(np.uint8(depths_gray[i]))  # uint8
            # dep_gray = dep_gray.transpose(Image.FLIP_LEFT_RIGHT)
            depth_gray_path = os.path.join(depth_gray_path_converted, basename+'.png')
            dep_gray.save(os.path.join(NYU_data_root, depth_gray_path), 'PNG', optimize=True)

        if 'scene_type' in data_types or 'csv' in data_types:
            obj_ref = sceneTypes[i]     # <HDF5 object reference>
            obj_data = f[obj_ref]       # <HDF5 dataset "yx": shape (7, 1), type "<u2"> 
            # Each object is an array of 2-byte integers (<u2 is the NumPy little-endian unsigned 2-byte integer datatype)
            # c = "".join(chr(k) for k in obj_data[0, :])
            c_list = ["".join(chr(k) for k in obj_data[j, :]) for j in range(len(obj_data))]
            scene_type_str = "".join(c_list)
            scene_type_str = scene_type_str.replace("_", " ")
            scene_type_str = scene_type_str.replace("excercise", "exercise")
            scene_type_list.append(scene_type_str)

        # if ('image' in data_types) and ('depth' in data_types) and ('scene_type' in data_types):
        if 'csv' in data_types:
            rgb_d_scene_list.append((rgb_img_path, depth_img_path, scene_type_str))

    # save data splits to csv file
    if 'scene_type' in data_types:
        scene_type_set = set(scene_type_list)
        print(f"Get {len(scene_type_set)} scene_type: ", scene_type_set)
        with open(os.path.join(NYU_data_root, 'scene_type.csv'), 'w', newline='\n') as csv_of:
            spamwriter = csv.writer(csv_of, delimiter=',')
            for i in scene_type_set:
                spamwriter.writerow([i])

    # if ('image' in data_types) and ('depth' in data_types) and ('scene_type' in data_types):
    if 'csv' in data_types:
        test_csv_list = []
        train_csv_list = []
        for idx, row in enumerate(rgb_d_scene_list):
            # print(row)
            if idx+1 in test_idx_list:
                test_csv_list.append(row)
            else:
                train_csv_list.append(row)
                
        with open(os.path.join(NYU_data_root, 'nyu_depth_v2_test.csv'), 'w', newline='\n') as csv_of:
            spamwriter = csv.writer(csv_of, delimiter=',')
            for i in test_csv_list:
                spamwriter.writerow(i)
        with open(os.path.join(NYU_data_root, 'nyu_depth_v2_train.csv'), 'w', newline='\n') as csv_of:
            spamwriter = csv.writer(csv_of, delimiter=',')
            for i in train_csv_list:
                spamwriter.writerow(i)


# prepare_nyu_dataset(
#     NYU_data_root = '/home/jieli/datasets/RGBD-NYU-v2',
#     # data_types = ['scene_type'],
#     # data_types = ['image', 'depth', 'scene_type', 'depth_gray']
#     data_types = ['image', 'depth', 'csv']
#     )