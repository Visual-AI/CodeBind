# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind
# @Reference     : ImageBind, Meta Platforms, Inc. and affiliates.
import os
import math
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import random
from PIL import Image
from PIL import ImageDraw
from torch.utils.data import Dataset

from models.codebind_model import ModalityType
from datasets import data
from datasets.data_transform_rgbd import load_and_transform_thermal_data_flir

flir_scene_names = ['car', 'person', 'sign', 'motor', 'truck', 'light', 'bike', 'hydrant', 'dog', 'other vehicle']
# flir_scene_names = ['car', 'person', 'sign', 'motor', 'truck']

class FLIRDataset(Dataset):
    def __init__(self, root_dir: str,
                 dataset_name: str,
                 modality_pair: list = ['vision', 'thermal'],
                 mode: str = None,  # mode is used for different data augmentation in train and test
                 split: str = 'train',
                 scale_factor=1,
                 device: str = 'cpu'):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.device = device
        self.split = split
        self.mode = mode
        self.class_names = flir_scene_names
        self.scale_factor = scale_factor
        self.modality_pair = modality_pair

        for m in self.modality_pair:
            assert m in ['vision', 'thermal', 'text'], f"Get '{m}'"
        self.has_vision = True if 'vision' in self.modality_pair else False
        self.has_thermal = True if 'thermal' in self.modality_pair else False
        self.has_text = True if 'text' in self.modality_pair else False

        self.path_list = list()
        csv_path = os.path.join('.datasets/FLIR_v2', f'flir_thermal_{split}.csv')  # split is 'train' or 'test'
        self.parse_csv(csv_path)

        # 放大N倍进行训练，将self.parse_csv repeat N份
        if self.scale_factor > 1 and split != 'test':
            self.path_list = self.path_list * int(self.scale_factor)
        print(f'FLIRDataset, {self.modality_pair}, split = {split}, length = {len(self.path_list)}')

    def __len__(self):
        return len(self.path_list)

    def get_classnames(self):
        return self.class_names

    def parse_csv(self, csv_path):
        # If the processed csv file for the LLVIP does not exist,
        # then the dataset needs to be prepared from the raw data.
        if not os.path.exists(csv_path):
            prepare_flir_dataset(self.root_dir)
        
        with open(csv_path, 'r', encoding="utf-8") as csv_data:
            for row in csv.reader(csv_data, delimiter=';'):
                image_path = os.path.join('.datasets/FLIR_v2/video_rgb_test/data', row[0])  # image
                thermal_path = os.path.join('.datasets/FLIR_v2/video_thermal_test/data', row[1])  # thermal
                scene_name = row[2]  # category
                rgb_bbox = list(map(int, row[3].split(',')))  # rgb bbox
                thermal_bbox = list(map(int, row[4].split(',')))  # thermal bbox

                # fliter out some categories
                if scene_name not in self.class_names:
                    continue
                # fliter out small bboxes
                _th = 10
                x1, y1, x2, y2 = rgb_bbox
                if x2 - x1 <= _th or y2 - y1 <= _th:
                    # print(f"image:{image_path}. RGB bbox: {x2}-{x1}, {y2}-{y1} must >= {_th}")
                    continue
                x1, y1, x2, y2 = thermal_bbox
                if x2 - x1 <= _th or y2 - y1 <= _th:
                    # print(f"image:{image_path}. Thermal bbox: {x2}-{x1}, {y2}-{y1} must >= {_th}")
                    continue
                self.path_list.append([image_path, thermal_path, scene_name, rgb_bbox, thermal_bbox])


    def __getitem__(self, index):
        scene_name = self.path_list[index][2]
        output_dict = {'label': scene_name}
        if self.has_vision and self.has_thermal:
            # if self.split == 'train':  # len(self.path_list[index]) == 2:
            #     _rgbt = load_and_transform_thermal_data(image_paths=[self.path_list[index][0]],
            #                                         thermal_paths=[self.path_list[index][1]],
            #                                         device=self.device, mode=self.mode)
            #     # _rgbd: [1, 4, H, W]
            #     output_dict.update({ModalityType.VISION: _rgb[0]})
            #     output_dict.update({ModalityType.THERMAL: _thermal[0]})
            
            # # test during traing
            # elif self.split == 'test':  # len(self.path_list[index]) == 4: # get cropped image and thermal data
            _rgbt = load_and_transform_thermal_data_flir(image_paths=[self.path_list[index][0]],
                                                    thermal_paths=[self.path_list[index][1]],
                                                    image_bboxes=[self.path_list[index][3]],
                                                    thermal_bboxes=[self.path_list[index][4]],
                                                    device=self.device, mode=self.mode)
            # _rgbt: [1, 4, H, W]
            
            # true_or_rand = 0 if index%2 else 1 
            # output_dict.update({ModalityType.VISION: _rgbd[0, true_or_rand, 0:3, ...]})  # [3, H, W]
            # output_dict.update({ModalityType.THERMAL: _rgbd[0, true_or_rand, 3, ...].unsqueeze(0)}) # [1, H, W]

            output_dict.update({ModalityType.VISION: _rgbt[0, 0:3, ...]})  # [3, H, W]
            output_dict.update({ModalityType.THERMAL: _rgbt[0, 3, ...].unsqueeze(0)}) # [1, H, W]

        elif self.has_vision:
            _rgb = load_and_transform_thermal_data_flir(image_paths=[self.path_list[index][0]],
                                                   image_bboxes=[self.path_list[index][3]],
                                                   device=self.device, mode=self.mode)
            output_dict.update({ModalityType.VISION: _rgb[0]})

        elif self.has_thermal:
            _thermal = load_and_transform_thermal_data_flir(thermal_paths=[self.path_list[index][1]],
                                                       thermal_bboxes=[self.path_list[index][4]],
                                                       device=self.device, mode=self.mode)
            output_dict.update({ModalityType.THERMAL: _thermal[0]})
        
        if self.has_text:
            t = random.choice(data.imagenet_templates)
            class_text = [t.format(scene_name_i) for scene_name_i in scene_name.split(';')]
            _text = data.load_and_transform_text([class_text], self.device)
            output_dict.update({ModalityType.TEXT: _text[0]})

        return output_dict


def obtain_img_info(anns_file):
    with open(anns_file, 'r') as f:
        anns = json.load(f)
        
    bboxes = []
    category_ids = []
    image_ids = []
    img_path_dict = {}
    img_size_dict = {}
    category_dict = {}
    for ann in anns['annotations']:           
        bboxes.append(ann['bbox'])
        category_ids.append(ann['category_id'])
        image_ids.append(ann['image_id'])
    for img in anns['images']:
        img_path_dict[img['id']] = img['file_name'].split('/')[-1]
        img_size_dict[img['id']] = [img['width'], img['height']]
    for cat in anns['categories']:
        category_dict[cat['id']] = cat['name']

    unique_images_ids = []
    img_info = []
    img_info_dict = None
    for bbox_i, category_id_i, image_id_i in zip(bboxes, category_ids, image_ids):
        if image_id_i not in unique_images_ids:
            if img_info_dict is not None:
                img_info.append(img_info_dict)
            unique_images_ids.append(image_id_i)
            img_info_dict = {'path': img_path_dict[image_id_i], 'size': img_size_dict[image_id_i], 
                            'scene_name': [category_dict[category_id_i]], 'bbox': [bbox_i]}
        else:
            img_info_dict['scene_name'].append(category_dict[category_id_i])
            img_info_dict['bbox'].append(bbox_i)
    
    return img_info

def change_bbox_format(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def prepare_flir_dataset(root_dir, dist_thred=0.1):

    img_info_rgb = obtain_img_info(os.path.join(root_dir, 'video_rgb_test/coco.json'))
    img_info_thermal = obtain_img_info(os.path.join(root_dir, 'video_thermal_test/coco.json'))
    
    with open(os.path.join(root_dir, 'rgb_to_thermal_vid_map.json'), 'r') as f:
        vid_map_dict = json.load(f)

    # find the matched rgb and thermal image pairs
    img_path_rgb = [img_info_i['path'] for img_info_i in img_info_rgb]
    img_path_thermal = [img_info_i['path'] for img_info_i in img_info_thermal]

    rgb_to_thermal_mapidx= {}
    for i, path_rgb in enumerate(img_path_rgb):  
        for j, path_thermal in enumerate(img_path_thermal):  
                if vid_map_dict[path_rgb] == path_thermal:  
                    rgb_to_thermal_mapidx.update({i: j})  
                    break  

    # get the matching bbox from rgb to thermal
    with open(os.path.join(root_dir, 'rgb_thermal_info.csv'), "w") as f:
        writer = csv.writer(f, delimiter=";")
        for idx, rgb_info in enumerate(img_info_rgb):
            if rgb_to_thermal_mapidx.get(idx) is None:
                continue
            thermal_info = img_info_thermal[rgb_to_thermal_mapidx[idx]]
            rgb_size = rgb_info['size']
            thermal_size = thermal_info['size']
            select_thermal_bbox = {}
            for category, bbox in zip(thermal_info['scene_name'], thermal_info['bbox']):
                if select_thermal_bbox.get(category) is not None:
                    select_thermal_bbox[category].append(bbox)
                else:
                    select_thermal_bbox[category] = [bbox]
            for category_i, bbox_i in zip(rgb_info['scene_name'], rgb_info['bbox']):
                select_bbox = select_thermal_bbox.get(category_i)
                if select_bbox is not None:
                    rgb_bbox_center = [(bbox_i[0] + bbox_i[2] / 2) / rgb_size[0], (bbox_i[1] + bbox_i[3] / 2) / rgb_size[1]]
                    thermal_bbox_centers = [[(bbox[0] + bbox[2] / 2) / thermal_size[0], (bbox[1] + bbox[3] / 2) / thermal_size[1]] for bbox in select_bbox]
                    rgb_bbox_area = bbox_i[2] * bbox_i[3] / (rgb_size[0] * rgb_size[1])
                    thermal_bbox_areas = [bbox[2] * bbox[3] / (thermal_size[0] * thermal_size[1]) for bbox in select_bbox]
                    min_score = 2
                    for i in range(len(select_bbox)):
                        thermal_bbox_center = thermal_bbox_centers[i]
                        thermal_bbox_area = thermal_bbox_areas[i]
                        bbox_center_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb_bbox_center, thermal_bbox_center)))
                        bbox_area_dist = abs(math.sqrt(rgb_bbox_area) - math.sqrt(thermal_bbox_area))
                        score = bbox_center_dist + 10 * bbox_area_dist
                        if score < min_score:
                            min_score = score
                            min_idx = i
                    if min_score < dist_thred:
                        bbox_i_pair = select_bbox[min_idx]
                        select_thermal_bbox[category_i].remove(bbox_i_pair)
                        # write to csv
                        bbox_i = change_bbox_format(bbox_i)
                        bbox_i_pair = change_bbox_format(bbox_i_pair)
                        bbox_i = ','.join(str(num) for num in bbox_i)
                        bbox_i_pair = ','.join(str(num) for num in bbox_i_pair)               
                        writer.writerow([rgb_info['path'], thermal_info['path'], category_i, bbox_i, bbox_i_pair])

    # train_test_split
    data = pd.read_csv(os.path.join(root_dir, 'rgb_thermal_info.csv'), delimiter=';', 
                       names=['rgb_path', 'thermal_path', 'scene_name', 'rgb_bbox', 'thermal_bbox'])
    train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['scene_name'], random_state=42)
    train_data.to_csv(os.path.join(root_dir, 'flir_thermal_train.csv'), sep=';', index=False, header=False)
    test_data.to_csv(os.path.join(root_dir, 'flir_thermal_test.csv'), sep=';', index=False, header=False)


# # %%
# def obtain_path_list():
#     csv_path = '/disk2/zychen/imagebind/.datasets/FLIR_v2/rgb_thermal_info.csv'
#     path_list = []
#     with open(csv_path, 'r', encoding="utf-8") as csv_data:
#         for row in csv.reader(csv_data, delimiter=';'):
#             image_path = os.path.join('.datasets/FLIR_v2/video_rgb_test/data', row[0])  # image
#             thermal_path = os.path.join('.datasets/FLIR_v2/video_thermal_test/data', row[1])  # thermal
#             scene_name = row[2]  # category
#             rgb_bbox = list(map(int, row[3].split(',')))  # rgb bbox
#             thermal_bbox = list(map(int, row[4].split(',')))  # thermal bbox

#             _th = 20
#             x1, y1, x2, y2 = rgb_bbox
#             if x2 - x1 <= _th or y2 - y1 <= _th:
#                 # print(f"image:{image_path}. RGB bbox: {x2}-{x1}, {y2}-{y1} must >= {_th}")
#                 continue
#             x1, y1, x2, y2 = thermal_bbox
#             if x2 - x1 <= _th or y2 - y1 <= _th:
#                 # print(f"image:{image_path}. Thermal bbox: {x2}-{x1}, {y2}-{y1} must >= {_th}")
#                 continue
#             path_list.append([image_path, thermal_path, scene_name, rgb_bbox, thermal_bbox])

#     all_scene = [path[2] for path in path_list]
#     from collections import Counter
#     print(Counter(all_scene))

#     return all_scene, path_list

# def bbox_visualization(all_scene, path_list):
#     # visualize some bounding boxes
#     # car, person, sign, motor, truck, light, bike, hydrant, dog
#     select_path_list = [i for i, scene_name in enumerate(all_scene) if scene_name == 'light']
#     # a = random.randint(0, len(path_list))
#     a = random.choice(select_path_list)
#     image_path, thermal_path, scene_name, rgb_bbox, thermal_bbox = path_list[a]
#     with open(image_path, "rb") as fopen:
#         rgb_image = Image.open(fopen).convert("RGB")
#     with open(thermal_path, "rb") as fopen:
#         thermal_image = Image.open(fopen).convert('L')


#     rgb_draw = ImageDraw.Draw(rgb_image)
#     rgb_draw.rectangle(rgb_bbox, outline="red", width=3)
#     thermal_draw = ImageDraw.Draw(thermal_image)
#     thermal_draw.rectangle(thermal_bbox, outline="red", width=3)

#     from IPython.display import display
#     display(rgb_image)
#     display(thermal_image)

# # %%
# all_scene, path_list = obtain_path_list()

# # %%
# bbox_visualization(all_scene, path_list)    