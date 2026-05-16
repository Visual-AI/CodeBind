# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind
# @Reference     : ImageBind, Meta Platforms, Inc. and affiliates.
import os
import xml.etree.ElementTree as ET
import random
import csv
from torch.utils.data import Dataset

from models.codebind_model import ModalityType
from datasets.data_transform_rgbd import load_and_transform_thermal_data_flir
from datasets import data
import pdb
random.seed(43)

# llvip_scene_names = ['person', 'man', 'woman', 'people', 'street', 'road', 'car', 'light', 'tree']
llvip_scene_names = ['person', 'background']

class LLVIPDataset(Dataset):
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
        self.class_names = llvip_scene_names
        self.scale_factor = scale_factor
        self.modality_pair = modality_pair

        for m in self.modality_pair:
            assert m in ['vision', 'thermal', 'text'], f"Get '{m}'"
        self.has_vision = True if 'vision' in self.modality_pair else False
        self.has_thermal = True if 'thermal' in self.modality_pair else False
        self.has_text = True if 'text' in self.modality_pair else False

        self.path_list = list()
        csv_path = os.path.join('.datasets/LLVIP', f'llvip_thermal_{split}.csv')  # split is 'train' or 'test'
        self.parse_csv(csv_path)

        # 放大N倍进行训练，将self.parse_csv repeat N份
        if self.scale_factor > 1 and split != 'test':
            self.path_list = self.path_list * int(self.scale_factor)
        print(f'LLVIPDataset, {self.modality_pair}, split = {split}, length = {len(self.path_list)}')

    def __len__(self):
        return len(self.path_list)

    def get_classnames(self):
        return self.class_names

    def parse_csv(self, csv_path):
        # If the processed csv file for the LLVIP does not exist,
        # then the dataset needs to be prepared from the raw data.
        if not os.path.exists(csv_path):
            prepare_llvip_dataset(self.root_dir)

        with open(csv_path, 'r', encoding="utf-8") as csv_data:
            for row in csv.reader(csv_data, delimiter=';'):
                image_path = row[0]  # image
                thermal_path = row[1]  # thermal
                scene_name = row[2] # scene name in ['person', 'background']
                image_bbox = list(map(int, row[3].split(',')))  # image bbox
                thermal_bbox = list(map(int, row[4].split(',')))  # thermal bbox

                _th = 10
                x1, y1, x2, y2 = image_bbox
                if x2 - x1 <= _th or y2 - y1 <= _th:
                    print(f"image:{image_path}. Bbox: {x2}-{x1}, {y2}-{y1} must >= {_th}")
                    continue
                self.path_list.append([image_path, thermal_path, scene_name, image_bbox, thermal_bbox])

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
            # _rgbt: [1, 2, 4, H, W]  2 is for true and rand bboxes
            
            # true_or_rand = 0 if index%2 else 1 
            # output_dict.update({ModalityType.VISION: _rgbd[0, true_or_rand, 0:3, ...]})  # [3, H, W]
            # output_dict.update({ModalityType.THERMAL: _rgbd[0, true_or_rand, 3, ...].unsqueeze(0)}) # [1, H, W]

            # output_dict.update({ModalityType.VISION: _rgbt[0, :, 0:3, ...]})  # [2, 3, H, W]
            # output_dict.update({ModalityType.THERMAL: _rgbt[0, :, 3, ...].unsqueeze(1)}) # [2, 1, H, W]

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


def get_LLVIP_bboxes(anno_dir, image_dir):
    all_filenames = []
    all_truebboxes = []
    for i in os.listdir(image_dir):
        all_filenames.append(i)
        bboxes = []
        root = ET.parse(anno_dir + '/' + i.split(".")[0] + '.xml').getroot()
        objects = root.findall('object')
        for obj in objects:
            if obj.find('name').text.strip() == 'person':
                bbox = obj.find('bndbox')
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                bbox_true = list(map(int, [xmin, ymin, xmax, ymax]))
                bboxes.append(bbox_true)
        all_truebboxes.append(bboxes)

    return all_filenames, all_truebboxes


def generate_random_bboxes(all_truebboxes):
    img_width, img_height = 1280, 1024
    all_randbboxes = []

    # generate random bboxes
    for bboxes in all_truebboxes:
        bboxes_rand = []
        for bbox in bboxes:
            bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if bbox_w == 0 or bbox_h == 0:
                print(f"ZERO: bbox_w == {bbox_w} or bbox_h == {bbox_h}")
            # generate a random bbox which has the size of current bbox
            bbox_generate_flag = True
            while bbox_generate_flag:
                x1 = random.randint(0, img_width - bbox_w)
                y1 = random.randint(0, img_height - bbox_h)
                bbox_rand = [x1, y1, x1 + bbox_w, y1 + bbox_h]
                # judge whether generated bbox_rand has intersection with other existing bboxes
                intersec_check = []
                for bbox_check in bboxes:
                    intersec_flag1 = (bbox_rand[0] < bbox_check[2]) and (bbox_rand[2] > bbox_check[0])
                    intersec_flag2 = (bbox_rand[1] < bbox_check[3]) and (bbox_rand[3] > bbox_check[1])
                    intersec_check.append(intersec_flag1 and intersec_flag2)
                if all(x is False for x in intersec_check):
                    # if generated bbox_rand has no intersection with existing bboxes, stop regeneration
                    bbox_generate_flag = False
            bboxes_rand.append(bbox_rand)

        all_randbboxes.append(bboxes_rand)
    return all_randbboxes


def prepare_llvip_dataset(llvip_data_root):
    annotation_dir = os.path.join(llvip_data_root, 'Annotations')
    image_train_dir = os.path.join(llvip_data_root, 'infrared/train')
    image_test_dir = os.path.join(llvip_data_root, 'infrared/test')

    # prepare train data info
    print("Preparing LLVIP image-thermal dataset csv files for train.")
    # with open(os.path.join('.datasets/LLVIP', 'llvip_thermal_train.csv'), 'w') as csvoutput:
    #     writer = csv.writer(csvoutput)
    #     for filename in os.listdir(image_train_dir):
    #         image_path = os.path.join(llvip_data_root, 'visible/train', filename)
    #         thermal_path = os.path.join(llvip_data_root, 'infrared/train', filename)
    #         writer.writerow([image_path, thermal_path])

    filenames, true_bboxes = get_LLVIP_bboxes(annotation_dir, image_train_dir)
    rand_bboxes = generate_random_bboxes(true_bboxes)

    with open(os.path.join('.datasets/LLVIP', 'llvip_thermal_train.csv'), 'w') as csvoutput:
        writer = csv.writer(csvoutput, delimiter=';')

        for filename, true_bbox, rand_bbox in zip(filenames, true_bboxes, rand_bboxes):
            image_path = os.path.join(llvip_data_root, 'visible/train', filename)
            thermal_path = os.path.join(llvip_data_root, 'infrared/train', filename)
            for true_bbox_i, rand_bbox_i in zip(true_bbox, rand_bbox):
                true_bbox_output = ','.join(str(num) for num in true_bbox_i)
                rand_bbox_output = ','.join(str(num) for num in rand_bbox_i)

                writer.writerow([image_path, thermal_path, 'person', true_bbox_output, true_bbox_output])
                writer.writerow([image_path, thermal_path, 'background', rand_bbox_output, rand_bbox_output])

    # prepare validation data info
    print("Preparing LLVIP image-thermal dataset csv files for evaluation.")
    filenames, true_bboxes = get_LLVIP_bboxes(annotation_dir, image_test_dir)
    rand_bboxes = generate_random_bboxes(true_bboxes)

    with open(os.path.join('.datasets/LLVIP', 'llvip_thermal_test.csv'), 'w') as csvoutput:
        writer = csv.writer(csvoutput, delimiter=';')

        for filename, true_bbox, rand_bbox in zip(filenames, true_bboxes, rand_bboxes):
            image_path = os.path.join(llvip_data_root, 'visible/test', filename)
            thermal_path = os.path.join(llvip_data_root, 'infrared/test', filename)
            for true_bbox_i, rand_bbox_i in zip(true_bbox, rand_bbox):
                true_bbox_output = ','.join(str(num) for num in true_bbox_i)
                rand_bbox_output = ','.join(str(num) for num in rand_bbox_i)

                writer.writerow([image_path, thermal_path, 'person', true_bbox_output, true_bbox_output])
                writer.writerow([image_path, thermal_path, 'background', rand_bbox_output, rand_bbox_output])



def prepare_llvip_dataset_v1(llvip_data_root):
    annotation_dir = os.path.join(llvip_data_root, '.datasets/LLVIP/Annotations')
    image_train_dir = os.path.join(llvip_data_root, '.datasets/LLVIP/infrared/train')
    image_test_dir = os.path.join(llvip_data_root, '.datasets/LLVIP/infrared/test')

    # prepare train data info
    print("Preparing LLVIP image-thermal dataset csv files for train.")
    with open(os.path.join(llvip_data_root, 'llvip_thermal_train.csv'), 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        for filename in os.listdir(image_train_dir):
            image_path = os.path.join(llvip_data_root, 'visible/train', filename)
            thermal_path = os.path.join(llvip_data_root, 'infrared/train', filename)
            writer.writerow([image_path, thermal_path])

    # prepare validation data info
    print("Preparing LLVIP image-thermal dataset csv files for evaluation.")
    filenames, true_bboxes = get_LLVIP_bboxes(annotation_dir, image_test_dir)
    rand_bboxes = generate_random_bboxes(true_bboxes)

    with open(os.path.join(llvip_data_root, 'llvip_thermal_test.csv'), 'w') as csvoutput:
        writer = csv.writer(csvoutput, delimiter=';')

        for filename, true_bbox, rand_bbox in zip(filenames, true_bboxes, rand_bboxes):
            image_path = os.path.join(llvip_data_root, 'visible/test', filename)
            thermal_path = os.path.join(llvip_data_root, 'infrared/test', filename)
            true_bbox_output = str()
            for true_bbox_i in true_bbox:
                true_bbox_output += ' ' + ','.join(str(num) for num in true_bbox_i)
                true_bbox_output = true_bbox_output.lstrip()
            rand_bbox_output = str()
            for rand_bbox_i in rand_bbox:
                rand_bbox_output += ' ' + ','.join(str(num) for num in rand_bbox_i)
                rand_bbox_output = rand_bbox_output.lstrip()

            writer.writerow([image_path, thermal_path, true_bbox_output, rand_bbox_output])
