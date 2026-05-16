# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind
# @Reference     : ImageBind, Meta Platforms, Inc. and affiliates.
# https://github.com/pytorch/vision/blob/eb2fb25304dd3180c092231bbe83cd88d25f7cb3/torchvision/transforms/autoaugment.py
# https://github.com/facebookresearch/omnivore/blob/main/omnivision/data/transforms/image_rgbd.py

import math
from typing import Dict, List, Optional, Tuple
from itertools import zip_longest

import numpy as np
import torch
import torchvision.transforms as pth_transforms
import torchvision.transforms.functional as F

from torchvision import transforms
from PIL import Image

import matplotlib as plt


# Ops which can be used on depth
DEPTH_OPS = [
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
    "Rotate",
    "Invert",
    "Identity",
]



# inv_normalize = transforms.Normalize(
#     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
#     std=[1/0.229, 1/0.224, 1/0.255]
# )
# inv_tensor = inv_normalize(tensor)
# 
def inv_normalize(modality_name, data_input):
    
    if modality_name in ['vision', 'image', 'video', 'tactile']:
        _mean = [0.48145466, 0.4578275, 0.40821073]
        _std = [0.26862954, 0.26130258, 0.27577711]
        
    elif modality_name == 'depth':
        _mean = [0.318441]
        _std = [0.11506188]

    elif modality_name == 'thermal':
        _mean = [0.44531357]
        _std = [0.26924619]
    
    elif modality_name == 'audio':
        _mean = [-4.268]
        _std = [9.138]

    else:
        raise NotImplementedError(f"inv_normalize not implement for modality {modality_name}")

    _std = [1.0 / i for i in _std]  
    _mean = [-1.0 * i * j for i,j in zip(_mean, _std)]
    inv_normalize = transforms.Normalize(mean=_mean, std=_std)
    if modality_name == 'audio':
        data_output = inv_normalize(data_input)
        a = torch.amax(data_output, dim=(1, 2, 3))[:, None, None, None]
        b = torch.amin(data_output, dim=(1, 2, 3))[:, None, None, None]
        data_output = (data_output - b) / (a - b)
        data_output = plt.cm.viridis(data_output.cpu().numpy())[..., 0:3]
        data_output = torch.from_numpy(data_output).to(data_input.device)
        data_output = data_output.squeeze(1).permute(0, 3, 1, 2)  
    else:
        data_output = inv_normalize(data_input)
    return data_output



def _apply_op(
    img: torch.Tensor,
    op_name: str,
    magnitude: float,
    interpolation: F.InterpolationMode,
    fill: Optional[List[float]],
):
    if op_name == "ShearX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(magnitude), 0.0],
            interpolation=interpolation,
            fill=fill,
        )
    elif op_name == "ShearY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(magnitude)],
            interpolation=interpolation,
            fill=fill,
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        # The tensor dtype must be torch.uint8
        # and values are expected to be in [0, 255]
        img = (img * 255).to(dtype=torch.uint8)
        img = F.posterize(img, int(magnitude))
        img = (img / 255.0).to(dtype=torch.float32)
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        # The tensor dtype must be torch.uint8
        # and values are expected to be in [0, 255]
        img = (img * 255).to(dtype=torch.uint8)
        img = F.equalize(img)
        img = (img / 255.0).to(dtype=torch.float32)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img


class RandAugment3d(torch.nn.Module):
    """
    Wrapper around torchvision RandAugment transform
    to support 4 channel input for RGBD data

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(
        self, num_bins: int, image_size: List[int]
    ) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (
                torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins),
                True,
            ),
            "TranslateY": (
                torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins),
                True,
            ),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.
        Returns:
            PIL Image or Tensor: Transformed image.
        """
        assert isinstance(img, torch.Tensor)

        C, H, W = img.shape
        images = [img[:3, ...]]  # RGB

        if C == 4:
            depth = img[3:4, ...]  # (1, H, W)
            images.append(depth)

        # Select ops
        # We sample an op and its metadata so that the same op
        # is applied to both RGB and D where relevant
        selected_ops = []
        for _ in range(self.num_ops):
            op_meta = self._augmentation_space(self.num_magnitude_bins, (H, W))
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            selected_ops.append(op_name)

        # Apply on both images and depth
        images_out = []
        for im in images:
            # Only apply some ops for depth if
            # they are part of DEPTH_OPS
            run_on_depth = C == 1 and op_name in DEPTH_OPS
            if C == 3 or run_on_depth:
                fill = self.fill
                if isinstance(im, torch.Tensor):
                    if isinstance(fill, (int, float)):
                        fill = [float(fill)] * C
                    elif fill is not None:
                        fill = [float(f) for f in fill]

                for op_name in selected_ops:
                    magnitudes, signed = op_meta[op_name]
                    magnitude = (
                        float(magnitudes[self.magnitude].item())
                        if magnitudes.ndim > 0
                        else 0.0
                    )
                    if signed and torch.randint(2, (1,)):
                        magnitude *= -1.0

                    im = _apply_op(
                        im,
                        op_name,
                        magnitude,
                        interpolation=self.interpolation,
                        fill=fill,
                    )

            # Save modified image
            images_out.append(im)

        # Concat the img and depth back if present
        images_out = torch.cat(images_out, dim=0)

        return images_out

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_ops={num_ops}"
        s += ", magnitude={magnitude}"
        s += ", num_magnitude_bins={num_magnitude_bins}"
        s += ", interpolation={interpolation}"
        s += ", fill={fill}"
        s += ")"
        return s.format(**self.__dict__)


class ColorJitter3d(pth_transforms.ColorJitter):
    """
    Apply ColorJitter on an image of shape (4, H, W)
    """

    def __init__(self, brightness, contrast, saturation, hue):
        """
        Args:
            strength (float): A number used to quantify the strength of the
                              color distortion.
        """
        super().__init__(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, image: torch.Tensor):
        if not isinstance(image, torch.Tensor):
            raise ValueError("Expected tensor input")
        C, H, W = image.shape
        if C != 4:
            err_msg = "This transform is for 4 channel RGBD input only; got %d" % C
            raise ValueError(err_msg)
        color_img = image[:3, ...]  # (3, H, W)
        depth_img = image[3:4, ...]  # (1, H, W)
        color_img_jitter = super().__call__(color_img)
        img = torch.cat([color_img_jitter, depth_img], dim=0)

        return img



class DepthNorm(torch.nn.Module):
    """
    Normalize the depth channel: in an RGBD input of shape (4, H, W),
    only the last channel is modified.
    The depth channel is also clamped at 0.0. The Midas depth prediction
    model outputs inverse depth maps - negative values correspond
    to distances far away so can be clamped at 0.0
    """

    def __init__(
        self,
        max_depth: float,
        clamp_max_before_scale: bool = False,
        min_depth: float = 0.01,
    ):
        """
        Args:
            max_depth (float): The max value of depth for the dataset
            clamp_max (bool): Whether to clamp to max_depth or to divide by max_depth
        """
        super().__init__()
        if max_depth < 0.0:
            raise ValueError("max_depth must be > 0; got %.2f" % max_depth)
        self.max_depth = max_depth
        self.clamp_max_before_scale = clamp_max_before_scale
        self.min_depth = min_depth

    def __call__(self, image: torch.Tensor):
        C, H, W = image.shape
        assert C in (1, 4)
        if C == 4:
            color_img = image[:3, ...]  # (3, H, W)
            depth_img = image[3:4, ...]  # (1, H, W)

            # Clamp to 0.0 to prevent negative depth values
            depth_img = depth_img.clamp(min=self.min_depth)

            # divide by max_depth
            if self.clamp_max_before_scale:
                depth_img = depth_img.clamp(max=self.max_depth)

            depth_img /= self.max_depth

            img = torch.cat([color_img, depth_img], dim=0)
            return img

        elif C == 1:
            depth_img = image
            # Clamp to 0.0 to prevent negative depth values
            depth_img = depth_img.clamp(min=self.min_depth)

            # divide by max_depth
            if self.clamp_max_before_scale:
                depth_img = depth_img.clamp(max=self.max_depth)

            # depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
            depth_img /= self.max_depth
            # depth_img = (depth_img - depth_img.mean()) / depth_img.std()
            return depth_img

sensor_to_params = {
    "kv1": {
        "baseline": 0.075,
    },
    "kv2": {
        "baseline": 0.075,
    },
    "realsense": {
        "baseline": 0.095,
    },
    "xtion": {
        "baseline": 0.095, # guessed based on length of 18cm for ASUS xtion v1
    },
}

def convert_depth_to_disparity(depth_image, sensor_type, focal_length, min_depth=0.01, max_depth=50):
    """
    depth_image is a numpy array that contains the scene depth
    sensor_type is camera type in SUNRGBD
    focal_length is camera intrinsics in SUNRGBD which can be found at the path: os.path.join(root_dir, "intrinsics.txt")
    """
    baseline = sensor_to_params[sensor_type]["baseline"]
    depth_in_meters = depth_image / 8000.
    if min_depth is not None:
        depth_in_meters = depth_in_meters.clip(min=min_depth, max=max_depth)
    disparity = baseline * float(focal_length) / depth_in_meters
    return disparity


# image and depth should be loaded as a whole under the paired transformation
# add depth to disparity
# add transformation
def load_and_transform_rgbd_data(image_paths=None, depth_paths=None,
                                 camera_types=None, focal_lengths=None, device=None, mode=None):
    if depth_paths is None and image_paths is None:
        return None
    if image_paths is not None and depth_paths is not None:
        # set data transform
        if mode == 'train':
            data_transform = transforms.Compose(
                [
                    DepthNorm(max_depth=75, min_depth=0.01, clamp_max_before_scale=True),
                    # Debug without data aug
                    # transforms.Resize(224),
                    # transforms.CenterCrop(224),
                    # data aug
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    RandAugment3d(magnitude=9),
                    ColorJitter3d(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                    transforms.RandomErasing(p=0.25),
                    # 
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073, 0.318441),
                        std=(0.26862954, 0.26130258, 0.27577711, 0.11506188),
                    ),
                ]
            )
        else:
            data_transform = transforms.Compose(
                [
                    DepthNorm(max_depth=75, min_depth=0.01, clamp_max_before_scale=True),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073, 0.318441),
                        std=(0.26862954, 0.26130258, 0.27577711, 0.11506188),
                    ),
                ]
            )

        rgbd_outputs = []
        for image_path, depth_path, camera_type, focal_length in zip(image_paths, depth_paths, camera_types, focal_lengths):

            # process rgb image
            with open(image_path, "rb") as fopen:
                rgb_image = Image.open(fopen).convert("RGB")
            rgb_tensor = transforms.functional.to_tensor(rgb_image)

            # process depth image
            with open(depth_path, "rb") as fopen:
                dep_image = Image.open(fopen) # uint 16
                dep_image = np.asarray(dep_image).astype(np.float32)

            # convert depth to disparity
            dep_image = convert_depth_to_disparity(dep_image, camera_type, focal_length)
            # dep_image: (H, w) --> (1, H, W)
            if dep_image.ndim == 2:
                dep_image = dep_image[None, ...]
            dep_tensor = torch.from_numpy(dep_image)

            rgbd_tensor = torch.cat([rgb_tensor, dep_tensor], dim=0)  # (4, H, W)
            rgbd_tensor = data_transform(rgbd_tensor)
            rgbd_outputs.append(rgbd_tensor)

        return torch.stack(rgbd_outputs, dim=0).to(device)

    elif image_paths is not None:
        data_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        rgb_outputs = []
        for image_path in image_paths:

            # process rgb image
            with open(image_path, "rb") as fopen:
                rgb_image = Image.open(fopen).convert("RGB")
            rgb_tensor = transforms.functional.to_tensor(rgb_image)

            rgb_tensor = data_transform(rgb_tensor)
            rgb_outputs.append(rgb_tensor)

        return torch.stack(rgb_outputs, dim=0).to(device)

    elif depth_paths is not None:
        assert camera_types is not None and focal_lengths is not None
        data_transform = transforms.Compose(
            [
                DepthNorm(max_depth=75, min_depth=0.01, clamp_max_before_scale=True),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=0.23207855,          # mean: 0.23207855 std: 0.11506188 for nyu ; 0.0418 0.0295 for sun
                    std=0.11506188,           # 0.318441 0.104445 for sun test;  guess: 0.318441 1.888 for sun
                ),
            ]
        )

        dep_outputs = []
        for depth_path, camera_type, focal_length in zip(depth_paths, camera_types, focal_lengths):

            # process depth image
            with open(depth_path, "rb") as fopen:
                dep_image = Image.open(fopen)  # uint 16
                dep_image = np.asarray(dep_image).astype(np.float32)

            # convert depth to disparity
            dep_image = convert_depth_to_disparity(dep_image, camera_type, focal_length)
            # dep_image: (H, w) --> (1, H, W)
            if dep_image.ndim == 2:
                dep_image = dep_image[None, ...]
            dep_tensor = torch.from_numpy(dep_image)
            dep_tensor = data_transform(dep_tensor)
            # with open('.datasets/depth/RGBD-SUN/disparity_stat.txt', "a+") as f:
            #     f.write(str(round(dep_tensor.min().item(), 4)) + " " + str(round(dep_tensor.max().item(), 4)) + " " +
            #             str(round(dep_tensor.mean().item(), 4)) + " " + str(round(dep_tensor.std().item(), 4)))
            #     f.write("\n")
            dep_outputs.append(dep_tensor)

        return torch.stack(dep_outputs, dim=0).to(device)


def load_and_transform_thermal_data(image_paths=None, thermal_paths=None,
                                 true_bboxes=None, rand_bboxes=None, device=None, mode=None):
    if thermal_paths is None and image_paths is None:
        return None
    if mode == 'train':
        data_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                RandAugment3d(magnitude=9),
                ColorJitter3d(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomErasing(p=0.25),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073, 1),
                    std=(0.26862954, 0.26130258, 0.27577711, 0.5),
                ),
            ]
        )
    else:
        if image_paths is not None and thermal_paths is not None:
            data_transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073, 1), #0.28004716
                        std=(0.26862954, 0.26130258, 0.27577711, 0.5), #0.17184836
                    ),
                ]
            )
        elif image_paths is not None:
            data_transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
        elif thermal_paths is not None:
            data_transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=1,
                        std=0.5,
                    ),
                ]
            )

    image_paths = [] if image_paths is None else image_paths
    thermal_paths = [] if thermal_paths is None else thermal_paths
    true_bboxes = [] if true_bboxes is None else true_bboxes
    rand_bboxes = [] if rand_bboxes is None else rand_bboxes

    rgbt_outputs = []
    for image_path, thermal_path, true_bbox, rand_bbox in zip_longest(image_paths, thermal_paths, true_bboxes, rand_bboxes, fillvalue=None):
        if image_path is not None and thermal_path is not None:
            with open(image_path, "rb") as fopen:
                rgb_image = Image.open(fopen).convert("RGB")
            with open(thermal_path, "rb") as fopen:
                thermal_image = Image.open(fopen).convert('L')
                
            if true_bbox is not None and rand_bbox is not None:
                rgbt_crop_tensor = []
                for crop_bbox in (true_bbox, rand_bbox):
                    # crop rgb image based on bounding boxes and convert to tensor
                    cropped_rgb = rgb_image.crop(crop_bbox)
                    # pad the cropped tensor to the original image size
                    # padding_size = (crop_bbox[0], crop_bbox[1], rgb_image.width - crop_bbox[2],
                    #                 rgb_image.height - crop_bbox[3])
                    # cropped_image = transforms.Pad(padding_size, fill=0)(cropped_image)
                    cropped_rgb = transforms.functional.to_tensor(cropped_rgb)

                    # crop thermal image based on bounding boxes, add channel and convert to tensor
                    cropped_thermal = thermal_image.crop(crop_bbox)
                    # pad the cropped tensor to the original image size
                    # padding_size = (crop_bbox[0], crop_bbox[1], thermal_image.width - crop_bbox[2],
                    #                 thermal_image.height - crop_bbox[3])
                    # cropped_image = transforms.Pad(padding_size, fill=0)(cropped_image)
                    cropped_thermal = np.asarray(cropped_thermal).astype(np.float32) / 255
                    cropped_thermal = cropped_thermal[None, ...]
                    cropped_thermal = torch.from_numpy(cropped_thermal)

                    # combine rgb and thermal tensor into rgbd tensor
                    cropped_rgbt = torch.cat([cropped_rgb, cropped_thermal], dim=0)  # (4, bbox_h, bbox_w)
                    cropped_rgbt = data_transform(cropped_rgbt)
                    rgbt_crop_tensor.append(cropped_rgbt)
                rgbt_tensor = torch.stack(rgbt_crop_tensor, dim=0)  # (2, 4, H, W)  2: true_bbox + rand_bbox
            else:
                rgb_tensor = transforms.functional.to_tensor(rgb_image)
                thermal_image = np.asarray(thermal_image).astype(np.float32) / 255
                thermal_image = thermal_image[None, ...]
                thermal_tensor = torch.from_numpy(thermal_image)
                rgbt_tensor = torch.cat([rgb_tensor, thermal_tensor], dim=0)  # (4, H, W)
                rgbt_tensor = data_transform(rgbt_tensor)  
            rgbt_outputs.append(rgbt_tensor)

        elif image_path is not None:
            with open(image_path, "rb") as fopen:
                rgb_image = Image.open(fopen).convert("RGB")

            if true_bbox is not None and rand_bbox is not None:
                rgbt_crop_tensor = []
                for crop_bbox in (true_bbox, rand_bbox):
                    # crop rgb image based on bounding boxes and convert to tensor
                    cropped_rgb = rgb_image.crop(crop_bbox)
                    # pad the cropped tensor to the original image size
                    # padding_size = (crop_bbox[0], crop_bbox[1], rgb_image.width - crop_bbox[2],
                    #                 rgb_image.height - crop_bbox[3])
                    # cropped_image = transforms.Pad(padding_size, fill=0)(cropped_image)
                    cropped_rgb = transforms.functional.to_tensor(cropped_rgb)
                    cropped_rgb = data_transform(cropped_rgb)
                    rgbt_crop_tensor.append(cropped_rgb)
                rgbt_tensor = torch.stack(rgbt_crop_tensor, dim=0)  # (2, 3, H, W)  2: true_bbox + rand_bbox
            else:
                rgb_tensor = transforms.functional.to_tensor(rgb_image)
                rgbt_tensor = data_transform(rgb_tensor)
            rgbt_outputs.append(rgbt_tensor)
        
        elif thermal_path is not None:
            with open(thermal_path, "rb") as fopen:
                thermal_image = Image.open(fopen).convert('L')

            if true_bbox is not None and rand_bbox is not None:
                rgbt_crop_tensor = []
                for crop_bbox in (true_bbox, rand_bbox):
                    # crop thermal image based on bounding boxes and convert to tensor
                    cropped_thermal = thermal_image.crop(crop_bbox)
                    # pad the cropped tensor to the original image size
                    # padding_size = (crop_bbox[0], crop_bbox[1], thermal_image.width - crop_bbox[2],
                    #                 thermal_image.height - crop_bbox[3])
                    # cropped_image = transforms.Pad(padding_size, fill=0)(cropped_image)
                    cropped_thermal = np.asarray(cropped_thermal).astype(np.float32) / 255
                    cropped_thermal = cropped_thermal[None, ...]
                    cropped_thermal = torch.from_numpy(cropped_thermal)
                    cropped_thermal = data_transform(cropped_thermal)
                    rgbt_crop_tensor.append(cropped_thermal)
                rgbt_tensor = torch.stack(rgbt_crop_tensor, dim=0)
            else:
                thermal_image = np.asarray(thermal_image).astype(np.float32) / 255
                thermal_image = thermal_image[None, ...]
                thermal_tensor = torch.from_numpy(thermal_image)
                rgbt_tensor = data_transform(thermal_tensor)
            rgbt_outputs.append(rgbt_tensor)

    return torch.stack(rgbt_outputs, dim=0).to(device)

    
    
def load_and_transform_thermal_data_v2(image_paths=None, thermal_paths=None, true_bboxes=None, rand_bboxes=None, device=None, mode=None):
    if thermal_paths is None and image_paths is None:
        return None

    if mode == 'train':
        data_transform_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.uint8),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(magnitude=9),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomErasing(p=0.25),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        data_transform_thermal = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.uint8),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(magnitude=9),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomErasing(p=0.25),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
                transforms.Grayscale(),
            ]
        )
    else:
        data_transform_rgbt = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        data_transform_thermal = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
                transforms.Grayscale(),
            ]
        )
        
    image_paths = [] if image_paths is None else image_paths
    thermal_paths = [] if thermal_paths is None else thermal_paths
    true_bboxes = [] if true_bboxes is None else true_bboxes
    rand_bboxes = [] if rand_bboxes is None else rand_bboxes

    rgb_outputs, thermal_outputs = [], []
    for image_path, thermal_path, true_bbox, rand_bbox in zip_longest(image_paths, thermal_paths, true_bboxes, rand_bboxes, fillvalue=None):

        # process rgb image
        if image_path is not None:
            with open(image_path, "rb") as fopen:
                rgb_image = Image.open(fopen).convert("RGB")
            
            if true_bbox is not None and rand_bbox is not None:
                # crop rgb image based on bounding boxes and convert to tensor
                rgb_crop_tensor = []
                for crop_bbox in (true_bbox, rand_bbox):
                    cropped_image = rgb_image.crop(crop_bbox)
                    # pad the cropped tensor to the original image size
                    # padding_size = (crop_bbox[0], crop_bbox[1], rgb_image.width - crop_bbox[2],
                    #                 rgb_image.height - crop_bbox[3])
                    # cropped_image = transforms.Pad(padding_size, fill=0)(cropped_image)
                    cropped_tensor = data_transform_rgb(cropped_image)
                    rgb_crop_tensor.append(cropped_tensor)

                rgb_tensor = torch.stack(rgb_crop_tensor, dim=0).to(device)  # (2, 3, bbox_h, bbox_w)  2: true_bbox + rand_bbox
            else:
                rgb_tensor = data_transform_rgb(rgb_image)

            rgb_outputs.append(rgb_tensor)

        # process thermal image
        if thermal_path is not None:
            with open(thermal_path, "rb") as fopen:
                thermal_image = Image.open(fopen).convert('RGB')
            
            if true_bbox is not None and rand_bbox is not None:
                # crop thermal image based on bounding boxes, add channel, convert to tensor and data transform
                thermal_crop_tensor = []
                for crop_bbox in (true_bbox, rand_bbox):
                    cropped_image = thermal_image.crop(crop_bbox)
                    # pad the cropped tensor to the original image size
                    # padding_size = (crop_bbox[0], crop_bbox[1], thermal_image.width - crop_bbox[2],
                    #                 thermal_image.height - crop_bbox[3])
                    # cropped_image = transforms.Pad(padding_size, fill=0)(cropped_image)
                    cropped_tensor = data_transform_thermal(cropped_image)
                    thermal_crop_tensor.append(cropped_tensor)

                thermal_tensor = torch.stack(thermal_crop_tensor, dim=0).to(device)  # (2, 1, bbox_h, bbox_w) 2: true_bbox + rand_bbox
            else:
                thermal_tensor = data_transform_thermal(thermal_image)     

            thermal_outputs.append(thermal_tensor)

    return torch.stack(rgb_outputs, dim=0) if rgb_outputs != [] else None, torch.stack(thermal_outputs, dim=0) if thermal_outputs != [] else None


def load_and_transform_thermal_data_flir(image_paths=None, thermal_paths=None,
                                         image_bboxes=None, thermal_bboxes=None, device=None, mode=None):
    if thermal_paths is None and image_paths is None:
        return None
    if mode == 'train':
        data_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                RandAugment3d(magnitude=9),
                ColorJitter3d(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomErasing(p=0.25),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073, 0.44531357),
                    std=(0.26862954, 0.26130258, 0.27577711, 0.26924619),
                ),
            ]
        )
        resizes = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
    else:
        if image_paths is not None and thermal_paths is not None:
            data_transform = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073, 0.44531357), 
                        std=(0.26862954, 0.26130258, 0.27577711, 0.26924619), 
                    ),
                ]
            )
            resizes = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
        elif image_paths is not None:
            data_transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
        elif thermal_paths is not None:
            data_transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=0.44531357,
                        std=0.26924619,
                    ),
                ]
            )

    image_paths = [] if image_paths is None else image_paths
    thermal_paths = [] if thermal_paths is None else thermal_paths
    image_bboxes = [] if image_bboxes is None else image_bboxes
    thermal_bboxes = [] if thermal_bboxes is None else thermal_bboxes

    rgbt_outputs = []
    for image_path, thermal_path, image_bbox, thermal_bbox in zip_longest(image_paths, thermal_paths, image_bboxes, thermal_bboxes, fillvalue=None):
        if image_path is not None and thermal_path is not None:
            with open(image_path, "rb") as fopen:
                rgb_image = Image.open(fopen).convert("RGB")
            with open(thermal_path, "rb") as fopen:
                thermal_image = Image.open(fopen).convert('L')
                
            if image_bbox is not None and thermal_bbox is not None:
                # crop rgb image based on bounding boxes and convert to tensor
                cropped_rgb = rgb_image.crop(image_bbox)
                # pad the cropped tensor to the original image size
                # padding_size = (crop_bbox[0], crop_bbox[1], rgb_image.width - crop_bbox[2],
                #                 rgb_image.height - crop_bbox[3])
                # cropped_image = transforms.Pad(padding_size, fill=0)(cropped_image)
                cropped_rgb = transforms.functional.to_tensor(cropped_rgb)
                cropped_rgb = resizes(cropped_rgb)

                # crop thermal image based on bounding boxes, add channel and convert to tensor
                cropped_thermal = thermal_image.crop(thermal_bbox)
                # pad the cropped tensor to the original image size
                # padding_size = (crop_bbox[0], crop_bbox[1], thermal_image.width - crop_bbox[2],
                #                 thermal_image.height - crop_bbox[3])
                # cropped_image = transforms.Pad(padding_size, fill=0)(cropped_image)
                cropped_thermal = np.asarray(cropped_thermal).astype(np.float32) / 255
                cropped_thermal = cropped_thermal[None, ...]
                cropped_thermal = torch.from_numpy(cropped_thermal)
                cropped_thermal = resizes(cropped_thermal)

                # combine rgb and thermal tensor into rgbd tensor
                cropped_rgbt = torch.cat([cropped_rgb, cropped_thermal], dim=0)  # (4, bbox_h, bbox_w)
                rgbt_tensor = data_transform(cropped_rgbt)
            else:
                rgb_tensor = transforms.functional.to_tensor(rgb_image)
                rgb_tensor = resizes(rgb_tensor)
                thermal_image = np.asarray(thermal_image).astype(np.float32) / 255
                thermal_image = thermal_image[None, ...]
                thermal_tensor = torch.from_numpy(thermal_image)
                thermal_tensor = resizes(thermal_tensor)
                rgbt_tensor = torch.cat([rgb_tensor, thermal_tensor], dim=0)  # (4, H, W)
                rgbt_tensor = data_transform(rgbt_tensor)  
            rgbt_outputs.append(rgbt_tensor)

        elif image_path is not None:
            with open(image_path, "rb") as fopen:
                rgb_image = Image.open(fopen).convert("RGB")

            if image_bbox is not None:
                # crop rgb image based on bounding boxes and convert to tensor
                cropped_rgb = rgb_image.crop(image_bbox)
                # pad the cropped tensor to the original image size
                # padding_size = (crop_bbox[0], crop_bbox[1], rgb_image.width - crop_bbox[2],
                #                 rgb_image.height - crop_bbox[3])
                # cropped_image = transforms.Pad(padding_size, fill=0)(cropped_image)
                cropped_rgb = transforms.functional.to_tensor(cropped_rgb)
                rgbt_tensor = data_transform(cropped_rgb)
            else:
                rgb_tensor = transforms.functional.to_tensor(rgb_image)
                rgbt_tensor = data_transform(rgb_tensor)
            rgbt_outputs.append(rgbt_tensor)
        
        elif thermal_path is not None:
            with open(thermal_path, "rb") as fopen:
                thermal_image = Image.open(fopen).convert('L')

            if thermal_bbox is not None:
                # crop thermal image based on bounding boxes and convert to tensor
                cropped_thermal = thermal_image.crop(thermal_bbox)
                # pad the cropped tensor to the original image size
                # padding_size = (crop_bbox[0], crop_bbox[1], thermal_image.width - crop_bbox[2],
                #                 thermal_image.height - crop_bbox[3])
                # cropped_image = transforms.Pad(padding_size, fill=0)(cropped_image)
                cropped_thermal = np.asarray(cropped_thermal).astype(np.float32) / 255
                cropped_thermal = cropped_thermal[None, ...]
                cropped_thermal = torch.from_numpy(cropped_thermal)
                rgbt_tensor = data_transform(cropped_thermal)
            else:
                thermal_image = np.asarray(thermal_image).astype(np.float32) / 255
                thermal_image = thermal_image[None, ...]
                thermal_tensor = torch.from_numpy(thermal_image)
                rgbt_tensor = data_transform(thermal_tensor)
            rgbt_outputs.append(rgbt_tensor)

    return torch.stack(rgbt_outputs, dim=0).to(device)