# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Dataset codes for CodeBind
# @Reference     : ImageBind, Meta Platforms, Inc. and affiliates.

import logging
import math
import imageio
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from PIL import Image
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo  # 

from models.multimodal_preprocessors import SimpleTokenizer

import pdb

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds

BPE_PATH = ".checkpoints/bpe/bpe_simple_vocab_16e6.txt.gz"


simple_templates = [
    'a {}.',
    'a photo of {}.',
    "This is a {}.",
]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


SCENE_CLS_TEMPLATE = [
    "An image depicting a {} environment.",
    "This location is best described as {}.",
    "This location is {}.",
    "A visual scene of {} setting.",
    "This picture showcases a {} environment.",
    "This place is {}.",
    "An example of {} scene category.",
    "An example of a {} scene.",
    "This scene can be described as {}.",
    "A visual scene of a {} location.",
    "This photograph captures a {} scene.",
    "This is a {} setting.",
    "This image corresponds to a {} scene.",
    "This photograph shows {}.",
    "This is an image of {}.",
    "A good image of {}.",
    "A photo of the nice {}.",
    "A picture of {}.",
    "A bright image of {}.",
]

text_template_dict = {
    'simple': simple_templates,
    'imagenet': imagenet_templates,
    'scene': SCENE_CLS_TEMPLATE,

}


def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    if abs(p) / n_frames > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and "
            "target_length (%d). Is the audio_target_length "
            "setting correct?",
            n_frames,
            target_length,
        )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank


def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


def load_and_transform_vision_data(image_paths, device, to_tensor=True):
    if image_paths is None:
        return None

    image_ouputs = []
    for image_path in image_paths:
        if to_tensor:
            data_transform = transforms.Compose(
                [
                    transforms.Resize(
                        224, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
        else:
            data_transform = transforms.Compose(
                [
                    transforms.Resize(
                        224, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224)
                ]
            )
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        if to_tensor:
            image = data_transform(image).to(device)
            image_ouputs.append(image)
        else:
            image = data_transform(image)
            image_ouputs.append(image)
    return image_ouputs if not to_tensor else torch.stack(image_ouputs, dim=0)




def load_and_transform_depth_data(depth_paths, device, to_tensor=True):
    if depth_paths is None:
        return None
    depth_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                DepthNorm(max_depth=7.5, min_depth=0.01),  # TODO: 设置参数 ?
                # transforms.Normalize(
                #     mean=0.0418,  # TODO 统一：image的mean std，以及统一RGB和depth的小数位数
                #     std=0.0295,
                # ),
            ]
        )

    depth_ouputs = []
    # print("depth_paths:", type(depth_paths), len(depth_paths))  # list ,1
    for depth_path in depth_paths:
        # dep_image = Image.open(depth_path)        # int32 
        dep_image = imageio.imread(depth_path)      # uint 16 
        dep_image = np.asarray(dep_image) / 8000.0  # /1000, mm to m.   # /8, shift 3       # numpy.float64
        # print("dep_image, min =", dep_image.min(), " max =", dep_image.max(), depth_path)
        
        if to_tensor:
            dep_tensor = torch.from_numpy(dep_image).float()    # float64 to float32, numpy array to torch tensor
            dep_tensor = torch.unsqueeze(dep_tensor, dim=0)     # expand dims 0, shape: (h, w) --> (1, h, w)
            dep_tensor = dep_tensor.to(device)
            dep_tensor = depth_transform(dep_tensor)
            depth_ouputs.append(dep_tensor)
        else:
            raise NotImplementedError
    return depth_ouputs if not to_tensor else torch.stack(depth_ouputs, dim=0)


def load_and_transform_text(text, device, use_clip_tokenizer=False):
    if text is None:
        return None
    # if use_clip_tokenizer:
    #     from transformers import CLIPTokenizer
    #     clip_tokenizer = CLIPTokenizer.from_pretrained("/home/vislab/jieli23/proj_zy/stablediffusion/checkpoints/stable_diffusion_2_unclip/tokenizer")
    #     text_inputs = clip_tokenizer(
    #             text,
    #             padding="max_length",
    #             max_length=clip_tokenizer.model_max_length,
    #             truncation=True,
    #             return_tensors="pt",
    #         )
    #     text_input_ids = text_inputs.input_ids
    #     tokens = text_input_ids.to(device)
    # else:
    tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


def load_and_transform_audio_data(
    audio_paths,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=1, # default is 3
    mean=-4.268,
    std=9.138,
    freqm_length=12,
    mode=None
):
    if audio_paths is None:
        return None

    audio_outputs = []
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )

    for audio_path in audio_paths:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.nelement() == 0:
            # with open('.datasets/ESC-50/unreadable_audio.txt', 'a+') as f:
            #     f.write(audio_path)
            #     f.write("\n")
            return None
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        all_clips_timepoints = get_clip_timepoints(
            clip_sampler, waveform.size(1) / sample_rate
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * sample_rate) : int(
                    clip_timepoints[1] * sample_rate
                ),
            ]
            waveform_melspec = waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)

        if mode == 'train':
            # Frequency masking, not do for eval set
            freqm = torchaudio.transforms.FrequencyMasking(freqm_length)
            all_clips = [freqm(ac) for ac in all_clips]
        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)

    return torch.stack(audio_outputs, dim=0)


def crop_boxes(boxes, x_offset, y_offset):
    """
    Perform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to perform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


class SpatialCrop(nn.Module):
    """
    Convert the video into 3 smaller clips spatially. Must be used after the
        temporal crops to get spatial crops, and should be used with
        -2 in the spatial crop at the slowfast augmentation stage (so full
        frames are passed in here). Will return a larger list with the
        3x spatial crops as well.
    """

    def __init__(self, crop_size: int = 224, num_crops: int = 3):
        super().__init__()
        self.crop_size = crop_size
        if num_crops == 3:
            self.crops_to_ext = [0, 1, 2]
            self.flipped_crops_to_ext = []
        elif num_crops == 1:
            self.crops_to_ext = [1]
            self.flipped_crops_to_ext = []
        else:
            raise NotImplementedError("Nothing else supported yet")

    def forward(self, videos):
        """
        Args:
            videos: A list of C, T, H, W videos.
        Returns:
            videos: A list with 3x the number of elements. Each video converted
                to C, T, H', W' by spatial cropping.
        """
        assert isinstance(videos, list), "Must be a list of videos after temporal crops"
        assert all([video.ndim == 4 for video in videos]), "Must be (C,T,H,W)"
        res = []
        for video in videos:
            for spatial_idx in self.crops_to_ext:
                res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])
            if not self.flipped_crops_to_ext:
                continue
            flipped_video = transforms.functional.hflip(video)
            for spatial_idx in self.flipped_crops_to_ext:
                res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
        return res


def load_and_transform_video_data(
    video_paths,
    device,
    clip_duration=2,
    clips_per_video=1, # default is 5
    sample_rate=16000,
):
    if video_paths is None:
        return None
    video_outputs = []
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)

    for video_path in video_paths:
        # try:  
        if True:
            video = EncodedVideo.from_path(
                video_path,
                decoder="decord",
                decode_audio=False,
                **{"sample_rate": sample_rate},
            ) 
            # pip install "git+https://github.com/facebookresearch/pytorchvideo.git"
            # EncodedVideo.from_path() got an unexpected keyword argument 'sample_rate'

            all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)

            all_video = []
            for clip_timepoints in all_clips_timepoints:
                # Read the clip, get frames
                clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
                if clip is None:
                    raise ValueError("No clip found")
                video_clip = frame_sampler(clip["video"])
                video_clip = video_clip / 255.0  # since this is float, need 0-1

                all_video.append(video_clip)

            all_video = [video_transform(clip) for clip in all_video]
            all_video = SpatialCrop(224, num_crops=3)(all_video) # each clip produces 3 spatial crops

            all_video = torch.stack(all_video, dim=0)
            video_outputs.append(all_video)

        # except:
            
        #     # with open('.datasets/ESC-50/unreadable_video.txt', 'a+') as f:
        #     #     f.write(video_path)
        #     #     f.write("\n")
        #     return None

    return torch.stack(video_outputs, dim=0).to(device)


class DepthNorm(nn.Module):
    """
    Normalize the depth channel: in a depth input of shape (1, H, W), or in an RGBD input of shape (4, H, W),
    only the last channel is modified.
    The depth channel is also clamped at 0.0. The Midas depth prediction
    model outputs inverse depth maps - negative values correspond
    to distances far away so can be clamped at 0.0
    """

    def __init__(
        self,
        max_depth: float,
        # clamp_max_before_scale: bool = False,
        min_depth: float = 0.001,           
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
        # self.clamp_max_before_scale = clamp_max_before_scale
        self.min_depth = min_depth

    def forward(self, image: torch.Tensor):
        C, H, W = image.shape

        if C == 1:
            depth_img = image  # (1, H, W)
        elif C == 4:
            color_img = image[:3, ...]  # (3, H, W)
            depth_img = image[3:4, ...]  # (1, H, W)
        else:
            err_msg = (
                f"This transform is for 1 channel Depth input of shape (1, H, W) or 4 channel RGBD input of shape (4, H, W); got {image.shape}"
            )
            raise ValueError(err_msg)
        # depth_min = depth_img.min()
        # depth_max = depth_img.max()

        # Clamp to 0.0 to prevent negative depth values
        depth_img = depth_img.clamp(min=self.min_depth)

        # if self.clamp_max_before_scale:
        depth_img = depth_img.clamp(max=self.max_depth)

        # -- 2.4%
        # depth_img = depth_img / self.max_depth
        # -- 7.49%
        # depth_img = 1.0 - (depth_img / self.max_depth)
        # --
        # depth_min = depth_img.min()
        # depth_img /= depth_min

        # --
        # depth_min = depth_img.min()
        # depth_max = depth_img.max()
        # depth_img = (depth_img - depth_min) / (depth_max - depth_min)

        # -- 20.49% 保留 clamp
        # -- 20.64% 去掉 clamp
        depth_min = depth_img.min()
        depth_max = depth_img.max()
        depth_img = 1.0 - (depth_img - depth_min) / (depth_max - depth_min)

        if C == 1:
            img = depth_img
        else:  # C == 4
            img = torch.cat([color_img, depth_img], dim=0)
        return img
