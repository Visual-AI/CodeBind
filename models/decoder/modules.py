# -*- coding: utf-8 -*-
# @Organization  : Visual AI Lab, The University of Hong Kong
# @Author        : Zeyu Chen  && Jie Li
# @Function      : Model codes for CodeBind
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from pytorchvideo import transforms as pv_transforms


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1)
        )
    
    def forward(self, x):
        return x + self._block(x)
    

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
    

class DecoderHead(nn.Module):
    def __init__(self, kernel_size, stride, patches_layout, scale, in_channels, hidden_channels, out_channels=3):
        super().__init__()
        # kernel_size = 7
        # stride = 7
        # patch_num = 16  # 每一行、列的patch数量, 224/14 = 16  
        # scale = 2       # 重建后的图像，相比于原图的尺寸缩小倍数
        self.patches_layout = patches_layout  # 1, int(img_resolution_h / scale / kernel_size), int(img_resolution_w / scale / kernel_size)
        self.scale = scale
        # assert img_resolution == self.patch_num * kernel_size * scale
        
        if isinstance(kernel_size, tuple):
            self.deconv = nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels, # out_channels, # hidden_channels,
                kernel_size=kernel_size,
                stride=stride
        )
        else:
            self.deconv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels, # out_channels, # hidden_channels, 
                kernel_size=kernel_size, 
                stride=stride)       
        
        self.hid_conv = ResidualStack(in_channels=hidden_channels,
                                      num_hiddens=hidden_channels,
                                      num_residual_layers=1,
                                      num_residual_hiddens=hidden_channels//2)

        self.out_conv = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=out_channels, 
                               kernel_size=1, 
                               padding=0,
                               stride=1)

    def forward(self, x, ori_data): # _exp_baseline
        # image: ori_data_shape = [batch_size, 3, 224, 224]
        # video: ori_data_shape = [batch_size, num_clips(15), 3, num_frames_per_clip(2), 224, 224]
        # audio: ori_data_shape = [batch_size, num_clips(3), 1, 128, 204]
        # depth & thermal: ori_data_shape = [batch_size, 1, 224, 224]

        # split tokens into patches as patches layout
        x = torch.unflatten(x, dim=2, sizes=self.patches_layout) # b * c * num_tokens -> b * c * patches_layout

        x = self.deconv(x)

        # if x.shape != output_shape:
        #     output_shape = list(output_shape)
        #     output_shape = [size if i not in (len(output_shape)-1, len(output_shape)-2) else int(size / self.scale) for i,size in enumerate(output_shape)]
        #     x = x.reshape(output_shape)
        # x = F.relu(x)
        
        # hidden layers
        # x = self.hid_conv(x)

        # x = self.out_conv(x)
        # x = F.relu(x)  # ! should never use a relu here.
        # align original data shape with decoder head input data shape since audio and video inputs consist of multiple clips
        if ori_data.ndim >= 5:
            B, S = ori_data.shape[:2]
            if x.shape[0] == B * S:
                ori_data = ori_data.view(B * S, *ori_data.shape[2:])
            elif x.shape[0] == B:
                ori_data = ori_data.mean(dim=1)
     
        # resize height and weight of original data to the same as x
        if ori_data.ndim == 5: # video
            ori_tensor = torch.stack([pv_transforms.functional.short_side_scale(clip, 112) for clip in ori_data])
        else: # ori_data.ndim == 4, image, depth, audio, thermal 
            ori_tensor = transforms.functional.resize(ori_data, list(x.shape[-2:]))
        
        # if ori_tensor and x still have different shape, then input is image where we need to repeat ori_tensor
        if x.shape != ori_tensor.shape:
            ori_tensor = ori_tensor[:, :, None, ...]
            new_shape = [1] * len(ori_tensor.shape)
            new_shape[2] = 2
            ori_tensor = ori_tensor.repeat(new_shape)
    
        return x, ori_tensor
