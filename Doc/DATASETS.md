# Datasets

This document provides instructions for preparing the datasets used in CodeBind.

## Supported Datasets

CodeBind supports the following datasets across multiple modalities:

- **Vision**: Places365, K400，MSR-VTT etc.
- **Audio**: AudioSet, AudioCaps, VGGSOUND, etc.
- **Depth**: SUN Depth, NYU Depth.
- **Thermal**: LLVIP, FLIR_v2.
- **Tactile**: Torch and Go.
- **EEG**: ImageNet-EEG.
- **Point Cloud**: ModelNet40.

## Preparation Instructions
Download the datasets and add symbolic link to ~/.datasets.


| Modality | Dataset | Download link | Other tools or reference |
| --- | --- | --- | --- |
| Image | Places365 | https://data.csail.mit.edu/places/places365/places365standard_easyformat.tar | - |
| Video | MSR-VTT | https://www.kaggle.com/datasets/vishnutheepb/msrvtt | - |
| Video | Kinetics400 (K400) | https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/dataset/k400.md | - |
| Audio | Audioset Audio-only (AS-A) | https://www.kaggle.com/datasets/zfturbo/audioset | - |
| Audio | AudioCaps | https://github.com/cdjkim/audiocaps | - |
| Audio | VGGSound (VGGS) | https://github.com/hche11/VGGSound | - |
| Audio | ESC 5-folds (ESC) | https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master | https://github.com/karolpiczak/ESC-50 |
| Audio | Clotho | https://zenodo.org/record/3490684 | https://github.com/audio-captioning/clotho-dataset |
| Depth | SUN Depth-only (SUN-D) | https://rgbd.cs.princeton.edu/ | https://github.com/open-mmlab/mmdetection3d/blob/main/data/sunrgbd/README.md |
| Depth | NYU_v2 (NYU-D) | https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html | - |
| Thermal | LLVIP | https://bupt-ai-cz.github.io/LLVIP/ | - |
| Thermal | FLIR_v2 | https://www.kaggle.com/datasets/samdazel/teledyne-flir-adas-thermal-dataset-v2 | - |
| 3D point cloud | ULIP | https://github.com/salesforce/ULIP | - |
| Tactile | Torch and Go | https://touch-and-go.github.io/ | - |
| EEG | ImageNet-EEG | https://github.com/perceivelab/eeg_visual_classification?tab=readme-ov-file | - |

We follow similar preprocessing methods in [ImageBind](https://github.com/facebookresearch/imagebind) and [Vit-Lens](https://github.com/TencentARC/ViT-Lens/blob/main/DATASETS.md). Also, we detail the dataset information in the Appendix in our paper.
