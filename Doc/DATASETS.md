# Datasets

This document provides instructions for preparing the datasets used in CodeBind.

## Supported Datasets

CodeBind supports the following datasets across multiple modalities:

- **Vision**: ImageNet, Places365, etc.
- **Text**: Various text corpora.
- **Audio**: AudioCaps, ESC-50, etc.
- **Depth**: NYU Depth V2.
- **Thermal**: FLIR.
- **Tactile**: Tactile datasets.
- **EEG**: EEG datasets.
- **Point Cloud**: 3D point cloud datasets.

## Preparation Instructions

### 1. Download Datasets
- Visit the official dataset websites and download the required data.
- Ensure you have sufficient storage space.

### 2. Preprocessing
- Use the provided scripts in `scripts/` to preprocess the data.
- Example: `python scripts/preprocess_imagenet.py`

### 3. Configuration
- Update `config/cfg_datasets.yaml` with dataset paths.
- Ensure paths are absolute or relative to the project root.

For more details, refer to the dataset-specific READMEs or contact the maintainers.