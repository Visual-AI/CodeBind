# Model Zoo

This document lists the available CodeBind model checkpoints and their download links.

## Available Checkpoints

### CodeBind-IB (ImageBind-based)
- **Model**: CodeBind-IB
- **Modalities**: Vision, Text, Audio, Depth, Thermal
- **Download**: [Link to be provided]
- **Description**: Pre-trained on ImageBind data.

### CodeBind-Full
- **Model**: CodeBind-Full
- **Modalities**: All nine modalities
- **Download**: [Link to be provided]
- **Description**: Fully trained model.

## Usage
To load a checkpoint:
```python
from models.codebind_model import load_module
model = load_module(checkpoint_path)
```

## Notes
- Model weights are for research purposes only.
- Ensure compatibility with your PyTorch version.

For updates, check the project repository.