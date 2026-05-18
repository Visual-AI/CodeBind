# Training & Inference

This document provides instructions for training and running inference with CodeBind.

## Training

### Prerequisites
- Install dependencies as per the main README.
- Prepare datasets (see [Doc/DATASETS.md](Doc/DATASETS.md)).

### Training Scripts
Use the scripts in `scripts/train/` for different modalities.

Example:
```bash
python main.py --config config/cfg_model.yaml --train
```

### Configuration
- Edit `config/cfg_model.yaml` for hyperparameters.
- Supported modes: LoRA, Full-tune, Head-tune.

## Inference

### Running Inference
```python
from main import CodeBind
model = CodeBind.load_from_checkpoint('path/to/checkpoint.ckpt')
# Perform inference
```

### Evaluation
- Use `application/` scripts for classification and retrieval tasks.
- Metrics: Accuracy, mAP, Recall.

For detailed scripts, refer to `scripts/eval/`.

## Troubleshooting
- Ensure GPU memory is sufficient.
- Check logs for errors.

Contact maintainers for support.