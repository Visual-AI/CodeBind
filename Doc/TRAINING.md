# Training & Inference

This document summarizes how to train and evaluate CodeBind with the provided configuration files and shell scripts.

## Preparation

Before running training or inference, please make sure that:

- Datasets are prepared following [DATASETS.md](DATASETS.md).
- Required checkpoints are downloaded following [MODEL_ZOO.md](MODEL_ZOO.md).
- Dataset paths and checkpoint paths in the corresponding config files are updated for your local environment.

## Training

Training scripts for different modalities are provided under `scripts/train/`. Each script calls `main.py` with the dataset config, model config, output directories, logging settings, and GPU devices.

For example:

```bash
sh scripts/train/exp_sun.sh
```


## Configuration

The main model and training hyperparameters are defined in `config/cfg_model.yaml`. Dataset-specific settings are defined in files such as `config/cfg_p365.yaml`, `config/cfg_msrvtt.yaml`, `config/cfg_llvip.yaml`, and other `config/cfg_*.yaml` files.

Common command-line options:

| Argument | Description |
| --- | --- |
| `--cfg` | Dataset configuration file. |
| `--cfg_model` | Model configuration file. Default: `config/cfg_model.yaml`. |
| `--expname` | Experiment name used for organizing outputs. |
| `--datasets` | Dataset name used by the data loader. |
| `--batch_size` | Batch size for training and validation. |
| `--max_epochs` | Maximum number of training epochs. |
| `--device` | Device list, for example `cuda:0` or `cuda:0 cuda:1`. |
| `--checkpoint_dir` | Directory for saving model checkpoints. |
| `--loggers_dir` | Directory for logger outputs. |
| `--save_cfg_filepath` | Path to save the merged training config. |

## Training Modes

CodeBind supports LoRA and head-tuning. By default, the provided scripts use head-tuning for vision-related datasets, including ImageNet1K, Places365, K400, and MSR-VTT, and LoRA for the other modalities.

The vector quantization settings can be overridden from the command line. For example, the default shared and modality-specific codebooks can be set with:

```bash
--codevector_dim 8 \
--codevector_num 1024 256
```

## Logging

Set `--loggers wandb` to enable Weights & Biases logging. The code uses your local wandb login by default, so open-source users should log in with their own account before training:

```bash
wandb login
```

If you want to specify a wandb project or entity explicitly, pass in bash scripts:

```bash
--wandb_project <your_project> \
--wandb_entity <your_entity>
```


## Inference

Evaluation scripts are provided under `scripts/eval/`. These scripts load trained checkpoints through `--resume_expname` and run evaluation on the corresponding modality or dataset.

For example:

```bash
sh scripts/eval/exp_sun.sh
```
