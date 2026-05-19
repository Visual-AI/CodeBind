# Model Zoo

This document lists the available CodeBind checkpoints and explains how to download and place them for evaluation.

## Checkpoint Directory

All downloaded checkpoints should be placed under (or connected via symoblic link):

```bash
~/.checkpoints/
```


## Available Checkpoints

| Model | Backbone | Modalities | Download | Notes |
| --- | --- | --- | --- | --- |
| CodeBind-IB | ImageBind | Vision, Audio, Depth, Thermal | [Hugging Face](https://huggingface.co/zykev/CodeBind) | CodeBind checkpoint trained from ImageBind features. |
| ImageBind-Huge | ImageBind | Vision, Text, Audio, Depth, IMU, Thermal | [Meta download](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) | Required ImageBind base checkpoint. |

## Download From Hugging Face

The CodeBind checkpoints are hosted at:

```text
https://huggingface.co/zykev/CodeBind
```

You can download them with `huggingface-cli`:

```bash
pip install -U huggingface_hub
huggingface-cli download zykev/CodeBind --local-dir ~/.checkpoints/
```


After downloading, make sure the checkpoint folders are available under `~/.checkpoints/`. For example:

```bash
ls ~/.checkpoints/exp_asa
```

## Download ImageBind Base Checkpoint

The ImageBind base checkpoint should also be placed under `~/.checkpoints/`:

```bash
mkdir -p ~/.checkpoints
wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth -O ~/.checkpoints/imagebind_huge.pth
```

## Usage

Evaluation scripts load trained checkpoints through `--resume_expname`. Point this argument to the downloaded experiment directory under `~/.checkpoints/`.

For example:

```bash
python main.py \
  --resume_expname ~/.checkpoints/CodeBind/path/to/experiment \
```

For the provided bash scripts, update the corresponding `--resume_expname` value in `scripts/eval/*.sh` to the checkpoint directory you want to evaluate.

## Notes

- Checkpoint weights are released for research purposes.
- For updates, please check the [CodeBind Hugging Face repository](https://huggingface.co/zykev/CodeBind).
