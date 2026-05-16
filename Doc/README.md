# Usage

Clone the repository
```bash
git clone --recurse-submodules -j8 git@github.com:waterljwant/imagebind.git
```

Create virtual environment

1. 

```shell
conda env create -f environment.yaml
```
2.

```shell
conda create --name mm python=3.11
```
first create virtual environment, then do:

change conda pkgs and envs location: https://blog.csdn.net/qq_41198681/article/details/132701216

change pip pkgs location: https://blog.csdn.net/weixin_44955407/article/details/129982442

make sure you use cuda 11.8

```shell
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
```

If you want to run applications, please install GroundingDINO (https://github.com/IDEA-Research/GroundingDINO).

First check cuda path in /.bashrc:

```bash
echo $CUDA_HOME
```

If it prints nothing, you need to add your cuda home to /.bashrc
```bash
export CUDA_HOME=/usr/local/cuda-11.3
```

Then install GroundingDINO:
```bash
git submodule add https://github.com/IDEA-Research/GroundingDINO.git submodules/GroundingDINO
cd submodules/GroundingDINO/
pip install -e .
```

Download pretrained weights

```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```
# 📦 Datasets
create symbolic link: 
```shell
ln -s [src] ./.datasets/depth/RGBD-SUN
``` 

# train and evalution

Run through perpared scripts
```shell
sh scripts/exp_sun.sh
```

```shell
python main.py --expname <name of directory> --cfg <config file for training datasets> --device cuda:0
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for main.py</span></summary>

  #### --expname
  The directory name to store training outputs.
  #### --cfg
  Config file path for training datasets.
  #### --train
  Flag to train or evaluate the model.
  #### --load_checkpoint_dir
  The directory to store pretrained checkpoints. It must be specified when evaluation.
  #### --batch_size
  Batch size
  #### --max_epochs
  Maximum iteration epochs
  #### --codevector_dim
  Dimension of codebook vectors. (default: 8)
  #### --codevector_num
  Expected a list of two elements. (e.g. [1024, 256] in default setting) The number of codebook vectors in common and specfic codebooks.
  #### --val_loss
  Flag for calculating loss during validation.

</details>
<br>
