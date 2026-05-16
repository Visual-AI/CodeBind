# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}


# expname=sun_lora_MVQ1024x1024_256x1024_vqloss
# expdir=exp_sun

# mkdir -p exp/${expdir}/${expname}

# # --- --- --- --- --- --- --- ---   Train  --- --- --- --- --- --- --- ---  
# python main.py \
# --expname ${expname} \
# --datasets sun_evalnyu \
# --batch_size 10 \
# --max_epochs 40 \
# --loggers wandb \
# --val_loss True \
# --contras_loss True \
# --uni_loss True \
# --codevector_dim 1024 \
# --cfg config/cfg_sun.yaml \
# --cfg_model config/cfg_model.yaml \
# --sun_root /hdddata/jieli23/dataset/RGBD-SUN \
# --nyu_root /hdddata/jieli23/dataset/RGBD-SUN \
# --checkpoint_dir exp/${expdir}/${expname}/checkpoints \
# --log_img_dir exp/${expdir}/${expname}/log_img \
# --lightning_log exp/${expdir}/${expname}/lightning_log \
# --save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
# --loggers_dir exp/${expdir}/${expname}/log \
# --device cuda:0 cuda:1 cuda:2 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log


timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}


expname=sun_lora_MVQ1024x8_256x8
expdir=exp_sun

mkdir -p exp/${expdir}/${expname}

# --- --- --- --- --- --- --- ---   Train  --- --- --- --- --- --- --- ---  
python main.py \
--expname ${expname} \
--datasets sun_evalnyu \
--batch_size 8 \
--max_epochs 40 \
--loggers wandb \
--val_loss True \
--cfg config/cfg_sun.yaml \
--cfg_model config/cfg_model.yaml \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--loggers_dir exp/${expdir}/${expname}/log \
--device cuda:1 cuda:2 cuda:3 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log


# small codebook
# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}

# expname=sun_lora_MVQ128x32_32x32
# expdir=exp_sun

# mkdir -p exp/${expdir}/${expname}

# # --- --- --- --- --- --- --- ---   Train  --- --- --- --- --- --- --- ---  
# python main.py \
# --expname ${expname} \
# --datasets sun_evalnyu \
# --batch_size 8 \
# --max_epochs 40 \
# --loggers wandb \
# --val_loss True \
# --codevector_dim 32 \
# --codevector_num 128 32 \
# --cfg config/cfg_sun.yaml \
# --cfg_model config/cfg_model.yaml \
# --checkpoint_dir exp/${expdir}/${expname}/checkpoints \
# --log_img_dir exp/${expdir}/${expname}/log_img \
# --lightning_log exp/${expdir}/${expname}/lightning_log \
# --save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
# --loggers_dir exp/${expdir}/${expname}/log \
# --device cuda:0 cuda:1 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log

