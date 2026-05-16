timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}


expname=flir_lora_MVQ1024x8_256x8_scale10
expdir=exp_flir


mkdir -p exp/${expdir}/${expname}


# --- --- --- --- --- --- --- ---   Train  --- --- --- --- --- --- --- ---  
python main.py \
--expname ${expname} \
--batch_size 8 \
--max_epochs 30 \
--loggers wandb \
--val_loss True \
--cfg config/cfg_flir.yaml \
--cfg_model config/cfg_model.yaml \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--loggers_dir exp/${expdir}/${expname}/log \
--device cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log


# small codebook
# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}

# expname=flir_lora_MVQ256x1024_64x1024
# expdir=exp_flir

# mkdir -p exp/${expdir}/${expname}

# python main.py \
# --expname ${expname} \
# --batch_size 8 \
# --max_epochs 30 \
# --loggers wandb \
# --val_loss True \
# --codevector_dim 1024 \
# --codevector_num 256 64 \
# --cfg config/cfg_flir.yaml \
# --cfg_model config/cfg_model.yaml \
# --checkpoint_dir exp/${expdir}/${expname}/checkpoints \
# --log_img_dir exp/${expdir}/${expname}/log_img \
# --lightning_log exp/${expdir}/${expname}/lightning_log \
# --save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
# --loggers_dir exp/${expdir}/${expname}/log \
# --device cuda:0 cuda:1 cuda:2 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log

