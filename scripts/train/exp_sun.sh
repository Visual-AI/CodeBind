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
--wandb_entity codebind \
--val_loss True \
--cfg config/cfg_sun.yaml \
--cfg_model config/cfg_model.yaml \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--loggers_dir exp/${expdir}/${expname}/log \
--device cuda:1 cuda:2 cuda:3 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log



# --codevector_dim 32 \
# --codevector_num 128 32 \
