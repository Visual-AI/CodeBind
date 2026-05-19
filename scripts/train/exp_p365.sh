timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}


expname=p365_lora_MVQ1024x8_256x8
expdir=exp_p365


mkdir -p exp/${expdir}/${expname}


# --- --- --- --- --- --- --- ---   Train  --- --- --- --- --- --- --- ---  
python main.py \
--expname ${expname} \
--scale_factor 1 \
--batch_size 10 \
--max_epochs 40 \
--loggers wandb \
--wandb_entity codebind \
--val_loss True \
--cfg config/cfg_p365.yaml \
--cfg_model config/cfg_model.yaml \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--loggers_dir exp/${expdir}/${expname}/log \
--device cuda:0 cuda:1 cuda:2 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log


# 