timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}


expname=ave_lora_MVQ1024x8_256x8
expdir=exp_ave

mkdir -p exp/${expdir}/${expname}


# --- --- --- --- --- --- --- ---   Train  --- --- --- --- --- --- --- ---  
python main.py \
--expname ${expname} \
--scale_factor 1 \
--batch_size 6 \
--max_epochs 30 \
--val_loss True \
--loggers wandb \
--cfg config/cfg_ave.yaml \
--cfg_model config/cfg_model.yaml \
--resume_expname .checkpoints/exp_from_zilong/exp_audio/audio_lora_MVQ1024x8_256x8 \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--loggers_dir exp/${expdir}/${expname}/log \
--device cuda:5 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log

