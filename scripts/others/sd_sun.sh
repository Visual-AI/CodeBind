timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}


expname=sd_sun_lora
expdir=exp_sun

mkdir -p exp/${expdir}/${expname}

# --- --- --- --- --- --- --- ---   Train  --- --- --- --- --- --- --- ---  
python main.py \
--expname ${expname} \
--datasets sun_evalnyu \
--batch_size 8 \
--max_epochs 20 \
--loggers wandb \
--val_loss True \
--sd_version True \
--cfg config/cfg_sun.yaml \
--cfg_model config/cfg_model.yaml \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--loggers_dir exp/${expdir}/${expname}/log \
--device cuda:2 cuda:3 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log