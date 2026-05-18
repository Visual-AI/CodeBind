timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}


expname=vggs_lora_MVQ1024x8_256x8
expdir=exp_vggs

mkdir -p exp/${expdir}/${expname}


# --- --- --- --- --- --- --- ---   Train  --- --- --- --- --- --- --- ---  
python main.py \
--expname ${expname} \
--scale_factor 1 \
--batch_size 6 \
--max_epochs 40 \
--val_loss True \
--cfg config/cfg_vggs.yaml \
--cfg_model config/cfg_model.yaml \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--loggers_dir exp/${expdir}/${expname}/log \
--device cuda:1 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log


# --device cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log
# --loggers wandb \
# --limit_train_batches 0.001 \
# --limit_val_batches 0.01 \
