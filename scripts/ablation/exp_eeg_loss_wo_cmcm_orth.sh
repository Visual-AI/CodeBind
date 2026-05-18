timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}

expname=eeg_lora_MVQ1024x8_256x8_loss_wo_cmcm_orth  # lora_MVQ1024x8_256x8
expdir=exp_eeg


mkdir -p exp/${expdir}/${expname}


python main.py \
--expname ${expname} \
--datasets eeg \
--batch_size 6 \
--max_epochs 30 \
--loggers wandb \
--val_loss True \
--loss_cmcm_weight 0.0 \
--loss_modal_decomp_weight 0.0 \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--cfg config/cfg_eeg.yaml \
--cfg_model config/cfg_model.yaml \
--device cuda:5 cuda:6 cuda:7 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log


# --loggers wandb \
# --limit_train_batches 0.1 \
# --limit_val_batches 0.1 \