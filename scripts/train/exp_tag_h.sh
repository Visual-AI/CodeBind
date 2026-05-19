timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}

expname=tag_h_lora_MVQ1024x8_256x8
expdir=exp_tag


mkdir -p exp/${expdir}/${expname}


python main.py \
--expname ${expname} \
--datasets tag_h \
--batch_size 6 \
--max_epochs 40 \
--loggers wandb \
--wandb_entity codebind \
--val_loss True \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--cfg config/cfg_tag.yaml \
--cfg_model config/cfg_model.yaml \
--device cuda:0 cuda:1 cuda:2 cuda:3 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log


# --loggers wandb \
# --limit_train_batches 0.001 \
# --limit_val_batches 0.01 \