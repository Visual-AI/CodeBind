timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}


expname=sun_all_1024x128x128  # K = 64, dim = 1024
expdir=exp_ablation_codebook


mkdir -p exp/${expdir}/${expname}



python main.py \
--expname ${expname} \
--datasets sun_evalnyu \
--batch_size 8 \
--max_epochs 30 \
--codevector_dim 128 \
--codevector_num 1024 128 \
--loggers wandb \
--val_loss True \
--cfg config/cfg_sun.yaml \
--cfg_model config/cfg_model.yaml \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--loggers_dir exp/${expdir}/${expname}/log \
--device cuda:0 cuda:5 cuda:6 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log

