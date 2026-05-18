timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}


expname=sun_all_1024x512  # K = 1024, dim = 512
expdir=exp_ablation_codebook


mkdir -p exp/${expdir}/${expname}


# --- --- --- --- --- --- --- ---   Train  --- --- --- --- --- --- --- ---  
python main.py \
--expname ${expname} \
--batch_size 8 \
--max_epochs 30 \
--loggers wandb \
--val_loss True \
--cfg config/cfg_sun.yaml \
--cfg_model config/cfg_ablation/cfg_model_ablation_all_1024x512.yaml \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--loggers_dir exp/${expdir}/${expname}/log \
--device cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log
