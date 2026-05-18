timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}


expname=sun_MVQ_cmcm
expdir=exp_ablation

mkdir -p exp/${expdir}/${expname}

# --- --- --- --- --- --- --- ---   Train  --- --- --- --- --- --- --- ---  
python main.py \
--expname ${expname} \
--datasets sun_evalnyu \
--batch_size 8 \
--max_epochs 30 \
--loggers wandb \
--val_loss True \
--loss_vq_reg_weight 0.0 \
--cfg config/cfg_sun.yaml \
--cfg_model config/cfg_ablation_old/cfg_model_ablation_MVQ.yaml \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--loggers_dir exp/${expdir}/${expname}/log \
--device cuda:4 cuda:5 cuda:6 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log



