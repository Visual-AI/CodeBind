timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}


expname=flir_lora_MVQ256x1024_64x1024_loss_woreg
expname=flir_lora_MVQ256x1024_64x1024_loss_woreg
expdir=exp_flir


mkdir -p exp/${expdir}/${expname}


# --- --- --- --- --- --- --- ---   Train  --- --- --- --- --- --- --- ---  
python main.py \
--expname ${expname} \
--batch_size 6 \
--max_epochs 30 \
--codevector_dim 1024 \
--codevector_num 256 64 \
--loggers wandb \
--val_loss True \
--loss_vq_reg_weight 0.0 \
--loss_modal_decomp_weight 0.0 \
--loss_uniform_weight 0.0 \
--cfg config/cfg_flir.yaml \
--cfg_model config/cfg_model.yaml \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--loggers_dir exp/${expdir}/${expname}/log \
--device cuda:1 cuda:2 cuda:3 cuda:4 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log