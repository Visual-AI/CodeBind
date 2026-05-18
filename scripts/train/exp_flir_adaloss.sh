timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}


expname=flir_lora_MVQ1024x8_256x8_adaloss
expdir=exp_flir


mkdir -p exp/${expdir}/${expname}


# --- --- --- --- --- --- --- ---   Train  --- --- --- --- --- --- --- ---  
python main_lossbalance.py \
--expname ${expname} \
--batch_size 8 \
--max_epochs 30 \
--loggers wandb \
--val_loss True \
--loss_vq_weight True \
--loss_vq_reg_weight True \
--loss_cmcm_weight True \
--loss_modal_decomp_weight True \
--loss_uniform_weight True \
--cfg config/cfg_flir.yaml \
--cfg_model config/cfg_model.yaml \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--loggers_dir exp/${expdir}/${expname}/log \
--device cuda:4 cuda:5 cuda:6 cuda:7 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log
