timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}


expname=eeg_all_1024x8_debug  # K = 1024, dim = 8
expdir=exp_ablation_codebook


mkdir -p exp/${expdir}/${expname}


# --- --- --- --- --- --- --- ---   Train  --- --- --- --- --- --- --- ---  
python main.py \
--expname ${expname} \
--batch_size 8 \
--max_epochs 40 \
--loss_modal_decomp_weight 0 \
--loss_uniform_weight 0 \
--vq_all_token False \
--loggers wandb \
--val_loss True \
--cfg config/cfg_eeg.yaml \
--cfg_model config/cfg_ablation/cfg_model_ablation_all_1024x8.yaml \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--loggers_dir exp/${expdir}/${expname}/log \
--device cuda:4 cuda:5 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log
