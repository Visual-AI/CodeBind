timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}


expname=flir_lora_MVQ1024x8_256x8_VQ4cls_cache_lr8e-4_bs16
expdir=exp_flir


mkdir -p exp/${expdir}/${expname}


# --- --- --- --- --- --- --- ---   Train  --- --- --- --- --- --- --- ---  
python main.py \
--expname ${expname} \
--scale_factor 1 \
--batch_size 16 \
--max_epochs 40 \
--num_workers 16 \
--flir_cache /home/jieli23/dataset/FLIR_ADAS_v2 \
--lr 8e-4 \
--lr_new 5e-4 \
--train True \
--train_mode lora \
--lora_module_mode all \
--lora_layer_idxs_audio 6 7 8 9 10 11 \
--val_loss True \
--modality_pair vision thermal text \
--modality_train vision thermal text \
--modality_reconstruction vision thermal \
--codevector_dim 8 \
--loss_vq_weight 1000.0 \
--loss_cmcm_weight 0.05 \
--vq_all_token False \
--modal_decomp_loss False \
--loss_modal_decomp_weight 0.2 \
--use_postprocessors_outencoder False \
--learnable_postprocessors True \
--norm_code True \
--kmeans_init True \
--intra_anchor_align True \
--cfg config/cfg_flir.yaml \
--cfg_model config/cfg_model.yaml \
--checkpoint_dir exp/${expdir}/${expname}/checkpoints \
--log_img_dir exp/${expdir}/${expname}/log_img \
--lightning_log exp/${expdir}/${expname}/lightning_log \
--save_cfg_filepath exp/${expdir}/${expname}/cfg_all_train.yaml \
--loggers_dir exp/${expdir}/${expname}/log \
--device cuda:0 2>&1 | tee ./exp/${expdir}/${expname}/${timestamp}_train.log


# --cfg_model config/cfg_model.yaml \

# --loggers wandb \