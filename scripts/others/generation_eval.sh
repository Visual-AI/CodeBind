BATCH_SIZE=30
DEVICE="cuda:6"
LOG_DIR="application/stable_unclip/logs"
mkdir -p ${LOG_DIR}


# depth generation
python application/imggen_eval.py \
--checkpoint_dir exp/exp_sun/sun_lora_MVQ1024x8_256x8/checkpoints \
--batch_size ${BATCH_SIZE} \
--train_mode lora \
--modality_retrieval depth \
--device ${DEVICE}2>&1 | tee ${LOG_DIR}/eval_depth.log


# audio generation
python application/imggen_eval.py \
--checkpoint_dir .checkpoints/exp_from_zilong/exp_audio/audio_lora_MVQ1024x8_256x8/checkpoints \
--batch_size ${BATCH_SIZE} \
--train_mode lora \
--modality_retrieval audio \
--device ${DEVICE}2>&1 | tee ${LOG_DIR}/eval_audio.log

# thermal generation
python application/imggen_eval.py \
--checkpoint_dir exp/exp_flir/flir_lora_MVQ1024x8_256x8_scale10/checkpoints \
--batch_size ${BATCH_SIZE} \
--train_mode lora \
--modality_retrieval thermal \
--device ${DEVICE}2>&1 | tee ${LOG_DIR}/eval_thermal.log

# vision generation
# python application/stable_unclip/pipeline.py \
# --checkpoint_dir /home/vislab/jieli23/works/vitlens/exp_from_zilong/in1k_headtune_MVQ1024x8_256x8 \
# --data_retrieval .datasets/sampled_in1k_data.csv \
# --train_mode headtune \
# --modality_retrieval vision \
# --embed_type common \
# --device cuda:1

# tactile generation
# python application/stable_unclip/pipeline.py \
# --checkpoint_dir /home/vislab/jieli23/works/imagebind/exp_tag/tag_m_lora_MVQ1024x8_256x8_acc42.6/checkpoints/lora \
# --data_retrieval .datasets/tactile_sample.csv \
# --train_mode lora \
# --modality_retrieval tactile \
# --batch_size 10 \
# --device cuda:7