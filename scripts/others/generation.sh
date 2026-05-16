BATCH_SIZE=10
DEVICE=cuda:0

# # depth generation
python application/pipeline_2.py \
--checkpoint_dir .checkpoints/exp_zeyu/exp_sun/sun_lora_MVQ1024x8_256x8/checkpoints \
--batch_size ${BATCH_SIZE} \
--train_mode lora \
--modality_retrieval depth \
--device ${DEVICE}


# # audio generation
# python application/pipeline_2.py \
# --checkpoint_dir .checkpoints/exp_from_zilong/exp_audio/audio_lora_MVQ1024x8_256x8/checkpoints \
# --batch_size ${BATCH_SIZE} \
# --train_mode lora \
# --modality_retrieval audio \
# --device ${DEVICE}

# thermal generation
# python application/pipeline_2.py \
# --checkpoint_dir .checkpoints/exp_zeyu/exp_flir/flir_lora_MVQ1024x8_256x8_scale10/checkpoints \
# --batch_size ${BATCH_SIZE} \
# --train_mode lora \
# --modality_retrieval thermal \
# --device ${DEVICE}

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