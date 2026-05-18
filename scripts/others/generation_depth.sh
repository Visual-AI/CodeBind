BATCH_SIZE=10
DEVICE=cuda:0

# # depth generation
python application/pipeline_2.py \
--checkpoint_dir .checkpoints/exp_zeyu/exp_sun/sun_lora_MVQ1024x8_256x8/checkpoints \
--batch_size ${BATCH_SIZE} \
--train_mode lora \
--use_baseline \
--modality_retrieval depth \
--device ${DEVICE}