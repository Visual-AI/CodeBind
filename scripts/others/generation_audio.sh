BATCH_SIZE=6
DEVICE=cuda:7

# audio generation
python application/pipeline_2.py \
--checkpoint_dir .checkpoints/exp_from_zilong/exp_audio/audio_lora_MVQ1024x8_256x8/checkpoints \
--batch_size ${BATCH_SIZE} \
--train_mode lora \
--use_baseline \
--modality_retrieval audio \
--device ${DEVICE}