BATCH_SIZE=6
DEVICE=cuda:6

# thermal generation
python application/pipeline_2.py \
--checkpoint_dir exp/exp_flir/flir_lora_MVQ1024x8_256x8_scale10/checkpoints \
--batch_size ${BATCH_SIZE} \
--train_mode lora \
--use_baseline \
--modality_retrieval thermal \
--device ${DEVICE}