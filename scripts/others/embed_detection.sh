
# depth
# python application/embedding_play.py \
# --checkpoint_dir /home/vislab/jieli23/works/imagebind/exp_sun/sun_lora_MVQ1024x1024_256x1024_fromp365/checkpoints \
# --modality_retrieval depth \
# --train_mode lora \
# --data_retrieval .datasets/sampled_data.csv \
# --device cuda:3

# audio
# python application/embedding_play.py \
# --checkpoint_dir /home/vislab/jieli23/works/vitlens/exp_from_zilong/exp_audio/audio_lora_MVQ1024x8_256x8/checkpoints \
# --modality_retrieval audio \
# --train_mode lora \
# --data_retrieval .datasets/sampled_data.csv \
# --image_dir application/embed_detection/ori_images_tag \
# --device cuda:7

# thermal
# python application/embedding_play.py \
# --checkpoint_dir exp/exp_flir/flir_lora_MVQ1024x8_256x8/checkpoints \
# --modality_retrieval thermal \
# --train_mode lora \
# --data_retrieval .datasets/sampled_data.csv \
# --image_dir application/embed_detection/ori_images_thermal \
# --device cuda:7

# tactile
python application/embedding_play.py \
--checkpoint_dir /home/vislab/jieli23/works/imagebind/exp_tag/tag_m_lora_MVQ1024x8_256x8_acc42.6/checkpoints/lora \
--modality_retrieval tactile \
--train_mode lora \
--data_retrieval .datasets/tactile_sample.csv \
--image_dir application/embed_detection/ori_images_tag \
--device cuda:7