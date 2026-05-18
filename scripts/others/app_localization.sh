python application/embedding_play.py \
--modality_retrieval depth \
--checkpoint_dir exp/sun_exp20_MVQ1024\*8_256\*8_2anchor_all_laterlayer_postproc_normcode_intra_kmeans_weight1000_0.05_0.1/checkpoints/lora \
--data_retrieval .datasets/sampled_data.csv \
--train_mode lora \
--device cuda:0