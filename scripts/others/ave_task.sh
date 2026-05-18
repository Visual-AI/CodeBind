python application/ave_event_class.py \
--checkpoint_dir exp/exp_ave/ave_lora_MVQ1024x8_256x8_dense/checkpoints \
--ave_task ave_va \
--dense_text \
--batch_size 100 \
--n_epoch 2 \
--device cuda:3

python application/ave_event_class.py \
--checkpoint_dir exp/exp_ave/ave_lora_MVQ1024x8_256x8_dense/checkpoints \
--ave_task ave_av \
--dense_text \
--batch_size 100 \
--n_epoch 2 \
--device cuda:3