device="cuda:0"

# flir
# baseline
# expname=flir_exp03_imagebind_baseline

# python linear_prob.py \
# --expname ${expname} \
# --device ${device} \
# --logger | tee ./exp/${expname}/train_lp.log


# ours
expname1=flir_exp11_MVQ1024*8_256*8_5class_weight50_trunkrecon_uniloss

# python linear_prob.py \
# --expname ${expname1} \
# --device ${device} \
# --logger | tee ./exp/${expname1}/train_lp.log

python linear_prob.py \
--expname ${expname1} \
--use_common_embed \
--device ${device} \
--logger | tee ./exp/${expname1}/train_lp_common.log
