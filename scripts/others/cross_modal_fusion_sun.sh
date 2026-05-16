
device="cuda:7"

# sun
# baseline
expname=sun_exp_pretrain_eval

python cross_modal_fusion.py \
--expname ${expname} \
--fusion_type concat \
--device ${device} \
--logger | tee ./exp/${expname}/train_fusion.log

python cross_modal_fusion.py \
--expname ${expname} \
--fusion_type sum \
--device ${device} \
--logger | tee ./exp/${expname}/train_fusion.log

python cross_modal_fusion.py \
--expname ${expname} \
--fusion_type attention \
--device ${device} \
--logger | tee ./exp/${expname}/train_fusion.log

# ours
expname1=sun_exp25_MVQ1024\*8_256\*8_2anchor_intra_weight1000_0.05_0.1_trunkrecon

python cross_modal_fusion.py \
--expname ${expname1} \
--fusion_type concat \
--device ${device} \
--logger | tee ./exp/${expname1}/train_fusion.log

python cross_modal_fusion.py \
--expname ${expname1} \
--fusion_type sum \
--device ${device} \
--logger | tee ./exp/${expname1}/train_fusion.log

python cross_modal_fusion.py \
--expname ${expname1} \
--fusion_type attention \
--device ${device} \
--logger | tee ./exp/${expname1}/train_fusion.log

# only use common embedding
python cross_modal_fusion.py \
--expname ${expname1} \
--fusion_type concat \
--use_common_embed \
--device ${device} \
--logger | tee ./exp/${expname1}/train_fusion.log

python cross_modal_fusion.py \
--expname ${expname1} \
--fusion_type sum \
--use_common_embed \
--device ${device} \
--logger | tee ./exp/${expname1}/train_fusion.log

python cross_modal_fusion.py \
--expname ${expname1} \
--fusion_type attention \
--use_common_embed \
--device ${device} \
--logger | tee ./exp/${expname1}/train_fusion.log