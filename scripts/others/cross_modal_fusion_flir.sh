
device="cuda:0"

# flir
# baseline
expname=flir_exp03_imagebind_baseline

# python cross_modal_fusion.py \
# --expname ${expname} \
# --fusion_type concat \
# --device ${device} \
# --logger 

python cross_modal_fusion.py \
--expname ${expname} \
--fusion_type sum \
--device ${device} \
--logger 

python cross_modal_fusion.py \
--expname ${expname} \
--fusion_type attention \
--device ${device} \
--logger 

# ours
expname1=flir_exp11_MVQ1024*8_256*8_5class_weight50_trunkrecon_uniloss

python cross_modal_fusion.py \
--expname ${expname1} \
--fusion_type concat \
--device ${device} \
--logger 

python cross_modal_fusion.py \
--expname ${expname1} \
--fusion_type sum \
--device ${device} \
--logger 

python cross_modal_fusion.py \
--expname ${expname1} \
--fusion_type attention \
--device ${device} \
--logger 

# only use common embedding
python cross_modal_fusion.py \
--expname ${expname1} \
--fusion_type concat \
--use_common_embed \
--device ${device} \
--logger 

python cross_modal_fusion.py \
--expname ${expname1} \
--fusion_type sum \
--use_common_embed \
--device ${device} \
--logger 

python cross_modal_fusion.py \
--expname ${expname1} \
--fusion_type attention \
--use_common_embed \
--device ${device} \
--logger 