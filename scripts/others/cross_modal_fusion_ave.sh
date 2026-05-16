
device="cuda:3"

# flir
# baseline
expname=exp/exp_ave/ave_eval_by_imagebind

python cross_modal_fusion.py \
--expname ${expname} \
--fusion_type concat \
--device ${device} \
--logger 

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
expname1=exp/exp_ave/ave_lora_MVQ1024x8_256x8

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