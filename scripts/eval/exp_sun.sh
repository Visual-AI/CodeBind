# # --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- --- 
# # set a new expname for evaluation
# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}
# expname=exp/exp_ablation/sun_MVQ_seperate
# mkdir -p ${expname}
# python main.py \
# --expname ${expname} \
# --resume_expname exp/exp_ablation/sun_MVQ_seperate \
# --train False \
# --batch_size 12 \
# --cfg config/cfg_sun.yaml \
# --device cuda:5 2>&1 | tee ${expname}/${timestamp}_test.log


# # --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- --- 
# # set a new expname for evaluation
# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}
# expname=exp/exp_ablation/sun_MVQ_typical
# mkdir -p ${expname}
# python main.py \
# --expname ${expname} \
# --resume_expname exp/exp_ablation/sun_MVQ_typical \
# --train False \
# --batch_size 12 \
# --cfg config/cfg_sun.yaml \
# --device cuda:5 2>&1 | tee ${expname}/${timestamp}_test.log


# # --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- --- 
# # set a new expname for evaluation
# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}
# expname=exp/exp_ablation/sun_MVQ_RC
# mkdir -p ${expname}
# python main.py \
# --expname ${expname} \
# --resume_expname exp/exp_ablation/sun_MVQ_RC \
# --train False \
# --batch_size 12 \
# --cfg config/cfg_sun.yaml \
# --device cuda:5 2>&1 | tee ${expname}/${timestamp}_test.log


# # --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- --- 
# # set a new expname for evaluation
# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}
# expname=exp/exp_ablation/sun_MVQ
# mkdir -p ${expname}
# python main.py \
# --expname ${expname} \
# --resume_expname exp/exp_ablation/sun_MVQ \
# --train False \
# --batch_size 12 \
# --cfg config/cfg_sun.yaml \
# --device cuda:5 2>&1 | tee ${expname}/${timestamp}_test.log


# --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- --- 
# set a new expname for evaluation
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_sun/sun_lora_MVQ1024x8_256x8_adaloss
mkdir -p ${expname}
python main.py \
--expname ${expname} \
--resume_expname exp/exp_sun/sun_lora_MVQ1024x8_256x8_adaloss \
--train False \
--batch_size 12 \
--cfg config/cfg_sun.yaml \
--device cuda:6 2>&1 | tee ${expname}/${timestamp}_test.log
