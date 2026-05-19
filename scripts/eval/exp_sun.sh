# --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- --- 
# set a new expname for evaluation
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_sun/sun_lora_MVQ1024x8_256x8
mkdir -p ${expname}


python main.py \
--expname ${expname} \
--resume_expname exp/exp_sun/sun_lora_MVQ1024x8_256x8 \
--train False \
--batch_size 12 \
--cfg config/cfg_sun.yaml \
--device cuda:6 2>&1 | tee ${expname}/${timestamp}_test.log
