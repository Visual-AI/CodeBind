# --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- ---
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_esc/esc_eval_by_asa
mkdir -p ${expname}

python main.py \
--expname ${expname} \
--resume_expname exp/exp_asa/asa_lora_MVQ1024x8_256x8 \
--train False \
--batch_size 40 \
--cfg config/cfg_esc.yaml \
--device cuda:1 2>&1 | tee ${expname}/${timestamp}_test.log

