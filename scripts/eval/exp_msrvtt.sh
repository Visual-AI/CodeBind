
# --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- ---  
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_msrvtt/msrvtt_eval_by_in1k
mkdir -p ${expname}
python main.py \
--expname ${expname} \
--resume_expname exp/exp_in1k/in1k_headtune_MVQ1024x8_256x8 \
--train False \
--batch_size 40 \
--cfg config/cfg_msrvtt.yaml \
--device cuda:0 2>&1 | tee ${expname}/${timestamp}_test.log



# --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- ---  
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_msrvtt/msrvtt_eval_by_k400
mkdir -p ${expname}
python main.py \
--expname ${expname} \
--resume_expname exp/exp_k400/k400_headtune_MVQ1024x8_256x8 \
--train False \
--batch_size 40 \
--cfg config/cfg_msrvtt.yaml \
--device cuda:0 2>&1 | tee ${expname}/${timestamp}_test.log



