# # --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- ---
# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}
# expname=exp/exp_esc/esc_eval_by_audio
# mkdir -p ${expname}
  
# python main.py \
# --expname ${expname} \
# --resume_expname exp/exp_audio/audio_lora_MVQ1024x8_256x8 \
# --train False \
# --batch_size 40 \
# --cfg config/cfg_esc.yaml \
# --device cuda:0 2>&1 | tee ${expname}/${timestamp}_test.log



# # --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- ---
# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}
# expname=exp/exp_esc/esc_eval_by_asa
# mkdir -p ${expname}
# python main.py \
# --expname ${expname} \
# --resume_expname exp/exp_asa/asa_lora_MVQ1024x8_256x8 \
# --train False \
# --batch_size 40 \
# --cfg config/cfg_esc.yaml \
# --device cuda:0 2>&1 | tee ${expname}/${timestamp}_test.log


# --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- ---
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_esc/esc_eval_by_asa_zl
mkdir -p ${expname}
python main.py \
--expname ${expname} \
--resume_expname /home/vislab/jieli23/works/vitlens/exp_from_zilong/exp_asa/asa_lora_MVQ1024x8_256x8 \
--train False \
--batch_size 40 \
--cfg config/cfg_esc.yaml \
--device cuda:1 2>&1 | tee ${expname}/${timestamp}_test.log


# --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- ---
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_esc/esc_eval_by_audio_zl
mkdir -p ${expname}
python main.py \
--expname ${expname} \
--resume_expname /home/vislab/jieli23/works/vitlens/exp_from_zilong/exp_audio/audio_lora_MVQ1024x8_256x8 \
--train False \
--batch_size 40 \
--cfg config/cfg_esc.yaml \
--device cuda:1 2>&1 | tee ${expname}/${timestamp}_test.log


