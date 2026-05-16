# --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- ---
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_clotho/clotho_eval_by_audio_zl_merge
mkdir -p ${expname}
python main.py \
--expname ${expname} \
--resume_expname /home/vislab/jieli23/works/vitlens/exp_from_zilong/exp_audio/audio_lora_MVQ1024x8_256x8 \
--train False \
--batch_size 10 \
--cfg config/cfg_clotho.yaml \
--device cuda:2 2>&1 | tee ${expname}/${timestamp}_test.log



# # --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- ---
# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}
# expname=exp/exp_clotho/clotho_eval_by_asa_zl_merge
# mkdir -p ${expname}
# python main.py \
# --expname ${expname} \
# --resume_expname /home/vislab/jieli23/works/vitlens/exp_from_zilong/exp_asa/asa_lora_MVQ1024x8_256x8 \
# --train False \
# --batch_size 10 \
# --cfg config/cfg_clotho.yaml \
# --device cuda:2 2>&1 | tee ${expname}/${timestamp}_test.log



# # --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- --- 
# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}
# expname=exp/exp_clotho/clotho_eval_by_audiocaps_zl_merge
# mkdir -p ${expname}
# python main.py \
# --expname ${expname} \
# --resume_expname /home/vislab/jieli23/works/vitlens/exp_from_zilong/exp_audiocaps/audiocaps_lora_MVQ1024x8_256x8 \
# --train False \
# --batch_size 10 \
# --cfg config/cfg_clotho.yaml \
# --device cuda:2 2>&1 | tee ${expname}/${timestamp}_test.log