# --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- ---
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_clotho/clotho_eval_by_audio
mkdir -p ${expname}
python main.py \
--expname ${expname} \
--resume_expname exp/exp_audio/audio_lora_MVQ1024x8_256x8 \
--train False \
--batch_size 40 \
--cfg config/cfg_clotho.yaml \
--device cuda:0 2>&1 | tee ${expname}/${timestamp}_test.log



# --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- ---
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_clotho/clotho_eval_by_asa
mkdir -p ${expname}
python main.py \
--expname ${expname} \
--resume_expname exp/exp_asa/asa_lora_MVQ1024x8_256x8 \
--train False \
--batch_size 40 \
--cfg config/cfg_clotho.yaml \
--device cuda:0 2>&1 | tee ${expname}/${timestamp}_test.log



# --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- --- 
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_clotho/clotho_eval_by_audiocaps
mkdir -p ${expname}
python main.py \
--expname ${expname} \
--resume_expname exp/exp_audiocaps/audiocaps_lora_MVQ1024x8_256x8 \
--train False \
--batch_size 40 \
--cfg config/cfg_clotho.yaml \
--device cuda:0 2>&1 | tee ${expname}/${timestamp}_test.log