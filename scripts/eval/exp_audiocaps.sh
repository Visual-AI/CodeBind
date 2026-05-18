

# resume_expname can be different from expname, but make sure the checkpoints are exists in resume_expname
# when resume_expname = expname, it means evaluation on the same experiment
# when resume_expname != expname, please set --expname ${expname} to create new directory for evaluation

# # --- --- --- --- --- --- --- ---   test  --- --- --- --- --- --- --- ---  
# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}
# expname=exp/exp_audiocaps/audiocaps_lora_MVQ1024x8_256x8
# python main.py \
# --resume_expname ${expname} \
# --train False \
# --scale_factor 1 \
# --batch_size 12 \
# --cfg config/cfg_audiocaps.yaml \
# --device cuda:0 2>&1 | tee ${expname}/${timestamp}_test.log



# --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- --- 
# set a new expname for evaluation
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_audiocaps/audiocaps_eval_by_audio
mkdir -p ${expname}
python main.py \
--expname ${expname} \
--resume_expname exp/exp_audio/audio_lora_MVQ1024x8_256x8 \
--train False \
--batch_size 12 \
--cfg config/cfg_audiocaps.yaml \
--device cuda:0 2>&1 | tee ${expname}/${timestamp}_test.log



# --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- --- 
# set a new expname for evaluation
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_audiocaps/audiocaps_eval_by_asa
mkdir -p ${expname}
python main.py \
--expname ${expname} \
--resume_expname exp/exp_asa/asa_lora_MVQ1024x8_256x8 \
--train False \
--batch_size 12 \
--cfg config/cfg_audiocaps.yaml \
--device cuda:0 2>&1 | tee ${expname}/${timestamp}_test.log



# --- --- --- --- --- --- --- ---  Test   --- --- --- --- --- --- --- --- 
# set a new expname for evaluation
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_audiocaps/audiocaps_eval_by_audiocaps
mkdir -p ${expname}
python main.py \
--expname ${expname} \
--resume_expname exp/exp_audiocaps/audiocaps_lora_MVQ1024x8_256x8 \
--train False \
--batch_size 12 \
--cfg config/cfg_audiocaps.yaml \
--device cuda:0 2>&1 | tee ${expname}/${timestamp}_test.log
