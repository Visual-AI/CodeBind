# sun
timestamp=$(date "+%Y%m%d_%H%M%S")
echo ${timestamp}
expname=exp/exp_sun/sun_eval_by_imagebind
mkdir -p ${expname}
python main.py \
--expname ${expname} \
--train False \
--datasets sun_evalnyu \
--batch_size 40 \
--cfg config/cfg_sun.yaml \
--device cuda:3 2>&1 | tee ${expname}/${timestamp}_test.log

# # asa
# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}
# expname=exp/exp_asa/asa_eval_by_imagebind
# mkdir -p ${expname}
# python main.py \
# --expname ${expname} \
# --train False \
# --batch_size 40 \
# --cfg config/cfg_asa.yaml \
# --device cuda:3 2>&1 | tee ${expname}/${timestamp}_test.log

# audiocaps
# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}
# expname=exp/exp_audiocaps/audiocaps_eval_by_imagebind
# mkdir -p ${expname}
# python main.py \
# --expname ${expname} \
# --train False \
# --batch_size 40 \
# --cfg config/cfg_audiocaps.yaml \
# --device cuda:3 2>&1 | tee ${expname}/${timestamp}_test.log

# ave
# timestamp=$(date "+%Y%m%d_%H%M%S")
# echo ${timestamp}
# expname=exp/exp_ave/ave_eval_by_imagebind
# mkdir -p ${expname}
# python main.py \
# --expname ${expname} \
# --train False \
# --batch_size 80 \
# --cfg config/cfg_ave.yaml \
# --device cuda:3 2>&1 | tee ${expname}/${timestamp}_test.log