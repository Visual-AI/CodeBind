# asa
expname1=/home/vislab/jieli23/works/vitlens/exp_from_zilong/exp_asa/asa_lora_MVQ1024x8_256x8  # load exising model expname

python modality_gap.py \
--expname ${expname1} \
--device cuda:1

# python codebook_analysis.py \
# --expname ${expname1} \
# --batch_size 10 \
# --device cuda:3

# audiocaps
# expname2=/home/vislab/jieli23/works/vitlens/exp_from_zilong/exp_audiocaps/audiocaps_lora_MVQ1024x8_256x8  # load exising model expname

# python modality_gap.py \
# --expname ${expname2} \
# --device cuda:3

# python codebook_analysis.py \
# --expname ${expname2} \
# --batch_size 10 \
# --device cuda:3

# depth
# expname3=/home/vislab/jieli23/works/imagebind/exp_sun/sun_lora_MVQ1024x1024_256x1024_fromp365  # load exising model expname
# python codebook_analysis.py \
# --expname ${expname3} \
# --batch_size 10 \
# --device cuda:4

# expname4=exp/exp_sun/sun_lora_MVQ256x1024_64x1024
# python codebook_analysis.py \
# --expname ${expname4} \
# --batch_size 10 \
# --device cuda:4


# modality gap baseline
expname3=exp/exp_asa/asa_eval_by_imagebind

python modality_gap.py \
--expname ${expname3} \
--device cuda:1

# expname4=exp/exp_audiocaps/audiocaps_eval_by_imagebind

# python modality_gap.py \
# --expname ${expname4} \
# --device cuda:5