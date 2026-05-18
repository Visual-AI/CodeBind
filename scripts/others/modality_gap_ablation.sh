# # flir
# expname1=exp/exp_ablation/flir_MVQ_seperate_1024*8  # load exising model expname
# python modality_gap.py \
# --expname ${expname1} \
# --batch_size 10 \
# --device cuda:1

# # sun
# expname2=exp/exp_ablation/sun_MVQ_seperate  # load exising model expname
# python modality_gap.py \
# --expname ${expname2} \
# --batch_size 10 \
# --device cuda:0

# # flir base
# expname3=exp/exp_flir/flir_lora_MVQ1024x8_256x8  # load exising model expname
# python modality_gap.py \
# --expname ${expname3} \
# --batch_size 10 \
# --device cuda:0

# # sun base
# expname4=/home/vislab/jieli23/works/imagebind/exp_sun/sun_lora_MVQ1024x1024_256x1024_fromp365  # load exising model expname
# python modality_gap.py \
# --expname ${expname4} \
# --batch_size 10 \
# --device cuda:0

# # flir
# expname1=exp/exp_flir/flir_lora_MVQ256x1024_64x1024_loss_woreg  # load exising model expname
# python modality_gap.py \
# --expname ${expname1} \
# --batch_size 10 \
# --device cuda:5

# python codebook_anaylysis.py \
# --expname ${expname1} \
# --visual_codedist \
# --device cuda:5

# sun
expname2=exp/exp_sun/sun_lora_MVQ256x1024_64x1024_loss_woreg  # load exising model expname
python modality_gap.py \
--expname ${expname2} \
--batch_size 10 \
--device cuda:5

python codebook_analysis.py \
--expname ${expname2} \
--visual_codedist \
--device cuda:5