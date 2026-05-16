


expname2=exp/exp_ablation_codebook/sun_all_1024x64x128  # load exising model expname


python application/recon_fdi.py \
--expname ${expname2} \
--prob_modality vision \
--device cuda:0


expname3=exp/exp_ablation_codebook/sun_all_1024x128x128  # load exising model expname


python application/recon_fdi.py \
--expname ${expname3} \
--prob_modality vision \
--device cuda:0