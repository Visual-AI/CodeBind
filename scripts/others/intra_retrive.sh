# stanford dog

# python application/intra_modality_retrieval.py \
# --checkpoint_dir /home/vislab/jieli23/works/vitlens/exp_from_zilong/in1k_headtune_MVQ1024x8_256x8 \
# --modality_retrieval vision \
# --train_mode headtune \
# --data_retrieval application/intra_retrieval/datasets/imagenet_dog.csv application/intra_retrieval/datasets/imagenet_dog.csv \
# --use_baseline \
# --embed_type common_common \
# --device cuda:0


python application/intra_modality_retrieval.py \
--checkpoint_dir /home/vislab/jieli23/works/vitlens/exp_from_zilong/in1k_headtune_MVQ1024x8_256x8 \
--modality_retrieval vision \
--train_mode headtune \
--data_retrieval application/intra_retrieval/datasets/imagenet_dog.csv application/intra_retrieval/datasets/imagenet_dog.csv \
--embed_type concat_concat \
--device cuda:0

python application/intra_modality_retrieval.py \
--checkpoint_dir /home/vislab/jieli23/works/vitlens/exp_from_zilong/in1k_headtune_MVQ1024x8_256x8 \
--modality_retrieval vision \
--train_mode headtune \
--data_retrieval application/intra_retrieval/datasets/imagenet_dog.csv application/intra_retrieval/datasets/imagenet_dog.csv \
--embed_type common_common \
--device cuda:0

# python application/intra_modality_retrieval.py \
# --checkpoint_dir /home/vislab/jieli23/works/vitlens/exp_from_zilong/in1k_headtune_MVQ1024x8_256x8 \
# --modality_retrieval vision \
# --train_mode headtune \
# --data_retrieval application/intra_retrieval/datasets/imagenet_dog.csv application/intra_retrieval/datasets/imagenet_dog.csv \
# --embed_type specific_specific \
# --device cuda:0

# oxford cat
# python application/intra_modality_retrieval.py \
# --checkpoint_dir /home/vislab/jieli23/works/vitlens/exp_from_zilong/in1k_headtune_MVQ1024x8_256x8 \
# --modality_retrieval vision \
# --train_mode headtune \
# --data_retrieval application/intra_retrieval/datasets/oxford_pet_cat.csv application/intra_retrieval/datasets/oxford_pet_cat.csv \
# --use_baseline \
# --embed_type common_common \
# --device cuda:0

python application/intra_modality_retrieval.py \
--checkpoint_dir /home/vislab/jieli23/works/vitlens/exp_from_zilong/in1k_headtune_MVQ1024x8_256x8 \
--modality_retrieval vision \
--train_mode headtune \
--data_retrieval application/intra_retrieval/datasets/oxford_pet_cat.csv application/intra_retrieval/datasets/oxford_pet_cat.csv \
--embed_type concat_concat \
--device cuda:0

python application/intra_modality_retrieval.py \
--checkpoint_dir /home/vislab/jieli23/works/vitlens/exp_from_zilong/in1k_headtune_MVQ1024x8_256x8 \
--modality_retrieval vision \
--train_mode headtune \
--data_retrieval application/intra_retrieval/datasets/oxford_pet_cat.csv application/intra_retrieval/datasets/oxford_pet_cat.csv \
--embed_type common_common \
--device cuda:0

# python application/intra_modality_retrieval.py \
# --checkpoint_dir /home/vislab/jieli23/works/vitlens/exp_from_zilong/in1k_headtune_MVQ1024x8_256x8 \
# --modality_retrieval vision \
# --train_mode headtune \
# --data_retrieval application/intra_retrieval/datasets/oxford_pet_cat.csv application/intra_retrieval/datasets/oxford_pet_cat.csv \
# --embed_type specific_specific \
# --device cuda:0


# oxford dog
# python application/intra_modality_retrieval.py \
# --checkpoint_dir /home/vislab/jieli23/works/vitlens/exp_from_zilong/in1k_headtune_MVQ1024x8_256x8 \
# --modality_retrieval vision \
# --train_mode headtune \
# --data_retrieval application/intra_retrieval/datasets/oxford_pet_dog.csv application/intra_retrieval/datasets/oxford_pet_dog.csv \
# --use_baseline \
# --embed_type common_common \
# --device cuda:0

python application/intra_modality_retrieval.py \
--checkpoint_dir /home/vislab/jieli23/works/vitlens/exp_from_zilong/in1k_headtune_MVQ1024x8_256x8 \
--modality_retrieval vision \
--train_mode headtune \
--data_retrieval application/intra_retrieval/datasets/oxford_pet_dog.csv application/intra_retrieval/datasets/oxford_pet_dog.csv \
--embed_type concat_concat \
--device cuda:0

python application/intra_modality_retrieval.py \
--checkpoint_dir /home/vislab/jieli23/works/vitlens/exp_from_zilong/in1k_headtune_MVQ1024x8_256x8 \
--modality_retrieval vision \
--train_mode headtune \
--data_retrieval application/intra_retrieval/datasets/oxford_pet_dog.csv application/intra_retrieval/datasets/oxford_pet_dog.csv \
--embed_type common_common \
--device cuda:0

# python application/intra_modality_retrieval.py \
# --checkpoint_dir /home/vislab/jieli23/works/vitlens/exp_from_zilong/in1k_headtune_MVQ1024x8_256x8 \
# --modality_retrieval vision \
# --train_mode headtune \
# --data_retrieval application/intra_retrieval/datasets/oxford_pet_dog.csv application/intra_retrieval/datasets/oxford_pet_dog.csv \
# --embed_type specific_specific \
# --device cuda:0