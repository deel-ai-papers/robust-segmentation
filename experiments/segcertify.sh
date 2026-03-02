##################################################### IIIT Pets ################################################

# Epsilon = 0.1 / MC 50 / Sigma 0.3
CUDA_VISIBLE_DEVICES=0 python scripts/segcertify.py \
  --dataset iiit_pets --data_root $TORCH_DATASETS --img_size 128 --square_imgs --config M1 \
  --type_param ortho --epsilon 0.1 --sigma 0.3 --n_samples 50  \
  --alpha 0.01 --prop_voting 0.1 --batch_size 10 --max_bs 100 --num_batches_eval 40 --klip

# Epsilon = 0.1 / MC 300 / Sigma 0.08
CUDA_VISIBLE_DEVICES=0 python scripts/segcertify.py \
  --dataset iiit_pets --data_root $TORCH_DATASETS --img_size 128 --square_imgs --config M1 \
  --type_param ortho --epsilon 0.1 --sigma 0.08 --n_samples 300  \
  --alpha 0.01 --prop_voting 0.1 --batch_size 10 --max_bs 100 --num_batches_eval 40 --klip

# Epsilon = 0.1 / MC 1000 / Sigma 0.05
CUDA_VISIBLE_DEVICES=0 python scripts/segcertify.py \
  --dataset iiit_pets --data_root $TORCH_DATASETS --img_size 128 --square_imgs --config M1 \
  --type_param ortho --epsilon 0.1 --sigma 0.05 --n_samples 1000  \
  --alpha 0.01 --prop_voting 0.1 --batch_size 10 --max_bs 100 --num_batches_eval 40 --klip

# Epsilon = 0.1 / MC 10_000 / Sigma 0.035
CUDA_VISIBLE_DEVICES=0 python scripts/segcertify.py \
  --dataset iiit_pets --data_root $TORCH_DATASETS --img_size 128 --square_imgs --config M1 \
  --type_param ortho --epsilon 0.1 --sigma 0.035 --n_samples 10000  \
  --alpha 0.01 --prop_voting 0.1 --batch_size 10 --max_bs 100 --num_batches_eval 40 --klip


##################################################### Cityscapes ##############################################

# Epsilon = 0.17 / MC 120 / Sigma = 0.2
CUDA_VISIBLE_DEVICES=0 python scripts/segcertify.py \
  --dataset cityscapes --data_root ../DATA/Cityscapes/ --img_size 1024 --square_imgs --config M2 \
  --type_param unconstrained --epsilon 0.17 --sigma 0.2 --n_samples 120  \
  --alpha 0.001 --prop_voting 0.1 --batch_size 10 --max_bs 4 --num_batches_eval 10

# Epsilon = 0.1 / MC 60 / Sigma = 0.3
CUDA_VISIBLE_DEVICES=0 python scripts/segcertify.py \
  --dataset cityscapes --data_root ../DATA/Cityscapes/ --img_size 1024 --square_imgs --config M2 \
  --type_param unconstrained --epsilon 0.1 --sigma 0.3 --n_samples 60  \
  --alpha 0.001 --prop_voting 0.1 --batch_size 10 --max_bs 4 --num_batches_eval 10 

# Epsilon = 0.1 / MC 80 / Sigma = 0.2
CUDA_VISIBLE_DEVICES=0 python scripts/segcertify.py \
  --dataset cityscapes --data_root ../DATA/Cityscapes/ --img_size 1024 --square_imgs --config M2 \
  --type_param unconstrained --epsilon 0.1 --sigma 0.2 --n_samples 80  \
  --alpha 0.001 --prop_voting 0.1 --batch_size 10 --max_bs 4 --num_batches_eval 10 

# Epsilon = 0.17 / MC 60 / Sigma = 0.4
CUDA_VISIBLE_DEVICES=0 python scripts/segcertify.py \
  --dataset cityscapes --data_root ../DATA/Cityscapes/ --img_size 1024 --square_imgs --config M2 \
  --type_param unconstrained --epsilon 0.17 --sigma 0.4 --n_samples 60  \
  --alpha 0.001 --prop_voting 0.1 --batch_size 10 --max_bs 4 --num_batches_eval 10 
