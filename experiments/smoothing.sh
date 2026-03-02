#!/bin/bash

# Train a model with noise augmentation (sigma=0.3) for randomized smoothing
# CUDA_VISIBLE_DEVICES=0 python scripts/train.py --type_param unconstrained \
#    --dataset iiit_pets --data_root $TORCH_DATASETS --img_size 128 --square_imgs \
#    --wandb_log --batch_size 30 --loss tau_cce --tau 1.0 --optimizer adamw --lr 1e-3 \
#    --wd 1e-4 --config M1 --epochs 150 --sigma_train 0.3

# Apply smooth attack and certification with matching sigma
CUDA_VISIBLE_DEVICES=0 python scripts/smooth_attack.py --dataset iiit_pets \
	--data_root $TORCH_DATASETS --img_size 128 --config M1 \
	--type_param unconstrained --batch_size 4 \
	--model_file iiit_pets_unconstrained_M1_tau_cce_tau1.0_nonlip.pth \
	--sigma 0.5 --n0 50 --n 250 --tau 0.0 --alpha 0.01 \
	--epsilon 0.5 --num_steps 40 --n_attack_samples 32 \
	--num_batches 5 --output_csv smooth_noise_trained.csv
