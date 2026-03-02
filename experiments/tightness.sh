#################################### LipNet ###################################

##### TRAINING #####

# LipNet - Tau 3.0
# CUDA_VISIBLE_DEVICES=1 python scripts/train.py --type_param ortho \
#    --dataset iiit_pets --data_root $TORCH_DATASETS --img_size 128 --square_imgs \
#    --wandb_log --batch_size 30 --loss tau_cce --tau 3.0  --optimizer adamw --lr 1e-3 \
#    --wd 1e-4 --config M1 --epochs 150 --sigma_train 0.0 --klip 
# LipNet - Tau = 8.0
# CUDA_VISIBLE_DEVICES=1 python scripts/train.py --type_param ortho \
#    --dataset iiit_pets --data_root $TORCH_DATASETS --img_size 128 --square_imgs \
#    --wandb_log --batch_size 30 --loss tau_cce --tau 8.0  --optimizer adamw --lr 1e-3 \
#    --wd 1e-4 --config M1 --epochs 150 --sigma_train 0.0 --klip 
# LipNet - Tau 16.0
# CUDA_VISIBLE_DEVICES=1 python scripts/train.py --type_param ortho \
#    --dataset iiit_pets --data_root $TORCH_DATASETS --img_size 128 --square_imgs \
#    --wandb_log --batch_size 30 --loss tau_cce --tau 16.0  --optimizer adamw --lr 1e-3 \
#    --wd 1e-4 --config M1 --epochs 150 --sigma_train 0.0 --klip 
# CUDA_VISIBLE_DEVICES=1 python scripts/train.py --type_param unconstrained \
#    --dataset iiit_pets --data_root $TORCH_DATASETS --img_size 128 --square_imgs \
#    --wandb_log --batch_size 30 --loss tau_cce --tau 1.0  --optimizer adamw --lr 1e-3 \
#    --wd 1e-4 --config M1 --epochs 150 --sigma_train 0.3

#!/bin/bash

#################################### LipNet ###################################

# --- EXPERIMENT PARAMETERS ---
NUM_BATCHES_LIP=5
NUM_BATCHES_SMOOTH=50
NUM_STEPS_ATTACK=50
NUM_STEPS_SMOOTH_ATTACK=50
ALPHA=0.001

##### ATTACKING #####

for EPSILON in 0.1 0.5 1.0 1.5; do
    echo "Running attacks for Epsilon = $EPSILON"

    # 1. Standard Attack (Iterating over Tau)
    for TAU in 3.0 8.0 16.0; do
        CUDA_VISIBLE_DEVICES=1 python scripts/attack.py --dataset iiit_pets \
            --data_root $TORCH_DATASETS --img_size 128 --config M1 --klip \
            --batch_size 100 --epsilon $EPSILON --certificate q1 \
            --num_batches $NUM_BATCHES_LIP --num_steps $NUM_STEPS_ATTACK --type_param ortho \
            --model_file iiit_pets_ortho_M1_tau_cce_tau${TAU}_rms.pth \
            --output_csv data/tightness.csv
    done

    # 2. Smooth Attack - Matching sigmas mapped per epsilon
    case $EPSILON in
        0.1) SIGMAS="0.1 0.2 0.3 0.4" ;;
        0.5) SIGMAS="0.2 0.3 0.4 0.5" ;;
        1.0) SIGMAS="0.4 0.5 0.6 0.8" ;;
        1.5) SIGMAS="0.4 0.5 0.6 0.8" ;;
        *)   echo "Error: No sigmas defined for EPSILON $EPSILON"; exit 1 ;;
    esac

    for SIGMA in $SIGMAS; do
        CUDA_VISIBLE_DEVICES=0 python scripts/smooth_attack.py --dataset iiit_pets \
            --data_root $TORCH_DATASETS --img_size 128 --config M1 \
            --type_param unconstrained --batch_size 10 \
            --model_file iiit_pets_unconstrained_M1_tau_cce_tau1.0_nonlip.pth \
            --sigma $SIGMA --n0 50 --n 100 --tau 0.0 --alpha $ALPHA \
            --epsilon $EPSILON --num_steps $NUM_STEPS_SMOOTH_ATTACK --n_attack_samples 32 \
            --num_batches $NUM_BATCHES_SMOOTH --output_csv data/smooth_tightness.csv
    done

    # 3. AllNet attack - Only empirical
    CUDA_VISIBLE_DEVICES=1 python scripts/attack.py --dataset iiit_pets \
        --data_root $TORCH_DATASETS --img_size 128 --config M1 \
        --batch_size 100 --epsilon $EPSILON --certificate q1 \
        --num_batches $NUM_BATCHES_LIP --num_steps $NUM_STEPS_ATTACK --type_param unconstrained \
        --model_file iiit_pets_unconstrained_M1_tau_cce_tau1.0_nonlip.pth \
        --output_csv data/tightness.csv

done
