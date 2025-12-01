# Kvasir (LipNet)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --dataset kvasir --data_root $KVASIR_SEG --epochs 500 \
  --img_size 256 --type_param ortho --loss tau_cce --tau 8.0 \
  --optimizer adamw --lr 5e-4 --wd 1e-6 --config S --batch_size 32 \
  --klip --wandb_log

# IIIT Pets (LipNet)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --type_param ortho \
   --dataset iiit_pets --data_root $TORCH_DATASETS --img_size 128 --square_imgs \
   --wandb_log --batch_size 30 --loss tau_cce --tau 8.0  --optimizer adamw --lr 1e-3 \
   --wd 1e-4 --config M1 --epochs 150 --sigma_train 0.0 --klip 

# Cityscapes (LipNet)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --type_param ortho \
   --dataset cityscapes --data_root $CITYSCAPES --img_size 1024 --square_imgs \
   --wandb_log --batch_size 8 --loss tau_cce --tau 20.0  --optimizer adamw --lr 1e-3 \
   --wd 1e-4 --config M1 --epochs 200 --sigma_train 0.0

# Cityscapes (AllNet)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --type_param unconstrained \
   --dataset cityscapes --data_root $CITYSCAPES --img_size 1024 --square_imgs \
   --wandb_log --batch_size 8 --loss tau_cce --tau 1.0  --optimizer adamw --lr 1e-3 \
   --wd 1e-4 --config M1 --epochs 200 --sigma_train 0.2
