# Visualize results
CUDA_VISIBLE_DEVICES=1 python scripts/infer.py \
  --dataset kvasir --data_root /datasets/shared_datasets/Kvasir-SEG \
  --type_param ortho --img_size 256 --square_imgs \
  --config S --klip --batch_size 3
