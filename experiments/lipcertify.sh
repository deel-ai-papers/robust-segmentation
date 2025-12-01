# Compare with smoothing certificate
CUDA_VISIBLE_DEVICES=0 python scripts/lipcertify.py \
  --dataset cityscapes --data_root $CITYSCAPES --img_size 1024 \
  --square_imgs --config M2 --type_param ortho --epsilon 0.1 \
  --batch_size 10 --klip
