# Certify the worst case FNR measures
CUDA_VISIBLE_DEVICES=0 python scripts/fnr.py --type_param ortho \
  --batch_size 10 --config M1 --klip --img_size 256 --epsilon 0.1 --fnr_threshold 0.95
