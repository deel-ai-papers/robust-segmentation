# Compare with smoothing certificate
for EPSILON in 0.001 0.01 0.05 0.1 0.3;
do
 for CLASS in {0..18};
 do
     CUDA_VISIBLE_DEVICES=1 python scripts/lipcertify.py \
       --dataset cityscapes --data_root $CITYSCAPES --img_size 1024 \
       --square_imgs --config M1 --type_param ortho --epsilon $EPSILON \
       --batch_size 10 --klip --class_wc_iou $CLASS \
       --output_csv data/lipcertify_results.csv
 done
done
