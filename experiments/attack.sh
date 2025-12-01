# Attack a LipNet model
CUDA_VISIBLE_DEVICES=0 python scripts/attack.py --dataset iiit_pets \
	--data_root $TORCH_DATASETS --img_size 128 --config M1 --klip \
	--batch_size 128 --epsilon 3.0 \
	--num_batches 5 --num_steps 500 --type_param ortho

# Attack an AllNet model
CUDA_VISIBLE_DEVICES=0 python scripts/attack.py --dataset iiit_pets \
	--data_root $TORCH_DATASETS --img_size 128 --config M1 \
	--batch_size 128 --epsilon 3.0 \
	--type_param unconstrained --num_steps 500

