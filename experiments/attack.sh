# Attack a LipNet model
CUDA_VISIBLE_DEVICES=1 python scripts/attack.py --dataset iiit_pets \
	--data_root $TORCH_DATASETS --img_size 128 --config M1 --klip \
	--batch_size 100 --epsilon 1.5 --certificate q1 \
	--num_batches 5 --num_steps 250 --type_param ortho
CUDA_VISIBLE_DEVICES=1 python scripts/attack.py --dataset iiit_pets \
	--data_root $TORCH_DATASETS --img_size 128 --config M1 --klip \
	--batch_size 100 --epsilon 1.5 \
	--num_batches 5 --num_steps 250 --type_param ortho \
	--certificate q2 --adv_threshold 0.95

# Attack an AllNet model
CUDA_VISIBLE_DEVICES=1 python scripts/attack.py --dataset iiit_pets \
	--data_root $TORCH_DATASETS --img_size 128 --config M1 \
	--batch_size 100 --epsilon 1.5 --num_batches 5 --certificate q1 \
	--type_param unconstrained --num_steps 250 --klip
CUDA_VISIBLE_DEVICES=1 python scripts/attack.py --dataset iiit_pets \
	--data_root $TORCH_DATASETS --img_size 128 --config M1 --klip \
	--batch_size 100 --epsilon 1.5 \
	--num_batches 5 --num_steps 250 --type_param unconstrained \
	--certificate q2 --adv_threshold 0.95
