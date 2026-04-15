#!/usr/bin/env sh

set -eu
if (set -o pipefail) 2>/dev/null; then
        set -o pipefail
fi

DATA_PATH="${DATA_PATH:-./datasets}"

# Accept common layouts:
# 1) DATA_PATH contains imagenet-r/ (already extracted)
# 2) DATA_PATH points directly to imagenet-r/
# 3) DATA_PATH contains imagenet-r.tar or imagenet-r.tar.gz
if [ -d "$DATA_PATH" ] && [ "$(basename "$DATA_PATH")" = "imagenet-r" ]; then
        DATA_PATH="$(dirname "$DATA_PATH")"
fi

if [ ! -d "$DATA_PATH/imagenet-r" ]; then
        if [ -f "$DATA_PATH/imagenet-r.tar.gz" ]; then
                echo "Found $DATA_PATH/imagenet-r.tar.gz, extracting..."
                tar -xzf "$DATA_PATH/imagenet-r.tar.gz" -C "$DATA_PATH"
        elif [ -f "$DATA_PATH/imagenet-r.tar" ]; then
                echo "Found $DATA_PATH/imagenet-r.tar"
                # continual_datasets/continual_datasets.py handles extraction from this tar.
        else
                echo "Error: could not find ImageNet-R data under DATA_PATH=$DATA_PATH"
                echo "Expected one of:"
                echo "  1) $DATA_PATH/imagenet-r/"
                echo "  2) $DATA_PATH/imagenet-r.tar"
                echo "  3) $DATA_PATH/imagenet-r.tar.gz"
                echo "You can also set DATA_PATH directly to the imagenet-r folder."
                echo "Example: DATA_PATH=/kaggle/input/imagenet-r bash training_scripts/train_imr_lora_sup21k.sh"
                exit 1
        fi
fi

echo "Using DATA_PATH=$DATA_PATH"

for seed in 42
do
torchrun \
        --nproc_per_node=1 \
        --master_port='29500' \
        main.py \
        imr_hideprompt_5e \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 128 \
        --ca_storage_efficient_method covariance \
        --epochs 20 \
        --data-path "$DATA_PATH" \
        --lr 0.0005 \
        --ca_lr 0.005 \
        --crct_epochs 30 \
        --seed $seed \
        --train_inference_task_only \
        --output_dir ./output/imr_vit_multi_centroid_mlp_2_seed$seed
done


for seed in 42
do
torchrun \
        --nproc_per_node=2 \
        --master_port='29513' \
        main.py \
        imr_lora \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 24 \
        --epochs 50 \
        --data-path "$DATA_PATH" \
        --ca_lr 0.005 \
        --crct_epochs 30 \
        --seed 42 \
        --lr 0.03 \
        --con 0.2 \
        --lora_rank 5 \
        --En gen \
        --tau -10 \
        --K 5 \
        --sched cosine \
        --dataset Split-Imagenet-R \
        --lora_momentum 0.4 \
        --lora_type hide \
        --trained_original_model ./output/imr_vit_multi_centroid_mlp_2_seed$seed \
        --output_dir ./output/test_imr_sup21k_lora_pe_seed42 
done
