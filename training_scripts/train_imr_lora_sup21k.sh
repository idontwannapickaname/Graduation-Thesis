#!/usr/bin/env sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/datasets}"
IMAGENET_R_URL="https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"

if [ ! -d "$DATA_PATH" ]; then
        echo "DATA_PATH does not exist, creating: $DATA_PATH"
        mkdir -p "$DATA_PATH"
fi

if [ ! -d "$DATA_PATH/imagenet-r" ]; then
        ARCHIVE_PATH="$DATA_PATH/imagenet-r.tar"

        if [ ! -f "$ARCHIVE_PATH" ]; then
                echo "imagenet-r not found. Downloading dataset archive to $ARCHIVE_PATH"
                if command -v curl >/dev/null 2>&1; then
                        curl -L "$IMAGENET_R_URL" -o "$ARCHIVE_PATH"
                elif command -v wget >/dev/null 2>&1; then
                        wget -O "$ARCHIVE_PATH" "$IMAGENET_R_URL"
                else
                        echo "Error: neither curl nor wget is installed; cannot download ImageNet-R."
                        exit 1
                fi
        fi

        echo "Extracting $ARCHIVE_PATH to $DATA_PATH"
        tar -xf "$ARCHIVE_PATH" -C "$DATA_PATH"
fi

if [ ! -d "$DATA_PATH/imagenet-r" ]; then
        echo "Error: ImageNet-R folder was not found after setup: $DATA_PATH/imagenet-r"
        exit 1
fi

echo "Using DATA_PATH=$DATA_PATH"

for seed in 42
do
torchrun --nproc_per_node=1 \
        --master_port='29500' \
        --use_env main.py \
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
torchrun --nproc_per_node=2 \
        --master_port='29513' \
        --use_env main.py \
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
