#!/usr/bin/env sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/datasets}"
IMAGENET_R_URL="https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
MODE="${MODE:-train}"
SEEDS="${SEEDS:-42}"

case "$MODE" in
    train|eval)
        ;;
    *)
        echo "Unsupported MODE=$MODE. Use one of: train, eval"
        exit 1
        ;;
esac

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
echo "Running MODE=$MODE"

for seed in $SEEDS
do
if [ "$MODE" = "train" ]; then
python main.py \
    imr_lear \
    --dataset seq-imagenet-r \
    --model_name LEAR \
    --backbone lear \
    --batch-size 32 \
    --epochs 5 \
    --data-path "$DATA_PATH" \
    --base_path "$DATA_PATH" \
    --lr 0.03 \
    --num_workers 4 \
    --seed "$seed" \
    --output_dir "./output/imr_lear_seed$seed"
fi

if [ "$MODE" = "eval" ]; then
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./checkpoints/lear_seq-imagenet-r_default_0_5_last.pt}"
python main.py \
    imr_lear \
    --eval \
    --loadcheck "$CHECKPOINT_PATH" \
    --dataset seq-imagenet-r \
    --model_name LEAR \
    --backbone lear \
    --batch-size 32 \
    --epochs 5 \
    --data-path "$DATA_PATH" \
    --base_path "$DATA_PATH" \
    --lr 0.03 \
    --num_workers 4 \
    --seed "$seed" \
    --output_dir "./output/imr_lear_seed$seed"
fi
done
