#!/usr/bin/env sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/datasets}"

if [ ! -d "$DATA_PATH" ]; then
    echo "DATA_PATH does not exist, creating: $DATA_PATH"
    mkdir -p "$DATA_PATH"
fi

echo "Using DATA_PATH=$DATA_PATH"

for seed in 42
do
python main.py \
    cifar100_lear \
    --dataset seq-cifar100 \
    --model_name LEAR \
    --backbone lear \
    --batch-size 32 \
    --epochs 5 \
    --data-path "$DATA_PATH" \
    --base_path "$DATA_PATH" \
    --lr 0.03 \
    --num_workers 4 \
    --seed "$seed" \
    --output_dir "./output/cifar100_lear_seed$seed"
done
