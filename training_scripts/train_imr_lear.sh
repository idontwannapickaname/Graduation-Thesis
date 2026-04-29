#!/usr/bin/env sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/datasets}"
IMAGENET_R_URL="https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
MODE="${MODE:-train}"
SEEDS="${SEEDS:-42}"

# LEAR -> HRM bridge settings
LEAR_CHECKPOINT_PATH="${LEAR_CHECKPOINT_PATH:-./lear/checkpoints/lear_seq-imagenet-r_default_0_5_last.pt}"
HRM_OUTPUT_DIR="${HRM_OUTPUT_DIR:-./output/imr_lear_to_hrm}"
HRM_CONFIG="${HRM_CONFIG:-imr_hideprompt_5e}"
HRM_MODEL="${HRM_MODEL:-vit_base_patch16_224}"
HRM_NUM_TASKS="${HRM_NUM_TASKS:-10}"
HRM_NUM_CLASSES="${HRM_NUM_CLASSES:-200}"

case "$MODE" in
    train|eval|hrm_eval|train_then_hrm_eval)
        ;;
    *)
        echo "Unsupported MODE=$MODE. Use one of: train, eval, hrm_eval, train_then_hrm_eval"
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

run_lear_train() {
    seed="$1"
    ckpt_prefix="imr_lear_bridge_seed${seed}"
    python main.py \
        imr_lear \
        --dataset seq-imagenet-r \
        --model_name LEAR \
        --backbone lear \
        --lear_savecheck last \
        --lear_ckpt_name "$ckpt_prefix" \
        --batch-size 32 \
        --epochs 5 \
        --data-path "$DATA_PATH" \
        --base_path "$DATA_PATH" \
        --lr 0.03 \
        --num_workers 4 \
        --seed "$seed" \
        --output_dir "./output/imr_lear_seed$seed"
}

run_lear_eval() {
    seed="$1"
    checkpoint_path="$2"
    python main.py \
        imr_lear \
        --lear_inference_only \
        --lear_loadcheck "$checkpoint_path" \
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
}

run_hrm_eval_from_lear() {
    seed="$1"
    checkpoint_path="$2"
    hrm_seed_output_dir="${HRM_OUTPUT_DIR}_seed${seed}"

    if [ ! -f "$checkpoint_path" ]; then
        candidate_file="./output/imr_lear_seed${seed}/lear_checkpoint_path.txt"
        if [ -f "$candidate_file" ]; then
            checkpoint_path="$(cat "$candidate_file")"
            echo "Resolved LEAR checkpoint from $candidate_file"
        fi
    fi

    if [ ! -f "$checkpoint_path" ]; then
        echo "Error: LEAR checkpoint not found: $checkpoint_path"
        echo "Set LEAR_CHECKPOINT_PATH to a valid .pt file or run MODE=train first."
        exit 1
    fi

    python scripts/convert_lear_checkpoint_to_hrm.py \
        --lear-checkpoint "$checkpoint_path" \
        --output-dir "$hrm_seed_output_dir" \
        --model "$HRM_MODEL" \
        --num-classes "$HRM_NUM_CLASSES" \
        --num-tasks "$HRM_NUM_TASKS" \
        --original-model-mlp-structure 2

    python main.py \
        "$HRM_CONFIG" \
        --eval \
        --train_inference_task_only \
        --model "$HRM_MODEL" \
        --original_model "$HRM_MODEL" \
        --dataset Split-Imagenet-R \
        --batch-size 32 \
        --epochs 5 \
        --data-path "$DATA_PATH" \
        --lr 0.03 \
        --num_workers 4 \
        --seed "$seed" \
        --output_dir "$hrm_seed_output_dir"
}

for seed in $SEEDS
do
if [ "$MODE" = "train" ]; then
    run_lear_train "$seed"
fi

if [ "$MODE" = "eval" ]; then
    run_lear_eval "$seed" "$LEAR_CHECKPOINT_PATH"
fi

if [ "$MODE" = "hrm_eval" ]; then
    run_hrm_eval_from_lear "$seed" "$LEAR_CHECKPOINT_PATH"
fi

if [ "$MODE" = "train_then_hrm_eval" ]; then
    run_lear_train "$seed"
    run_hrm_eval_from_lear "$seed" "$LEAR_CHECKPOINT_PATH"
fi
done
