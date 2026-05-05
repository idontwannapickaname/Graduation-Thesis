#!/usr/bin/env sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/datasets}"
IMAGENET_R_URL="https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
MODE="${MODE:-train}"
SEEDS="${SEEDS:-42}"

# LEAR -> HRM bridge settings
LEAR_CHECKPOINT_PATH="${LEAR_CHECKPOINT_PATH:-}"
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

resolve_lear_checkpoint() {
    seed="$1"
    requested_path="$2"

    if [ -n "$requested_path" ] && [ -f "$requested_path" ]; then
        printf '%s\n' "$requested_path"
        return 0
    fi

    candidate_file="./output/imr_lear_seed${seed}/lear_checkpoint_path.txt"
    if [ -f "$candidate_file" ]; then
        resolved_path="$(head -n 1 "$candidate_file")"
        if [ -f "$resolved_path" ]; then
            printf '%s\n' "$resolved_path"
            return 0
        fi
    fi

    latest_candidate="$(find ./lear/checkpoints -maxdepth 1 -type f -name '*.pt' | sort | tail -n 1 || true)"
    if [ -n "$latest_candidate" ] && [ -f "$latest_candidate" ]; then
        printf '%s\n' "$latest_candidate"
        return 0
    fi

    return 1
}

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

    mkdir -p "$hrm_seed_output_dir"

    resolved_checkpoint="$(resolve_lear_checkpoint "$seed" "$checkpoint_path" || true)"
    if [ -z "$resolved_checkpoint" ] || [ ! -f "$resolved_checkpoint" ]; then
        echo "Error: LEAR checkpoint not found: $checkpoint_path"
        echo "Set LEAR_CHECKPOINT_PATH to a valid .pt file or run MODE=train first so the script can record lear_checkpoint_path.txt."
        exit 1
    fi

    echo "Using LEAR checkpoint: $resolved_checkpoint"

    learner_state_dict_path="$hrm_seed_output_dir/lear_state_dict.pt"
    python - "$resolved_checkpoint" "$learner_state_dict_path" <<'PY'
import sys
from pathlib import Path
import torch

source_path = Path(sys.argv[1]).expanduser().resolve()
target_path = Path(sys.argv[2]).expanduser().resolve()

checkpoint = torch.load(str(source_path), map_location='cpu', weights_only=False)
state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

torch.save({'model': state_dict}, str(target_path))
print(f"Saved LEAR state dict to: {target_path}")
PY

    python scripts/convert_lear_checkpoint_to_hrm.py \
        --lear-checkpoint "$learner_state_dict_path" \
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
