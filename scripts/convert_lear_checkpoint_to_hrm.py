#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from timm.models import create_model

# Register HRM/TII ViT model variants in timm's registry.
import vits.hide_prompt_vision_transformer  # noqa: F401


def _extract_state_dict(checkpoint_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], dict):
        return checkpoint_obj["model"]
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    raise ValueError("Unsupported checkpoint format: expected a dict or dict with 'model' key.")


def _candidate_keys(source_key: str):
    # Prefer exact key first, then progressively strip prefixes.
    candidates = [source_key]
    parts = source_key.split(".")
    for idx in range(1, len(parts)):
        candidates.append(".".join(parts[idx:]))

    # Common wrappers in LEAR/Mammoth and distributed checkpoints.
    normalized = []
    for key in candidates:
        for prefix in ("module.", "net.", "model.", "backbone.", "wrappee."):
            if key.startswith(prefix):
                normalized.append(key[len(prefix):])
        normalized.append(key)

    seen = set()
    for key in normalized:
        if key not in seen:
            seen.add(key)
            yield key


def _select_transfer_weights(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], int]:
    mapped: Dict[str, torch.Tensor] = {}
    used_source = 0

    for source_key, source_value in source_state.items():
        if not isinstance(source_value, torch.Tensor):
            continue

        matched = False
        for cand in _candidate_keys(source_key):
            if cand in target_state and target_state[cand].shape == source_value.shape:
                if cand not in mapped:
                    mapped[cand] = source_value
                    used_source += 1
                matched = True
                break

        if matched:
            continue

    return mapped, used_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a LEAR checkpoint to HRM/TII per-task checkpoint files.",
    )
    parser.add_argument("--lear-checkpoint", required=True, type=str, help="Path to LEAR checkpoint (.pt)")
    parser.add_argument("--output-dir", required=True, type=str, help="Output root containing checkpoint/task*_checkpoint.pth")
    parser.add_argument("--model", default="vit_base_patch16_224", type=str, help="HRM model name")
    parser.add_argument("--num-classes", default=200, type=int, help="Number of classes")
    parser.add_argument("--num-tasks", default=10, type=int, help="Number of tasks to emit")
    parser.add_argument("--drop", default=0.0, type=float)
    parser.add_argument("--drop-path", default=0.0, type=float)
    parser.add_argument("--pretrained", action="store_true", help="Initialize target model with pretrained timm weights")
    parser.add_argument(
        "--original-model-mlp-structure",
        type=int,
        nargs="*",
        default=[2],
        help="MLP structure expected by HRM TII model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    learner_ckpt = Path(args.lear_checkpoint).expanduser().resolve()
    if not learner_ckpt.exists():
        raise FileNotFoundError(f"LEAR checkpoint not found: {learner_ckpt}")

    target_root = Path(args.output_dir).expanduser().resolve()
    ckpt_dir = target_root / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LEAR checkpoint: {learner_ckpt}")
    loaded = torch.load(str(learner_ckpt), map_location="cpu", weights_only=False)
    source_state = _extract_state_dict(loaded)

    print(f"Building HRM model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        mlp_structure=args.original_model_mlp_structure,
    )

    target_state = model.state_dict()
    mapped, used_source = _select_transfer_weights(source_state, target_state)
    missing_before = len(target_state) - len(mapped)

    load_result = model.load_state_dict(mapped, strict=False)

    print(f"Mapped tensors: {len(mapped)}")
    print(f"Used source tensors: {used_source}")
    print(f"Target tensors not filled from LEAR: {missing_before}")
    print(f"Missing keys after load: {len(load_result.missing_keys)}")
    print(f"Unexpected keys after load: {len(load_result.unexpected_keys)}")

    payload = {
        "model": model.state_dict(),
        "optimizer": {},
        "epoch": 0,
        "args": {
            "converted_from": str(learner_ckpt),
            "converter": "scripts/convert_lear_checkpoint_to_hrm.py",
        },
    }

    for task_idx in range(1, args.num_tasks + 1):
        out_file = ckpt_dir / f"task{task_idx}_checkpoint.pth"
        torch.save(payload, str(out_file))

    print(f"Wrote {args.num_tasks} HRM checkpoint files under: {ckpt_dir}")


if __name__ == "__main__":
    main()
