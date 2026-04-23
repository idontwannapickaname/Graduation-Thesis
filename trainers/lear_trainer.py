import os
import shlex
import subprocess
import sys
from pathlib import Path
import glob


def _normalize_lear_device(device_arg: str | None) -> str | None:
    if not device_arg:
        return None

    value = device_arg.strip().lower()
    if value in {"cpu", "mps", "auto"}:
        return None
    if value == "cuda":
        return "0"
    if value.startswith("cuda:"):
        return value.split(":", 1)[1]
    if "," in value:
        parts = [p.strip() for p in value.split(",")]
        normalized = []
        for part in parts:
            if part.startswith("cuda:"):
                normalized.append(part.split(":", 1)[1])
            else:
                normalized.append(part)
        return ",".join(normalized)
    return value


def train(args):
    repo_root = Path(__file__).resolve().parents[1]
    lear_root = repo_root / "lear"
    lear_main = lear_root / "main_domain.py"

    if not lear_main.exists():
        raise FileNotFoundError(f"LEAR entrypoint not found: {lear_main}")

    base_path_arg = args.base_path
    if base_path_arg == "./data/" and getattr(args, "data_path", None):
        base_path_arg = args.data_path

    base_path = Path(base_path_arg).expanduser()
    if not base_path.is_absolute():
        base_path = (repo_root / base_path).resolve()

    # Backward-compatibility shim for ImageNet-R configs/loaders that still
    # reference paths like "lear/data/imagenet-r/...".
    if args.dataset == "seq-imagenet-r":
        source_imr = base_path / "imagenet-r"
        lear_data_dir = lear_root / "data"
        target_imr = lear_data_dir / "imagenet-r"

        if source_imr.exists():
            lear_data_dir.mkdir(parents=True, exist_ok=True)
            if not target_imr.exists():
                try:
                    os.symlink(source_imr, target_imr, target_is_directory=True)
                    print(f"Created symlink: {target_imr} -> {source_imr}")
                except OSError:
                    # Symlink may be unsupported; mirror as fallback.
                    import shutil
                    shutil.copytree(source_imr, target_imr, dirs_exist_ok=True)
                    print(f"Copied ImageNet-R into compatibility path: {target_imr}")

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cmd = [
        sys.executable,
        str(lear_main),
        "--dataset",
        args.dataset,
        "--model",
        args.model_name,
        "--backbone",
        args.backbone,
        "--lr",
        str(args.lr),
        "--batch_size",
        str(args.batch_size),
        "--n_epochs",
        str(args.epochs),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--base_path",
        str(base_path),
    ]

    if getattr(args, "lear_savecheck", "none") != "none":
        run_cmd.extend(["--savecheck", str(args.lear_savecheck)])
    if getattr(args, "lear_ckpt_name", ""):
        run_cmd.extend(["--ckpt_name", str(args.lear_ckpt_name)])
    if getattr(args, "lear_loadcheck", ""):
        run_cmd.extend(["--loadcheck", str(args.lear_loadcheck)])
    if getattr(args, "lear_inference_only", False):
        run_cmd.extend(["--inference_only", "1"])

    lear_device = _normalize_lear_device(getattr(args, "device", None))
    if lear_device is not None:
        run_cmd.extend(["--device", lear_device])

    env = os.environ.copy()
    env.setdefault("MAMMOTH_TEST", "0")

    print("Launching LEAR training command:")
    print(" ".join(shlex.quote(part) for part in run_cmd))
    print(f"Working directory: {lear_root}")
    print(f"Output directory: {output_dir}")

    subprocess.run(run_cmd, cwd=str(lear_root), env=env, check=True)

    # Persist the resolved checkpoint path for downstream bridge scripts.
    if getattr(args, "lear_savecheck", "none") != "none":
        ckpt_dir = lear_root / "checkpoints"
        ckpt_prefix = str(getattr(args, "lear_ckpt_name", "")).strip()
        if ckpt_prefix:
            pattern = str(ckpt_dir / f"{ckpt_prefix}_*.pt")
        else:
            pattern = str(ckpt_dir / "*.pt")
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
        if candidates:
            latest_ckpt = Path(candidates[-1]).resolve()
            out_file = output_dir / "lear_checkpoint_path.txt"
            out_file.write_text(str(latest_ckpt) + "\n", encoding="utf-8")
            print(f"Recorded LEAR checkpoint path: {out_file} -> {latest_ckpt}")
