import os
import shlex
import subprocess
import sys
from pathlib import Path


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

    if args.device:
        run_cmd.extend(["--device", args.device])

    env = os.environ.copy()
    env.setdefault("MAMMOTH_TEST", "0")

    print("Launching LEAR training command:")
    print(" ".join(shlex.quote(part) for part in run_cmd))
    print(f"Working directory: {lear_root}")
    print(f"Output directory: {output_dir}")

    subprocess.run(run_cmd, cwd=str(lear_root), env=env, check=True)
