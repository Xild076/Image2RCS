import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str]) -> float:
    print("$ " + " ".join(cmd))
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="Benchmark training and inference runtime with profiling enabled")
    parser.add_argument("--train", action="store_true", help="Run train benchmark")
    parser.add_argument("--infer", action="store_true", help="Run inference benchmark")
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=str, default="auto")
    parser.add_argument("--cache-mode", type=str, default="memory", choices=["off", "memory", "disk"])
    parser.add_argument("--cache-dir", type=str, default=".cache/image2rcs")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--input", type=str, default="data/images/f-22_images")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_train = args.train or (not args.train and not args.infer)
    run_infer = args.infer or (not args.train and not args.infer)

    if run_train:
        train_cmd = [
            sys.executable,
            "src/train.py",
            "--epochs",
            str(args.train_epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            args.num_workers,
            "--cache-mode",
            args.cache_mode,
            "--cache-dir",
            args.cache_dir,
            "--device",
            args.device,
            "--eval-every",
            "1",
            "--profile",
        ]
        if args.cpu_threads > 0:
            train_cmd.extend(["--cpu-threads", str(args.cpu_threads)])
        train_elapsed = run_command(train_cmd)
        print(f"[benchmark] train elapsed={train_elapsed:.3f}s")

    if run_infer:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        infer_cmd = [
            sys.executable,
            "src/inference.py",
            "--checkpoint",
            str(checkpoint_path),
            "--input",
            args.input,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            args.num_workers,
            "--cache-mode",
            args.cache_mode,
            "--cache-dir",
            args.cache_dir,
            "--device",
            args.device,
            "--profile",
        ]
        if args.cpu_threads > 0:
            infer_cmd.extend(["--cpu-threads", str(args.cpu_threads)])
        infer_elapsed = run_command(infer_cmd)
        print(f"[benchmark] inference elapsed={infer_elapsed:.3f}s")


if __name__ == "__main__":
    main()
