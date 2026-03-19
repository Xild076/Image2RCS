import argparse
from collections import defaultdict
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import AircraftRCSDataset, build_image_transform, inverse_transform_target_tensor, load_image_samples
from model import build_model
from perf_utils import autotune_num_workers, configure_cpu_threads, format_worker_timings, parse_num_workers
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_samples(samples, val_split: float, seed: int):
    rng = np.random.default_rng(seed)
    grouped_samples: dict[str, list[tuple[Path, float]]] = defaultdict(list)
    for image_path, target in samples:
        category = image_path.parent.name
        grouped_samples[category].append((image_path, target))

    train_samples: list[tuple[Path, float]] = []
    val_samples: list[tuple[Path, float]] = []

    for group in grouped_samples.values():
        group_size = len(group)
        if group_size <= 1:
            train_samples.extend(group)
            continue

        indices = np.arange(group_size)
        rng.shuffle(indices)
        val_size = int(round(group_size * val_split))
        val_size = min(max(1, val_size), group_size - 1)

        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        val_samples.extend(group[i] for i in val_idx)
        train_samples.extend(group[i] for i in train_idx)

    if not train_samples or not val_samples:
        raise ValueError("Stratified split failed to produce both train and validation samples")

    train_order = rng.permutation(len(train_samples))
    val_order = rng.permutation(len(val_samples))
    train_samples = [train_samples[i] for i in train_order]
    val_samples = [val_samples[i] for i in val_order]
    return train_samples, val_samples


def create_dataloaders(
    csv_path: str,
    images_root: str,
    image_size: int,
    batch_size: int,
    num_workers: str | int,
    val_split: float,
    seed: int,
    target_mode: str,
    device: torch.device,
    cache_mode: str,
    cache_dir: str,
    profile: bool,
):
    samples = load_image_samples(csv_path=csv_path, images_root=images_root)
    train_samples, val_samples = split_samples(samples, val_split=val_split, seed=seed)

    train_dataset = AircraftRCSDataset(
        samples=train_samples,
        image_transform=build_image_transform(image_size=image_size, train=True),
        target_mode=target_mode,
        cache_mode=cache_mode,
        cache_dir=cache_dir,
    )
    val_dataset = AircraftRCSDataset(
        samples=val_samples,
        image_transform=build_image_transform(image_size=image_size, train=False),
        target_mode=target_mode,
        cache_mode=cache_mode,
        cache_dir=cache_dir,
    )

    pin_memory = device.type == "cuda"
    parsed_num_workers = parse_num_workers(num_workers)
    worker_timings: dict[int, float] = {}
    if parsed_num_workers == "auto":
        resolved_num_workers, worker_timings = autotune_num_workers(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
        )
    else:
        resolved_num_workers = int(parsed_num_workers)

    persistent_workers = resolved_num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=resolved_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=resolved_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    if profile:
        print(
            f"[profile] num_workers={resolved_num_workers} "
            f"(auto timings: {format_worker_timings(worker_timings)})"
        )

    return train_loader, val_loader

class RelativeHuberLoss(nn.Module):
    def __init__(self, target_mode: str, relative_floor: float = 0.05, beta: float = 0.2):
        super().__init__()
        self.target_mode = target_mode
        self.relative_floor = relative_floor
        self.beta = beta

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_raw = inverse_transform_target_tensor(prediction, mode=self.target_mode)
        target_raw = inverse_transform_target_tensor(target, mode=self.target_mode)

        pred_raw = torch.clamp(pred_raw, min=0.0)
        denom = torch.clamp(target_raw.abs(), min=self.relative_floor)
        rel_error = (pred_raw - target_raw) / denom
        abs_rel = rel_error.abs()

        loss = torch.where(
            abs_rel < self.beta,
            0.5 * (rel_error ** 2) / self.beta,
            abs_rel - 0.5 * self.beta,
        )
        return loss.mean()


class HybridRCSLoss(nn.Module):
    def __init__(
        self,
        target_mode: str,
        alpha: float = 0.7,
        smoothl1_beta: float = 0.25,
        relative_floor: float = 0.05,
        relative_beta: float = 0.2,
    ):
        super().__init__()
        self.alpha = alpha
        self.abs_loss = nn.SmoothL1Loss(beta=smoothl1_beta)
        self.rel_loss = RelativeHuberLoss(
            target_mode=target_mode,
            relative_floor=relative_floor,
            beta=relative_beta,
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        abs_component = self.abs_loss(prediction, target)
        rel_component = self.rel_loss(prediction, target)
        return self.alpha * abs_component + (1.0 - self.alpha) * rel_component

def run_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    total_count = 0
    data_load_time = 0.0
    forward_backward_time = 0.0
    wait_start = time.perf_counter()

    for images, targets in loader:
        data_load_time += time.perf_counter() - wait_start
        step_start = time.perf_counter()

        non_blocking = device.type == "cuda"
        images = images.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            predictions = model(images)
            loss = criterion(predictions, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        forward_backward_time += time.perf_counter() - step_start
        wait_start = time.perf_counter()

    stats = {
        "data_load": data_load_time,
        "forward_backward": forward_backward_time,
    }
    return total_loss / max(total_count, 1), stats


@torch.inference_mode()
def evaluate(model, loader, criterion, device, target_mode: str):
    model.eval()
    total_loss = 0.0
    total_count = 0
    abs_error_sum = 0.0
    data_load_time = 0.0
    eval_compute_time = 0.0
    wait_start = time.perf_counter()

    for images, targets in loader:
        data_load_time += time.perf_counter() - wait_start
        step_start = time.perf_counter()

        non_blocking = device.type == "cuda"
        images = images.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        predictions = model(images)
        loss = criterion(predictions, targets)

        pred_values = predictions.squeeze(1)
        target_values = targets.squeeze(1)
        pred_rcs = inverse_transform_target_tensor(pred_values, mode=target_mode)
        true_rcs = inverse_transform_target_tensor(target_values, mode=target_mode)
        abs_error_sum += torch.abs(pred_rcs - true_rcs).sum().item()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        eval_compute_time += time.perf_counter() - step_start
        wait_start = time.perf_counter()

    avg_loss = total_loss / max(total_count, 1)
    avg_mae = abs_error_sum / max(total_count, 1)
    stats = {
        "data_load": data_load_time,
        "eval_compute": eval_compute_time,
    }
    return avg_loss, avg_mae, stats


def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int, best_val_loss: float, image_size: int, target_mode: str):
    checkpoint = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "image_size": image_size,
        "target_mode": target_mode,
    }
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="Train image-to-RCS regression model")
    parser.add_argument("--csv-path", type=str, default="data/aircraft_rcs.csv")
    parser.add_argument("--images-root", type=str, default="data/images")
    parser.add_argument("--output", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=str, default="auto", help='DataLoader workers, integer or "auto"')
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", type=str, default="log1p", choices=["log1p", "none"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="disk", choices=["off", "memory", "disk"])
    parser.add_argument("--cache-dir", type=str, default=".cache/image2rcs")
    parser.add_argument("--cpu-threads", type=int, default=0, help="Set torch CPU threads; 0 keeps runtime default")
    parser.add_argument("--profile", action="store_true", help="Print timing breakdown per epoch")
    parser.add_argument("--eval-every", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--loss", type=str, default="smoothl1", choices=["hybrid", "relative-huber", "smoothl1"])
    parser.add_argument("--loss-alpha", type=float, default=0.7)
    parser.add_argument("--relative-floor", type=float, default=0.05)
    parser.add_argument("--relative-beta", type=float, default=0.2)
    parser.add_argument("--smoothl1-beta", type=float, default=0.25)
    args = parser.parse_args()

    set_seed(args.seed)
    configure_cpu_threads(args.cpu_threads if args.cpu_threads > 0 else None)
    if args.eval_every < 1:
        raise ValueError("--eval-every must be >= 1")
    if not (0.0 < args.val_split < 1.0):
        raise ValueError("--val-split must be between 0 and 1")
    if not (0.0 <= args.loss_alpha <= 1.0):
        raise ValueError("--loss-alpha must be between 0 and 1")
    if args.relative_floor <= 0:
        raise ValueError("--relative-floor must be > 0")
    if args.relative_beta <= 0:
        raise ValueError("--relative-beta must be > 0")
    if args.smoothl1_beta <= 0:
        raise ValueError("--smoothl1-beta must be > 0")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train_loader, val_loader = create_dataloaders(
        csv_path=args.csv_path,
        images_root=args.images_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        target_mode=args.target_mode,
        device=device,
        cache_mode=args.cache_mode,
        cache_dir=args.cache_dir,
        profile=args.profile,
    )

    model = build_model(pretrained=not args.no_pretrained, out_dim=1).to(device)
    if args.loss == "hybrid":
        criterion = HybridRCSLoss(
            target_mode=args.target_mode,
            alpha=args.loss_alpha,
            smoothl1_beta=args.smoothl1_beta,
            relative_floor=args.relative_floor,
            relative_beta=args.relative_beta,
        )
    elif args.loss == "relative-huber":
        criterion = RelativeHuberLoss(
            target_mode=args.target_mode,
            relative_floor=args.relative_floor,
            beta=args.relative_beta,
        )
    else:
        criterion = nn.SmoothL1Loss(beta=args.smoothl1_beta)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = GradScaler(device=device.type, enabled=device.type == "cuda")

    best_val_loss = float("inf")
    profile_rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_stats = run_epoch(model, train_loader, criterion, optimizer, device, scaler)

        val_loss = None
        val_mae = None
        eval_total_time = 0.0
        eval_stats = {"data_load": 0.0, "eval_compute": 0.0}
        checkpoint_io_time = 0.0

        should_evaluate = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if should_evaluate:
            eval_start = time.perf_counter()
            val_loss, val_mae, eval_stats = evaluate(model, val_loader, criterion, device, args.target_mode)
            eval_total_time = time.perf_counter() - eval_start
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_start = time.perf_counter()
                save_checkpoint(
                    path=str(output_path),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    image_size=args.image_size,
                    target_mode=args.target_mode,
                )
                checkpoint_io_time = time.perf_counter() - checkpoint_start

        epoch_time = time.perf_counter() - epoch_start
        val_loss_text = f"{val_loss:.6f}" if val_loss is not None else "skipped"
        val_mae_text = f"{val_mae:.6f}" if val_mae is not None else "skipped"
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | train_loss={train_loss:.6f} "
            f"| val_loss={val_loss_text} | val_mae={val_mae_text}"
        )
        if args.profile:
            print(
                f"[profile] data_load={train_stats['data_load']:.3f}s "
                f"| fwd_bwd={train_stats['forward_backward']:.3f}s "
                f"| eval={eval_total_time:.3f}s "
                f"| ckpt_io={checkpoint_io_time:.3f}s "
                f"| epoch={epoch_time:.3f}s"
            )
            profile_rows.append(
                {
                    "data_load": train_stats["data_load"],
                    "forward_backward": train_stats["forward_backward"],
                    "eval": eval_total_time,
                    "eval_data_load": eval_stats["data_load"],
                    "eval_compute": eval_stats["eval_compute"],
                    "checkpoint_io": checkpoint_io_time,
                    "epoch": epoch_time,
                }
            )

    if args.profile and profile_rows:
        row_count = float(len(profile_rows))
        avg_data = sum(r["data_load"] for r in profile_rows) / row_count
        avg_fwd_bwd = sum(r["forward_backward"] for r in profile_rows) / row_count
        avg_eval = sum(r["eval"] for r in profile_rows) / row_count
        avg_eval_data = sum(r["eval_data_load"] for r in profile_rows) / row_count
        avg_eval_compute = sum(r["eval_compute"] for r in profile_rows) / row_count
        avg_ckpt = sum(r["checkpoint_io"] for r in profile_rows) / row_count
        avg_epoch = sum(r["epoch"] for r in profile_rows) / row_count
        print(
            "[profile][avg] "
            f"data_load={avg_data:.3f}s | fwd_bwd={avg_fwd_bwd:.3f}s | eval={avg_eval:.3f}s "
            f"(eval_data={avg_eval_data:.3f}s, eval_compute={avg_eval_compute:.3f}s) "
            f"| ckpt_io={avg_ckpt:.3f}s | epoch={avg_epoch:.3f}s"
        )

    if best_val_loss == float("inf"):
        print("No validation was run; no checkpoint saved.")
    else:
        print(f"Saved best checkpoint to {output_path} with val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()
