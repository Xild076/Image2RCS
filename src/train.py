import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
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
from tqdm import tqdm

from dataset import AircraftRCSDataset, build_image_transform, inverse_transform_target_tensor, load_image_samples
from device_utils import (
    configure_torch_for_device,
    cuda_probe_failure_summary,
    describe_device,
    resolve_device,
    usable_cuda_device_indices,
)
from model import SUPPORTED_MODEL_TYPES, build_model, normalize_model_type, resolve_model_config
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
    memory_cache_items: int,
    persistent_workers: bool,
    prefetch_factor: int,
    profile: bool,
    hdf5_path: str = "data/aircraft_dataset.h5",
):
    samples = load_image_samples(csv_path=csv_path, images_root=images_root, hdf5_path=hdf5_path if cache_mode == "hdf5" else None)
    train_samples, val_samples = split_samples(samples, val_split=val_split, seed=seed)

    train_dataset = AircraftRCSDataset(
        samples=train_samples,
        image_transform=build_image_transform(image_size=image_size, train=True),
        target_mode=target_mode,
        cache_mode=cache_mode,
        cache_dir=cache_dir,
        memory_cache_items=memory_cache_items,
        hdf5_path=hdf5_path,
    )
    val_dataset = AircraftRCSDataset(
        samples=val_samples,
        image_transform=build_image_transform(image_size=image_size, train=False),
        target_mode=target_mode,
        cache_mode=cache_mode,
        cache_dir=cache_dir,
        memory_cache_items=memory_cache_items,
        hdf5_path=hdf5_path,
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

    loader_common = {
        "num_workers": resolved_num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": bool(persistent_workers and resolved_num_workers > 0),
    }
    if resolved_num_workers > 0:
        loader_common["prefetch_factor"] = max(1, int(prefetch_factor))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_common,
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

def run_epoch(model, loader, criterion, optimizer, device, scaler, epoch=None, show_pbar=True):
    model.train()
    total_loss = 0.0
    total_count = 0
    data_load_time = 0.0
    forward_backward_time = 0.0
    wait_start = time.perf_counter()

    desc = f"Epoch {epoch} Train" if epoch is not None else "Training"
    pbar = tqdm(loader, desc=desc, leave=False, disable=not show_pbar)
    for images, targets in pbar:
        data_load_time += time.perf_counter() - wait_start
        step_start = time.perf_counter()

        non_blocking = device.type in ["cuda", "mps"]
        images = images.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        
        amp_enabled = device.type in ["cuda", "mps"]
        amp_device = device.type if amp_enabled else "cpu"
        with autocast(device_type=amp_device, enabled=amp_enabled):
            predictions = model(images)
            loss = criterion(predictions, targets)

        if hasattr(scaler, "scale") and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        
        pbar.set_postfix({"loss": f"{total_loss / total_count:.4f}"})
        
        forward_backward_time += time.perf_counter() - step_start
        wait_start = time.perf_counter()

    stats = {
        "data_load": data_load_time,
        "forward_backward": forward_backward_time,
    }
    return total_loss / max(total_count, 1), stats


@torch.inference_mode()
def evaluate(model, loader, criterion, device, target_mode: str, epoch=None, show_pbar=True):
    model.eval()
    total_loss = 0.0
    total_count = 0
    abs_error_sum = 0.0
    data_load_time = 0.0
    eval_compute_time = 0.0
    wait_start = time.perf_counter()

    desc = f"Epoch {epoch} Eval" if epoch is not None else "Evaluating"
    pbar = tqdm(loader, desc=desc, leave=False, disable=not show_pbar)
    for images, targets in pbar:
        data_load_time += time.perf_counter() - wait_start
        step_start = time.perf_counter()

        non_blocking = device.type in ["cuda", "mps"]
        images = images.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        amp_enabled = device.type in ["cuda", "mps"]
        amp_device = device.type if amp_enabled else "cpu"
        with autocast(device_type=amp_device, enabled=amp_enabled):
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_quality_record(
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_mae: float,
    best_val_loss: float,
    learning_rate: float,
) -> dict:
    return {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_mae_m2": float(val_mae),
        "best_val_loss": float(best_val_loss),
        "learning_rate": float(learning_rate),
        "updated_at": utc_now_iso(),
    }


def write_quality_log(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def parse_gpu_id_list(value: str, usable_ids: list[int]) -> list[int]:
    token = value.strip().lower()
    if token == "auto":
        return list(usable_ids)

    selected: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid --gpu-ids value: {value}") from exc
        selected.append(idx)

    if not selected:
        raise ValueError("--gpu-ids did not contain any GPU ids")
    invalid = [idx for idx in selected if idx not in usable_ids]
    if invalid:
        details = cuda_probe_failure_summary()
        raise ValueError(
            f"Requested GPU ids are not usable with this runtime: {invalid}. "
            f"Usable ids: {usable_ids}. Details: {details}"
        )
    return selected


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    image_size: int,
    target_mode: str,
    model_type: str,
    model_config: dict,
    quality_snapshot: dict | None = None,
    quality_history: list[dict] | None = None,
):
    checkpoint = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "image_size": image_size,
        "target_mode": target_mode,
        "model_type": model_type,
        "model_config": dict(model_config),
    }
    if quality_snapshot is not None:
        checkpoint["quality"] = quality_snapshot
    if quality_history is not None:
        checkpoint["quality_history"] = quality_history
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="Train image-to-RCS regression model")
    parser.add_argument("--csv-path", type=str, default="data/aircraft_rcs.csv")
    parser.add_argument("--images-root", type=str, default="data/images")
    parser.add_argument("--output", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=str, default="auto", help='DataLoader workers, integer or "auto"')
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--model-type", type=str, default="resnet18_se", choices=SUPPORTED_MODEL_TYPES)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--se-reduction", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", type=str, default="log1p", choices=["log1p", "none"])
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto|gpu|cuda|cuda:N|mps|cpu (auto prefers CUDA, then MPS, then CPU)",
    )
    parser.add_argument(
        "--multi-gpu",
        type=str,
        default="auto",
        choices=["auto", "off", "on"],
        help="Use DataParallel across multiple usable CUDA GPUs (auto enables when possible).",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="auto",
        help='Comma-separated CUDA ids for DataParallel, e.g. "0,1" (default: auto).',
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() for faster training (PyTorch 2.0+)")
    parser.add_argument("--cache-mode", type=str, default="hdf5", choices=["off", "memory", "disk", "hdf5"])
    parser.add_argument("--hdf5-path", type=str, default="data/aircraft_dataset.h5", help="Path to HDF5 database if cache-mode is hdf5")
    parser.add_argument("--cache-dir", type=str, default=".cache/image2rcs")
    parser.add_argument(
        "--memory-cache-items",
        type=int,
        default=256,
        help="Max decoded images kept in RAM when --cache-mode=memory.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs (higher speed, higher RAM).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=1,
        help="Prefetched batches per worker when num_workers > 0.",
    )
    parser.add_argument("--cpu-threads", type=int, default=0, help="Set torch CPU threads; 0 keeps runtime default")
    parser.add_argument("--profile", action="store_true", help="Print timing breakdown per epoch")
    parser.add_argument("--eval-every", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--loss", type=str, default="smoothl1", choices=["hybrid", "relative-huber", "smoothl1"])
    parser.add_argument("--loss-alpha", type=float, default=0.7)
    parser.add_argument("--relative-floor", type=float, default=0.05)
    parser.add_argument("--relative-beta", type=float, default=0.2)
    parser.add_argument("--smoothl1-beta", type=float, default=0.25)
    parser.add_argument("--no-pbar", action="store_true", help="Disable the per-epoch progress bars")
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
    if args.memory_cache_items < 1:
        raise ValueError("--memory-cache-items must be >= 1")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")
    if args.hidden_dim < 1:
        raise ValueError("--hidden-dim must be >= 1")
    if not (0.0 <= args.dropout < 1.0):
        raise ValueError("--dropout must be in [0, 1)")
    if args.se_reduction < 1:
        raise ValueError("--se-reduction must be >= 1")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    configure_torch_for_device(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    if device.type != "cuda" and torch.cuda.is_available():
        details = cuda_probe_failure_summary()
        if details:
            print(f"[warn] CUDA detected but unusable with this Torch build. Falling back to {device}. Details: {details}")
    print(f"Using device: {describe_device(device)}")
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
        memory_cache_items=args.memory_cache_items,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        profile=args.profile,
        hdf5_path=args.hdf5_path,
    )

    model_type = normalize_model_type(args.model_type)
    model_config = resolve_model_config(
        model_type=model_type,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        se_reduction=args.se_reduction,
    )
    print(f"Model type: {model_type} | config={model_config}")
    model = build_model(
        pretrained=not args.no_pretrained,
        out_dim=1,
        model_type=model_type,
        model_config=model_config,
    ).to(device)

    best_val_loss = float("inf")
    start_epoch = 1
    opt_state = None
    sched_state = None

    if hasattr(args, "resume") and args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        # Strip prefixes
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                stripped[k[10:]] = v
            elif k.startswith("module."):
                stripped[k[7:]] = v
            else:
                stripped[k] = v
        # Try strict first, then unstrict
        try:
            model.load_state_dict(stripped, strict=True)
        except RuntimeError as e:
            print(f"Warning: strict load failed, trying non-strict. {e}")
            model.load_state_dict(stripped, strict=False)

        opt_state = checkpoint.get("optimizer_state_dict")
        sched_state = checkpoint.get("scheduler_state_dict")
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))


    if device.type == "cuda":
        usable_ids = usable_cuda_device_indices()
        requested_ids = parse_gpu_id_list(args.gpu_ids, usable_ids)
        primary_idx = device.index if device.index is not None else requested_ids[0]
        if primary_idx not in requested_ids:
            requested_ids = [primary_idx] + [idx for idx in requested_ids if idx != primary_idx]
        torch.cuda.set_device(primary_idx)
        if args.multi_gpu == "on" and len(requested_ids) < 2:
            print(f"[warn] --multi-gpu=on requested, but only one usable CUDA device found: {requested_ids}")

        should_wrap_dp = (
            args.multi_gpu == "on"
            or (
                args.multi_gpu == "auto"
                and args.device.strip().lower() in {"auto", "gpu", "cuda"}
                and len(requested_ids) > 1
            )
        )
        if should_wrap_dp and len(requested_ids) > 1:
            model = nn.DataParallel(model, device_ids=requested_ids, output_device=requested_ids[0])
            print(f"Multi-GPU: DataParallel enabled on CUDA devices {requested_ids}")
        else:
            print(f"Single-GPU mode on cuda:{primary_idx} (usable CUDA devices: {usable_ids})")
            
    if args.compile:
        print("Compiling model (this may take a minute...)")
        model = torch.compile(model)

            


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

    optimizer_kwargs = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    if opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state)
        except Exception as e:
            print(f"Warning: Could not load optimizer: {e}")
    if sched_state is not None:
        try:
            scheduler.load_state_dict(sched_state)
        except Exception as e:
            print(f"Warning: Could not load scheduler: {e}")

    amp_enabled = device.type in ["cuda", "mps"]
    scaler_device = device.type if device.type in ["cuda", "mps"] else "cpu"
    try:
        from torch.amp import GradScaler
        scaler = GradScaler(scaler_device, enabled=amp_enabled)
    except:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(enabled=amp_enabled) if device.type == "cuda" else GradScaler(enabled=False)

    profile_rows: list[dict[str, float]] = []
    quality_history: list[dict] = []
    quality_log_path = output_path.with_name(f"{output_path.stem}_quality_log.json")

    show_pbar = not args.no_pbar
    for epoch in tqdm(range(start_epoch, args.epochs + 1)):
        epoch_start = time.perf_counter()
        train_loss, train_stats = run_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch=epoch, show_pbar=show_pbar)

        val_loss = None
        val_mae = None
        eval_total_time = 0.0
        eval_stats = {"data_load": 0.0, "eval_compute": 0.0}
        checkpoint_io_time = 0.0

        should_evaluate = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if should_evaluate:
            eval_start = time.perf_counter()
            val_loss, val_mae, eval_stats = evaluate(model, val_loader, criterion, device, args.target_mode, epoch=epoch, show_pbar=show_pbar)
            eval_total_time = time.perf_counter() - eval_start
            scheduler.step(val_loss)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            quality_record = build_quality_record(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_mae=val_mae,
                best_val_loss=min(best_val_loss, val_loss),
                learning_rate=learning_rate,
            )
            quality_history.append(quality_record)
            write_quality_log(quality_log_path, quality_history)

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
                    model_type=model_type,
                    model_config=model_config,
                    quality_snapshot=quality_record,
                    quality_history=quality_history,
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
    if quality_history:
        print(f"Saved checkpoint quality log to {quality_log_path}")


if __name__ == "__main__":
    main()
