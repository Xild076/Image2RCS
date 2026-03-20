import argparse
import time
from pathlib import Path
from typing import Iterable

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

from dataset import ImageTensorCache, build_image_transform, inverse_transform_target_tensor
from device_utils import configure_torch_for_device, describe_device, resolve_device
from model import build_model, normalize_model_type
from perf_utils import autotune_num_workers, configure_cpu_threads, format_worker_timings, parse_num_workers


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    target_mode = checkpoint.get("target_mode", "log1p")
    image_size = int(checkpoint.get("image_size", 224))
    model_type = normalize_model_type(checkpoint.get("model_type", "resnet18_baseline"))
    model_config = checkpoint.get("model_config", {})
    if not isinstance(model_config, dict):
        model_config = {}
    model = build_model(
        pretrained=False,
        out_dim=1,
        model_type=model_type,
        model_config=model_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = build_image_transform(image_size=image_size, train=False)
    return model, transform, target_mode


class InferenceImageDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        transform,
        cache_mode: str,
        cache_dir: str,
        memory_cache_items: int,
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.cache = ImageTensorCache(
            mode=cache_mode,
            cache_dir=cache_dir,
            max_memory_items=memory_cache_items,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image_tensor = self.cache.get(image_path)
        return self.transform(image_tensor), str(image_path)


@torch.inference_mode()
def predict_images(
    model,
    transform,
    image_paths: list[Path],
    device: torch.device,
    target_mode: str,
    batch_size: int,
    num_workers: str | int,
    cache_mode: str,
    cache_dir: str,
    memory_cache_items: int,
    profile: bool,
):
    dataset = InferenceImageDataset(
        image_paths=image_paths,
        transform=transform,
        cache_mode=cache_mode,
        cache_dir=cache_dir,
        memory_cache_items=memory_cache_items,
    )

    pin_memory = device.type == "cuda"
    parsed_num_workers = parse_num_workers(num_workers)
    worker_timings: dict[int, float] = {}
    if parsed_num_workers == "auto":
        resolved_num_workers, worker_timings = autotune_num_workers(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )
    else:
        resolved_num_workers = int(parsed_num_workers)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": resolved_num_workers,
        "pin_memory": pin_memory,
        # One inference pass does not benefit from persistent workers and keeps RSS lower.
        "persistent_workers": False,
    }
    if resolved_num_workers > 0:
        # Reduce prefetched batches in worker queues to keep RAM usage down.
        loader_kwargs["prefetch_factor"] = 1
    loader = DataLoader(**loader_kwargs)

    predictions: list[tuple[str, float]] = []
    data_load_time = 0.0
    compute_time = 0.0
    wait_start = time.perf_counter()

    for batch_images, batch_paths in loader:
        data_load_time += time.perf_counter() - wait_start
        step_start = time.perf_counter()

        non_blocking = device.type == "cuda"
        batch_images = batch_images.to(device, non_blocking=non_blocking)
        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            batch_pred = model(batch_images).squeeze(1)
        batch_rcs = inverse_transform_target_tensor(batch_pred, mode=target_mode)
        batch_rcs = torch.clamp_min(batch_rcs, 0.0)
        batch_values = batch_rcs.detach().cpu().tolist()

        predictions.extend((path, float(value)) for path, value in zip(batch_paths, batch_values))
        compute_time += time.perf_counter() - step_start
        wait_start = time.perf_counter()

    profile_stats = {
        "num_workers": resolved_num_workers,
        "worker_timings": worker_timings,
        "data_load": data_load_time,
        "compute": compute_time,
    }
    if profile:
        print(
            f"[profile] num_workers={resolved_num_workers} "
            f"(auto timings: {format_worker_timings(worker_timings)})"
        )
    return predictions, profile_stats


def iter_images(path: Path) -> Iterable[Path]:
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif", ".tif", ".tiff"}
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for file_path in sorted(path.iterdir()):
            if file_path.suffix.lower() in valid_ext:
                yield file_path


def main():
    parser = argparse.ArgumentParser(description="Run RCS inference on a single image or image folder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to an image or a directory of images")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto|gpu|cuda|cuda:N|mps|cpu (auto prefers CUDA, then MPS, then CPU)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size for directory input")
    parser.add_argument("--num-workers", type=str, default="auto", help='DataLoader workers, integer or "auto"')
    parser.add_argument("--cache-mode", type=str, default="memory", choices=["off", "memory", "disk"])
    parser.add_argument("--cache-dir", type=str, default=".cache/image2rcs")
    parser.add_argument(
        "--memory-cache-items",
        type=int,
        default=256,
        help="Max decoded images kept in RAM when --cache-mode=memory.",
    )
    parser.add_argument("--cpu-threads", type=int, default=0, help="Set torch CPU threads; 0 keeps runtime default")
    parser.add_argument("--profile", action="store_true", help="Print timing and throughput")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.memory_cache_items < 1:
        raise ValueError("--memory-cache-items must be >= 1")
    configure_cpu_threads(args.cpu_threads if args.cpu_threads > 0 else None)

    device = resolve_device(args.device)
    configure_torch_for_device(device)
    print(f"Using device: {describe_device(device)}")
    model, transform, target_mode = load_checkpoint(args.checkpoint, device)

    input_path = Path(args.input)
    image_paths = list(iter_images(input_path))
    if not image_paths:
        raise ValueError(f"No valid images found at {input_path}")

    start = time.perf_counter()
    predictions, profile_stats = predict_images(
        model=model,
        transform=transform,
        image_paths=image_paths,
        device=device,
        target_mode=target_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_mode=args.cache_mode,
        cache_dir=args.cache_dir,
        memory_cache_items=args.memory_cache_items,
        profile=args.profile,
    )
    elapsed = time.perf_counter() - start

    for image_path, prediction in predictions:
        print(f"{image_path}: {prediction:.6f} m^2")

    if args.profile:
        throughput = len(predictions) / max(elapsed, 1e-9)
        print(
            f"[profile] images={len(predictions)} | total={elapsed:.3f}s "
            f"| data_load={profile_stats['data_load']:.3f}s "
            f"| compute={profile_stats['compute']:.3f}s "
            f"| throughput={throughput:.2f} img/s"
        )


if __name__ == "__main__":
    main()
