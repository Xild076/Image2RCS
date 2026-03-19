from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Literal, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    from pillow_heif import register_heif_opener
except ImportError:  # Optional dependency; JPEG/PNG/WebP still work without it.
    register_heif_opener = None
else:
    register_heif_opener()


ImageSample = Tuple[Path, float]
CacheMode = Literal["off", "memory", "disk"]
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif", ".tif", ".tiff"}
_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]
DEFAULT_CACHE_DIR = Path(".cache/image2rcs")


def resolve_image_folder(folder_value: str, images_root: Path) -> Path:
    folder_path = Path(folder_value)
    if folder_path.is_absolute():
        return folder_path
    if folder_path.parts[:2] == ("data", "images"):
        return folder_path
    return images_root / folder_path


def load_image_samples(csv_path: str, images_root: str = "data/images") -> List[ImageSample]:
    csv_file = Path(csv_path)
    root = Path(images_root)
    df = pd.read_csv(csv_file)

    required_columns = {"image_folder", "rcs"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {csv_file}: {missing_text}")

    samples: List[ImageSample] = []
    for row in df.itertuples(index=False):
        folder = resolve_image_folder(str(row.image_folder), root)
        if not folder.exists() or not folder.is_dir():
            continue
        rcs_value = float(row.rcs)
        image_files = [p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS]
        for image_file in sorted(image_files):
            samples.append((image_file, rcs_value))

    if not samples:
        raise ValueError("No image samples found. Check CSV paths and image directories.")

    return samples


def build_image_transform(image_size: int = 224, train: bool = False) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32), interpolation=InterpolationMode.BILINEAR, antialias=True),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), interpolation=InterpolationMode.BILINEAR, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
                transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
        ]
    )


def transform_target(value: float, mode: str = "log1p") -> float:
    if mode == "log1p":
        return float(math.log1p(value))
    if mode == "none":
        return float(value)
    raise ValueError(f"Unsupported target mode: {mode}")


def inverse_transform_target(value: float, mode: str = "log1p") -> float:
    if mode == "log1p":
        return float(math.expm1(value))
    if mode == "none":
        return float(value)
    raise ValueError(f"Unsupported target mode: {mode}")


def inverse_transform_target_tensor(values: torch.Tensor, mode: str = "log1p") -> torch.Tensor:
    if mode == "log1p":
        return torch.expm1(values)
    if mode == "none":
        return values
    raise ValueError(f"Unsupported target mode: {mode}")


class ImageTensorCache:
    def __init__(
        self,
        mode: CacheMode = "disk",
        cache_dir: str | Path | None = None,
        max_memory_items: int = 256,
    ):
        self.mode: CacheMode = mode
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        if self.mode == "disk":
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if max_memory_items < 1:
            raise ValueError("max_memory_items must be >= 1")
        self.max_memory_items = int(max_memory_items)
        self._memory_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.evictions = 0

    def _cache_identity(self, image_path: Path) -> str:
        resolved = image_path.resolve()
        stat = resolved.stat()
        return f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}"

    def _cache_path(self, cache_identity: str) -> Path:
        digest = hashlib.sha1(cache_identity.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.pt"

    @staticmethod
    def _pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
        normalized = ImageOps.exif_transpose(image)
        try:
            rgb = normalized.convert("RGB")
            try:
                return TF.pil_to_tensor(rgb).contiguous()
            finally:
                rgb.close()
        finally:
            if normalized is not image:
                normalized.close()

    @staticmethod
    def _decode_heif_with_sips(image_path: Path) -> torch.Tensor | None:
        if image_path.suffix.lower() not in {".heic", ".heif"}:
            return None
        if shutil.which("sips") is None:
            return None

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            subprocess.run(
                ["sips", "-s", "format", "jpeg", str(image_path), "--out", str(tmp_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with Image.open(tmp_path) as converted:
                return ImageTensorCache._pil_image_to_tensor(converted)
        except (subprocess.SubprocessError, OSError, ValueError):
            return None
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _decode_to_tensor(image_path: Path) -> torch.Tensor:
        try:
            with Image.open(image_path) as image:
                return ImageTensorCache._pil_image_to_tensor(image)
        except (OSError, ValueError):
            converted = ImageTensorCache._decode_heif_with_sips(image_path)
            if converted is not None:
                return converted
            raise

    def _load_disk_tensor(self, cache_path: Path) -> torch.Tensor | None:
        if not cache_path.exists():
            return None
        try:
            tensor = torch.load(cache_path, map_location="cpu")
            if torch.is_tensor(tensor):
                self.hits += 1
                return tensor
        except Exception:
            pass
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None

    def get(self, image_path: Path) -> torch.Tensor:
        if self.mode == "off":
            return self._decode_to_tensor(image_path)

        cache_identity = self._cache_identity(image_path)

        if self.mode == "memory":
            cached = self._memory_cache.get(cache_identity)
            if cached is not None:
                self._memory_cache.move_to_end(cache_identity)
                self.hits += 1
                return cached
            tensor = self._decode_to_tensor(image_path)
            self._memory_cache[cache_identity] = tensor
            if len(self._memory_cache) > self.max_memory_items:
                self._memory_cache.popitem(last=False)
                self.evictions += 1
            self.misses += 1
            return tensor

        cache_path = self._cache_path(cache_identity)
        cached = self._load_disk_tensor(cache_path)
        if cached is not None:
            return cached

        tensor = self._decode_to_tensor(image_path)
        self.misses += 1

        tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            torch.save(tensor, tmp_path)
            os.replace(tmp_path, cache_path)
            self.writes += 1
        except OSError:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

        return tensor

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "evictions": self.evictions,
            "memory_entries": len(self._memory_cache),
        }


class AircraftRCSDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[ImageSample],
        image_transform: Callable | None = None,
        target_mode: str = "log1p",
        cache_mode: CacheMode = "disk",
        cache_dir: str | Path | None = None,
        memory_cache_items: int = 256,
    ):
        self.samples = list(samples)
        self.image_transform = image_transform if image_transform is not None else build_image_transform()
        self.target_mode = target_mode
        self.image_cache = ImageTensorCache(
            mode=cache_mode,
            cache_dir=cache_dir,
            max_memory_items=memory_cache_items,
        )

        targets = [transform_target(float(rcs_value), mode=target_mode) for _, rcs_value in self.samples]
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, _ = self.samples[index]
        image_tensor = self.image_cache.get(image_path)
        image_tensor = self.image_transform(image_tensor)
        return image_tensor, self.targets[index]
