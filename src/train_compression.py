from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import ImageSample, ImageTensorCache


ALGORITHM_VERSION = "medoid_v1"
MANIFEST_VERSION = 1
EMBEDDING_THUMB_SIZE = 8


@dataclass(frozen=True)
class CompressionResult:
    samples: list[ImageSample]
    sample_weights: list[float]
    original_count: int
    compressed_count: int
    effective_weight_sum: float
    ratio: float
    cache_hit: bool
    fingerprint: str
    manifest_path: Path


def resolve_train_compression_enabled(mode: str, model_type: str) -> bool:
    token = str(mode).strip().lower()
    if token == "on":
        return True
    if token == "off":
        return False
    if token != "auto":
        raise ValueError(f"Unsupported compression mode: {mode}")
    return str(model_type).strip().lower() == "resnet18_se"


def build_or_load_train_compression(
    train_samples: Sequence[ImageSample],
    *,
    csv_path: str,
    images_root: str,
    hdf5_path: str | None,
    split_seed: int,
    val_split: float,
    group_size: int,
    cache_dir: str | Path,
    rebuild: bool = False,
) -> CompressionResult:
    if group_size < 2:
        raise ValueError("group_size must be >= 2")

    original_count = len(train_samples)
    if original_count == 0:
        raise ValueError("Cannot compress an empty train split")

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    payload = _build_fingerprint_payload(
        train_samples=train_samples,
        csv_path=csv_path,
        images_root=images_root,
        hdf5_path=hdf5_path,
        split_seed=split_seed,
        val_split=val_split,
        group_size=group_size,
    )
    fingerprint_raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha1(fingerprint_raw.encode("utf-8")).hexdigest()
    manifest_path = cache_root / f"compression_{fingerprint}.json"

    if manifest_path.exists() and not rebuild:
        loaded = _load_manifest(manifest_path)
        return CompressionResult(
            samples=loaded["samples"],
            sample_weights=loaded["sample_weights"],
            original_count=original_count,
            compressed_count=len(loaded["samples"]),
            effective_weight_sum=float(sum(loaded["sample_weights"])),
            ratio=float(original_count / max(len(loaded["samples"]), 1)),
            cache_hit=True,
            fingerprint=fingerprint,
            manifest_path=manifest_path,
        )

    compressed_samples, sample_weights = _compress_train_samples(
        train_samples=train_samples,
        images_root=Path(images_root),
        hdf5_path=hdf5_path,
        group_size=group_size,
    )

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": payload,
        "stats": {
            "original_count": original_count,
            "compressed_count": len(compressed_samples),
            "effective_weight_sum": float(sum(sample_weights)),
            "ratio": float(original_count / max(len(compressed_samples), 1)),
        },
        "records": [
            {
                "path": str(path),
                "target": float(target),
                "weight": float(weight),
            }
            for (path, target), weight in zip(compressed_samples, sample_weights)
        ],
    }
    _atomic_write_json(manifest_path, manifest)

    return CompressionResult(
        samples=compressed_samples,
        sample_weights=sample_weights,
        original_count=original_count,
        compressed_count=len(compressed_samples),
        effective_weight_sum=float(sum(sample_weights)),
        ratio=float(original_count / max(len(compressed_samples), 1)),
        cache_hit=False,
        fingerprint=fingerprint,
        manifest_path=manifest_path,
    )


def _build_fingerprint_payload(
    *,
    train_samples: Sequence[ImageSample],
    csv_path: str,
    images_root: str,
    hdf5_path: str | None,
    split_seed: int,
    val_split: float,
    group_size: int,
) -> dict:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "group_size": int(group_size),
        "embedding_thumb_size": EMBEDDING_THUMB_SIZE,
        "split_seed": int(split_seed),
        "val_split": float(val_split),
        "train_split_signature": _train_split_signature(train_samples),
        "csv": _file_signature(csv_path),
        "images_root": _path_signature(images_root),
        "hdf5": _file_signature(hdf5_path),
    }


def _path_signature(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    return str(Path(path_value).expanduser().resolve())


def _file_signature(path_value: str | Path | None) -> dict:
    if path_value is None:
        return {"path": None, "exists": False}
    path = Path(path_value).expanduser()
    resolved = path.resolve()
    if not resolved.exists():
        return {
            "path": str(resolved),
            "exists": False,
        }
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _train_split_signature(train_samples: Sequence[ImageSample]) -> str:
    hasher = hashlib.sha1()
    for image_path, target in train_samples:
        hasher.update(str(image_path).encode("utf-8"))
        hasher.update(b"|")
        hasher.update(f"{float(target):.12g}".encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _load_manifest(path: Path) -> dict[str, list]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"Compression manifest is empty: {path}")

    samples: list[ImageSample] = []
    sample_weights: list[float] = []
    for record in records:
        samples.append((Path(record["path"]), float(record["target"])))
        sample_weights.append(float(record["weight"]))

    if any(weight <= 0 for weight in sample_weights):
        raise ValueError(f"Compression manifest has non-positive weights: {path}")

    return {
        "samples": samples,
        "sample_weights": sample_weights,
    }


def _compress_train_samples(
    *,
    train_samples: Sequence[ImageSample],
    images_root: Path,
    hdf5_path: str | None,
    group_size: int,
) -> tuple[list[ImageSample], list[float]]:
    grouped_samples: dict[str, list[ImageSample]] = defaultdict(list)
    for sample in train_samples:
        grouped_samples[sample[0].parent.name].append(sample)

    compressed_samples: list[ImageSample] = []
    sample_weights: list[float] = []

    h5_file = h5py.File(hdf5_path, "r") if hdf5_path and Path(hdf5_path).exists() else None
    h5_images = h5_file["images"] if h5_file is not None and "images" in h5_file else None

    try:
        for folder_name in sorted(grouped_samples):
            folder_samples = sorted(grouped_samples[folder_name], key=lambda item: str(item[0]))
            embeddings = _build_embeddings(folder_samples, images_root=images_root, h5_images=h5_images)
            embedding_array = embeddings.cpu().numpy()

            for group_indices in _greedy_group_indices(embedding_array, group_size=group_size):
                medoid_index = _select_medoid_index(embedding_array, group_indices)
                medoid_sample = folder_samples[medoid_index]
                compressed_samples.append(medoid_sample)
                sample_weights.append(float(len(group_indices)))
    finally:
        if h5_file is not None:
            h5_file.close()

    if not compressed_samples:
        raise ValueError("Compression produced no samples")

    return compressed_samples, sample_weights


def _build_embeddings(
    samples: Sequence[ImageSample],
    *,
    images_root: Path,
    h5_images,
) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    for image_path, _ in samples:
        try:
            image_tensor = _decode_image_tensor(image_path, images_root=images_root, h5_images=h5_images)
            vectors.append(_build_embedding_vector(image_tensor, thumb_size=EMBEDDING_THUMB_SIZE))
        except Exception:
            vectors.append(_hashed_embedding_vector(image_path, thumb_size=EMBEDDING_THUMB_SIZE))
    return torch.stack(vectors, dim=0)


def _decode_image_tensor(image_path: Path, *, images_root: Path, h5_images) -> torch.Tensor:
    if h5_images is not None:
        for key in _candidate_hdf5_keys(image_path, images_root=images_root):
            if key in h5_images:
                try:
                    binary_data = h5_images[key][()]
                    with Image.open(io.BytesIO(bytes(binary_data))) as image:
                        return ImageTensorCache._pil_image_to_tensor(image)
                except Exception:
                    continue

    return ImageTensorCache._decode_to_tensor(image_path)


def _candidate_hdf5_keys(image_path: Path, *, images_root: Path) -> list[str]:
    candidates: list[str] = []

    try:
        candidates.append(image_path.relative_to(images_root).as_posix())
    except ValueError:
        pass

    try:
        candidates.append(image_path.resolve().relative_to(images_root.resolve()).as_posix())
    except Exception:
        pass

    candidates.append((Path(image_path.parent.name) / image_path.name).as_posix())

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _build_embedding_vector(image_tensor: torch.Tensor, *, thumb_size: int) -> torch.Tensor:
    image_float = image_tensor.to(dtype=torch.float32)
    if image_float.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(image_float.shape)}")

    if image_float.shape[0] == 1:
        gray = image_float / 255.0
    else:
        red = image_float[0]
        green = image_float[1]
        blue = image_float[2]
        gray = (0.2989 * red + 0.5870 * green + 0.1140 * blue).unsqueeze(0) / 255.0

    thumbnail = F.interpolate(
        gray.unsqueeze(0),
        size=(thumb_size, thumb_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    vector = thumbnail.reshape(-1)

    vector = vector - vector.mean()
    norm = torch.linalg.norm(vector)
    if torch.isfinite(norm) and norm > 0:
        vector = vector / norm

    return vector.contiguous()


def _hashed_embedding_vector(image_path: Path, *, thumb_size: int) -> torch.Tensor:
    vector_size = thumb_size * thumb_size
    digest = hashlib.sha1(str(image_path).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2 ** 32)
    rng = np.random.default_rng(seed)
    vector = torch.from_numpy(rng.standard_normal(vector_size).astype(np.float32))
    norm = torch.linalg.norm(vector)
    if torch.isfinite(norm) and norm > 0:
        vector = vector / norm
    return vector


def _greedy_group_indices(embedding_array: np.ndarray, *, group_size: int) -> list[np.ndarray]:
    sample_count = int(embedding_array.shape[0])
    remaining = np.arange(sample_count, dtype=np.int64)
    groups: list[np.ndarray] = []

    while remaining.size > 0:
        if remaining.size <= group_size:
            groups.append(remaining.copy())
            break

        anchor_index = int(remaining[0])
        anchor_vec = embedding_array[anchor_index : anchor_index + 1]
        candidates = embedding_array[remaining]
        distances = np.sum((candidates - anchor_vec) ** 2, axis=1)
        nearest_positions = np.argsort(distances, kind="mergesort")[:group_size]
        chosen = remaining[nearest_positions]
        groups.append(chosen.astype(np.int64, copy=False))

        keep_mask = np.ones(remaining.shape[0], dtype=bool)
        keep_mask[nearest_positions] = False
        remaining = remaining[keep_mask]

    return groups


def _select_medoid_index(embedding_array: np.ndarray, group_indices: np.ndarray) -> int:
    if int(group_indices.size) == 1:
        return int(group_indices[0])

    group_vectors = embedding_array[group_indices]
    pairwise = np.sum((group_vectors[:, None, :] - group_vectors[None, :, :]) ** 2, axis=2)
    medoid_position = int(np.argmin(np.sum(pairwise, axis=1)))
    return int(group_indices[medoid_position])
