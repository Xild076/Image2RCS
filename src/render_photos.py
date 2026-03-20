from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFilter


FIT_MARGIN = 1.08
EXTENT_MARGIN = 1.45
DEFAULT_ORIENTATION_MODES = (
    ("upright", 0.0),
    ("inverted", 180.0),
    ("vertical_cw", 90.0),
    ("vertical_ccw", 270.0),
)
SKY_PRESETS = (
    {
        "name": "lightblue",
        "bg_color": [0.55, 0.72, 0.96, 1.0],
        "ambient": [0.40, 0.40, 0.42],
        "dir_intensity": 2.8,
        "fill_intensity": 12.0,
    },
    {
        "name": "duskorange",
        "bg_color": [0.95, 0.55, 0.32, 1.0],
        "ambient": [0.32, 0.25, 0.20],
        "dir_intensity": 2.2,
        "fill_intensity": 10.0,
    },
    {
        "name": "black",
        "bg_color": [0.02, 0.02, 0.04, 1.0],
        "ambient": [0.12, 0.12, 0.15],
        "dir_intensity": 1.8,
        "fill_intensity": 8.0,
    },
    {
        "name": "stormblue",
        "bg_color": [0.32, 0.43, 0.59, 1.0],
        "ambient": [0.23, 0.25, 0.30],
        "dir_intensity": 2.0,
        "fill_intensity": 8.6,
    },
    {
        "name": "overcast",
        "bg_color": [0.72, 0.77, 0.82, 1.0],
        "ambient": [0.46, 0.46, 0.47],
        "dir_intensity": 1.7,
        "fill_intensity": 7.8,
    },
    {
        "name": "mintsky",
        "bg_color": [0.58, 0.86, 0.80, 1.0],
        "ambient": [0.35, 0.39, 0.38],
        "dir_intensity": 2.6,
        "fill_intensity": 10.4,
    },
    {
        "name": "sandhaze",
        "bg_color": [0.90, 0.78, 0.60, 1.0],
        "ambient": [0.38, 0.33, 0.28],
        "dir_intensity": 2.4,
        "fill_intensity": 9.8,
    },
    {
        "name": "sunsetpink",
        "bg_color": [0.94, 0.60, 0.66, 1.0],
        "ambient": [0.36, 0.30, 0.31],
        "dir_intensity": 2.3,
        "fill_intensity": 9.2,
    },
)
LIGHT_POSITION_PRESETS = (
    {
        "name": "left_high",
        "dir_offset": [1.4, -1.0, 1.8],
        "fill_offset": [1.0, -0.8, 1.0],
    },
    {
        "name": "right_high",
        "dir_offset": [-1.4, -1.0, 1.8],
        "fill_offset": [-1.0, -0.8, 1.0],
    },
    {
        "name": "front_low",
        "dir_offset": [0.0, -1.8, 1.0],
        "fill_offset": [0.0, -1.1, 0.8],
    },
    {
        "name": "back_rim",
        "dir_offset": [0.0, 1.8, 1.4],
        "fill_offset": [0.0, 1.0, 1.0],
    },
    {
        "name": "top_down",
        "dir_offset": [0.0, 0.0, 2.4],
        "fill_offset": [0.0, -0.7, 0.8],
    },
    {
        "name": "diag_front",
        "dir_offset": [1.2, -1.7, 1.3],
        "fill_offset": [0.8, -1.0, 0.9],
    },
)
LIGHT_STYLE_PRESETS = (
    {
        "name": "soft",
        "ambient_scale": 1.22,
        "dir_scale": 0.68,
        "fill_scale": 0.76,
    },
    {
        "name": "neutral",
        "ambient_scale": 1.0,
        "dir_scale": 1.0,
        "fill_scale": 1.0,
    },
    {
        "name": "harsh",
        "ambient_scale": 0.72,
        "dir_scale": 1.62,
        "fill_scale": 0.64,
    },
    {
        "name": "bright",
        "ambient_scale": 1.10,
        "dir_scale": 1.36,
        "fill_scale": 1.24,
    },
)
DISTANCE_LABELS = ("near", "mid", "far", "xfar", "ultra", "extreme")
RESOLUTION_PROFILES: dict[str, tuple[tuple[int, int], ...]] = {
    "fixed": (),
    "balanced_mix": (
        (448, 448),
        (512, 384),
        (384, 512),
        (640, 360),
        (360, 640),
    ),
    "wide_heavy": (
        (448, 448),
        (576, 320),
        (320, 576),
        (640, 360),
        (360, 640),
    ),
    "near_square": (
        (448, 448),
        (480, 416),
        (416, 480),
    ),
}
NOISE_PROFILE_SETTINGS: dict[str, dict[str, float]] = {
    "light": {
        "background_replace_weight": 0.72,
        "background_replace_weight_min": 0.35,
        "background_replace_weight_max": 0.84,
        "airport_visible_probability": 0.46,
        "noise_scale_min": 0.55,
        "noise_scale_max": 1.20,
        "bg_grain_sigma": 8.0,
        "fg_grain_sigma": 2.0,
        "vignette_strength": 0.07,
        "blur_max": 0.55,
        "compression_prob": 0.22,
        "compression_quality_min": 72.0,
        "compression_quality_max": 90.0,
        "color_drift": 0.06,
        "fog_strength": 0.14,
        "structure_density": 0.36,
        "foreground_overlay_alpha": 34.0,
        "artifact_prob": 0.18,
        "artifact_strength": 0.28,
    },
    "balanced": {
        "background_replace_weight": 0.88,
        "background_replace_weight_min": 0.48,
        "background_replace_weight_max": 0.94,
        "airport_visible_probability": 0.66,
        "noise_scale_min": 0.70,
        "noise_scale_max": 1.38,
        "bg_grain_sigma": 13.0,
        "fg_grain_sigma": 3.8,
        "vignette_strength": 0.12,
        "blur_max": 0.9,
        "compression_prob": 0.34,
        "compression_quality_min": 56.0,
        "compression_quality_max": 82.0,
        "color_drift": 0.1,
        "fog_strength": 0.2,
        "structure_density": 0.56,
        "foreground_overlay_alpha": 46.0,
        "artifact_prob": 0.28,
        "artifact_strength": 0.42,
    },
    "aggressive_background": {
        "background_replace_weight": 0.97,
        "background_replace_weight_min": 0.58,
        "background_replace_weight_max": 1.0,
        "airport_visible_probability": 0.79,
        "noise_scale_min": 0.82,
        "noise_scale_max": 1.64,
        "bg_grain_sigma": 21.0,
        "fg_grain_sigma": 5.5,
        "vignette_strength": 0.19,
        "blur_max": 1.35,
        "compression_prob": 0.52,
        "compression_quality_min": 42.0,
        "compression_quality_max": 74.0,
        "color_drift": 0.17,
        "fog_strength": 0.28,
        "structure_density": 0.82,
        "foreground_overlay_alpha": 62.0,
        "artifact_prob": 0.45,
        "artifact_strength": 0.62,
    },
}
BACKGROUND_SCENE_PROFILES = ("airport_heavy",)
REVIEW_VIEW_SPECS = (
    ("front", 0, 10, 0.0),
    ("front_high", 0, 35, 0.0),
    ("front_low", 0, -25, 0.0),
    ("right", 90, 10, 0.0),
    ("rear", 180, 10, 0.0),
    ("left", 270, 10, 0.0),
    ("diag_fl", 45, 20, 0.0),
    ("diag_fr", 315, 20, 0.0),
    ("diag_rl", 225, 20, 0.0),
    ("diag_rr", 135, 20, 0.0),
    ("top", 0, 70, 0.0),
    ("bottom", 180, -65, 180.0),
)


@dataclass(frozen=True)
class RenderConfig:
    render_size: int
    azimuth_step_deg: int
    elevation_min_deg: int
    elevation_max_deg: int
    elevation_step_deg: int
    fit_margin: float = FIT_MARGIN
    extent_margin: float = EXTENT_MARGIN
    gc_every_frames: int = 40
    orientation_modes: tuple[tuple[str, float], ...] = DEFAULT_ORIENTATION_MODES
    distance_min_scale: float = 0.55
    distance_max_scale: float = 1.95
    offcenter_xy_scale: float = 0.34
    offcenter_z_scale: float = 0.20
    center_hold_probability: float = 0.22
    cloud_probability: float = 0.55
    cloud_max_layers: int = 3
    frame_seed: int = 23
    output_format: str = "webp"
    output_quality: int = 78
    resolution_profile: str = "balanced_mix"
    noise_profile: str = "aggressive_background"
    background_scene_profile: str = "airport_heavy"
    combination_strategy: str = "exhaustive"
    variants_per_combo: int = 1
    distance_bins: int = 3
    target_images_per_model: int = 400
    combo_selection: str = "pairwise"

    @property
    def elevation_degs(self) -> list[int]:
        if self.elevation_step_deg <= 0:
            raise ValueError("Elevation step must be greater than 0.")
        if self.elevation_max_deg < self.elevation_min_deg:
            raise ValueError("Elevation max must be >= elevation min.")
        return list(range(self.elevation_min_deg, self.elevation_max_deg + 1, self.elevation_step_deg))

    @property
    def resolution_options(self) -> list[tuple[str, int, int]]:
        profile = str(self.resolution_profile).strip().lower()
        if profile == "fixed":
            size = int(self.render_size)
            return [(f"{size}x{size}", size, size)]

        options = RESOLUTION_PROFILES.get(profile)
        if not options:
            choices = ", ".join(sorted(RESOLUTION_PROFILES))
            raise ValueError(f"Unsupported resolution profile: {self.resolution_profile}. Expected one of: {choices}")
        return [(f"{w}x{h}", int(w), int(h)) for w, h in options]


def canonicalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def slugify(text: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return token or "model"


def folder_token(text: str) -> str:
    token = re.sub(r"[^a-z0-9_-]+", "_", text.lower()).strip("_")
    return token or "model"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def discover_mesh_files(render_root: Path) -> list[Path]:
    patterns = ("*.glb", "*.gltf", "*.obj")
    mesh_paths: list[Path] = []
    for pattern in patterns:
        mesh_paths.extend(render_root.rglob(pattern))
    filtered = [
        path
        for path in mesh_paths
        if path.is_file()
        and "images" not in {part.lower() for part in path.parts}
        and not path.name.lower().endswith("_images")
    ]
    return sorted(set(filtered))


def model_key_from_mesh_path(mesh_path: Path, render_root: Path) -> str:
    relative = mesh_path.relative_to(render_root)
    if len(relative.parts) > 1:
        for part in relative.parts[:-1]:
            if part.lower() not in {"source", "model", "models", "meshes"}:
                return part
    return mesh_path.stem


def build_existing_images_index(images_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not images_root.exists():
        return index
    for child in images_root.iterdir():
        if child.is_dir() and child.name.endswith("_images"):
            key = canonicalize(child.name[: -len("_images")])
            if key and key not in index:
                index[key] = child
    return index


def resolve_output_dir(
    mesh_path: Path,
    render_root: Path,
    images_root: Path,
    images_index: dict[str, Path],
) -> Path:
    model_key = model_key_from_mesh_path(mesh_path, render_root)
    model_token = canonicalize(model_key)
    existing = images_index.get(model_token)
    if existing is not None:
        return existing

    fallback = images_root / f"{folder_token(model_key)}_images"
    fallback.mkdir(parents=True, exist_ok=True)
    images_index[model_token] = fallback
    return fallback


def clear_existing_renders(output_dir: Path, prefix: str) -> int:
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    removed = 0
    for image_file in output_dir.glob(f"{prefix}*"):
        if image_file.suffix.lower() not in valid_exts:
            continue
        image_file.unlink(missing_ok=True)
        removed += 1
    return removed


def check_missing_textures(mesh_path: Path) -> None:
    mtl_path = mesh_path.with_suffix(".mtl")
    if not mtl_path.exists():
        return

    missing: list[str] = []
    lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("map_Kd "):
            continue
        tex_rel = stripped.split(None, 1)[1]
        tex_path = mtl_path.parent / tex_rel
        if not tex_path.exists():
            missing.append(tex_rel)

    if missing:
        sample = ", ".join(missing[:3])
        print(
            f"[warn] {mesh_path.name}: missing {len(missing)} texture files from {mtl_path.name}. "
            f"Example: {sample}"
        )


def to_renderable_meshes(asset) -> list:
    if isinstance(asset, trimesh.Scene):
        if not asset.geometry:
            raise ValueError("Loaded scene has no geometry to render.")
        meshes = [geometry for geometry in asset.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("Loaded scene has no triangular mesh geometry to render.")
        return meshes
    if isinstance(asset, trimesh.Trimesh):
        return [asset]
    raise ValueError(f"Unsupported geometry type for rendering: {type(asset).__name__}")


def to_untextured_meshes(meshes: list) -> list:
    sanitized = []
    default_color = np.array([190, 190, 190, 255], dtype=np.uint8)
    for mesh in meshes:
        mesh_copy = mesh.copy()
        if hasattr(mesh_copy, "faces") and len(mesh_copy.faces) > 0:
            face_colors = np.tile(default_color, (len(mesh_copy.faces), 1))
            mesh_copy.visual = trimesh.visual.color.ColorVisuals(mesh=mesh_copy, face_colors=face_colors)
        elif hasattr(mesh_copy, "vertices") and len(mesh_copy.vertices) > 0:
            vertex_colors = np.tile(default_color, (len(mesh_copy.vertices), 1))
            mesh_copy.visual = trimesh.visual.color.ColorVisuals(mesh=mesh_copy, vertex_colors=vertex_colors)
        sanitized.append(mesh_copy)
    return sanitized


def build_pyrender_mesh(mesh_path: Path, meshes: list, force_untextured: bool = False) -> pyrender.Mesh:
    if force_untextured:
        return pyrender.Mesh.from_trimesh(to_untextured_meshes(meshes), smooth=False)
    try:
        return pyrender.Mesh.from_trimesh(meshes)
    except (TypeError, ValueError) as exc:
        msg = str(exc).lower()
        if "texture" not in msg and "reformat" not in msg:
            raise
        print(f"[warn] {mesh_path.name}: texture/material conversion failed ({exc}); rendering without textures.")
        return pyrender.Mesh.from_trimesh(to_untextured_meshes(meshes), smooth=False)


def compute_bounds(mesh_or_meshes):
    bounds_list = [mesh.bounds for mesh in mesh_or_meshes if hasattr(mesh, "bounds")]
    if not bounds_list:
        raise ValueError("No valid mesh bounds found in loaded geometry.")
    mins = np.min(np.stack([bound[0] for bound in bounds_list], axis=0), axis=0)
    maxs = np.max(np.stack([bound[1] for bound in bounds_list], axis=0), axis=0)
    center = 0.5 * (mins + maxs)
    extents = maxs - mins
    diagonal = float(np.linalg.norm(extents))
    return center, extents, diagonal


def safe_normalize(vector: np.ndarray, eps: float = 1e-9) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < eps:
        return None
    normalized = vector / norm
    if not np.all(np.isfinite(normalized)):
        return None
    return normalized


def look_at(
    camera_position,
    target=np.array([0.0, 0.0, 0.0]),
    up=np.array([0.0, 0.0, 1.0]),
    roll_deg: float = 0.0,
):
    camera_position = np.asarray(camera_position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    if not np.all(np.isfinite(camera_position)):
        camera_position = np.zeros(3, dtype=np.float64)
    if not np.all(np.isfinite(target)):
        target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    if not np.all(np.isfinite(up)):
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    forward = safe_normalize(target - camera_position)
    if forward is None:
        forward = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    up = safe_normalize(up)
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if abs(float(np.dot(forward, up))) > 0.985:
        fallback_ups = (
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        for candidate in fallback_ups:
            if abs(float(np.dot(forward, candidate))) < 0.985:
                up = candidate
                break

    right = safe_normalize(np.cross(forward, up))
    if right is None:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    true_up = safe_normalize(np.cross(right, forward))
    if true_up is None:
        true_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    roll = math.radians(roll_deg)
    cos_r = math.cos(roll)
    sin_r = math.sin(roll)
    right_r = safe_normalize((cos_r * right) + (sin_r * true_up))
    up_r = safe_normalize((-sin_r * right) + (cos_r * true_up))
    if right_r is None:
        right_r = right
    if up_r is None:
        up_r = true_up

    pose = np.eye(4)
    pose[:3, 0] = right_r
    pose[:3, 1] = up_r
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_position
    return pose


def load_mesh_asset(mesh_path: Path):
    try:
        return trimesh.load(mesh_path, force="scene", process=False)
    except TypeError:
        return trimesh.load(mesh_path, force="scene")


def save_png_array(color_buffer: np.ndarray, output_path: Path) -> None:
    image = Image.fromarray(color_buffer)
    try:
        image.save(output_path)
    finally:
        image.close()


def normalize_output_format(value: str) -> str:
    token = value.strip().lower()
    if token == "jpg":
        token = "jpeg"
    if token not in {"png", "jpeg", "webp"}:
        raise ValueError(f"Unsupported output format: {value}. Expected png, jpeg, or webp.")
    return token


def output_extension(output_format: str) -> str:
    return {
        "png": ".png",
        "jpeg": ".jpg",
        "webp": ".webp",
    }[normalize_output_format(output_format)]


def save_render_array(color_buffer: np.ndarray, output_path: Path, output_format: str, output_quality: int) -> None:
    image = Image.fromarray(color_buffer)
    fmt = normalize_output_format(output_format)

    rgb_image: Image.Image | None = None
    try:
        if fmt == "png":
            image.save(output_path, format="PNG", optimize=True, compress_level=9)
            return

        rgb_image = image.convert("RGB")
        if fmt == "jpeg":
            rgb_image.save(
                output_path,
                format="JPEG",
                quality=int(output_quality),
                optimize=True,
                progressive=True,
            )
            return

        rgb_image.save(
            output_path,
            format="WEBP",
            quality=int(output_quality),
            method=6,
        )
    finally:
        if rgb_image is not None:
            rgb_image.close()
        image.close()


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.blake2b(f"{base_seed}:{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def frame_rng(model_seed: int, frame_index: int) -> np.random.Generator:
    frame_seed = (model_seed + ((frame_index + 1) * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(frame_seed)


def build_distance_bins(render_config: RenderConfig) -> list[tuple[str, float, float]]:
    if render_config.distance_bins <= 0:
        return [("mid", render_config.distance_min_scale, render_config.distance_max_scale)]

    lo = float(render_config.distance_min_scale)
    hi = float(render_config.distance_max_scale)
    if hi <= lo:
        return [("mid", lo, lo)]

    bin_edges = np.linspace(lo, hi, num=render_config.distance_bins + 1, dtype=np.float64)
    bins: list[tuple[str, float, float]] = []
    for idx in range(render_config.distance_bins):
        label = DISTANCE_LABELS[idx] if idx < len(DISTANCE_LABELS) else f"dist{idx + 1:02d}"
        bins.append((label, float(bin_edges[idx]), float(bin_edges[idx + 1])))
    return bins


def build_frame_combinations(
    render_config: RenderConfig,
) -> list[tuple[dict, dict, dict, str, float, float, str, int, int, int]]:
    distance_bins = build_distance_bins(render_config)
    resolutions = render_config.resolution_options
    combos: list[tuple[dict, dict, dict, str, float, float, str, int, int, int]] = []

    if render_config.combination_strategy == "cyclic":
        cycle_len = max(
            len(SKY_PRESETS),
            len(LIGHT_POSITION_PRESETS),
            len(LIGHT_STYLE_PRESETS),
            len(distance_bins),
            len(resolutions),
        )
        for cycle_idx in range(cycle_len):
            sky = SKY_PRESETS[cycle_idx % len(SKY_PRESETS)]
            light = LIGHT_POSITION_PRESETS[cycle_idx % len(LIGHT_POSITION_PRESETS)]
            light_style = LIGHT_STYLE_PRESETS[cycle_idx % len(LIGHT_STYLE_PRESETS)]
            dist_name, dist_min, dist_max = distance_bins[cycle_idx % len(distance_bins)]
            resolution_name, width, height = resolutions[cycle_idx % len(resolutions)]
            for variant_idx in range(render_config.variants_per_combo):
                combos.append(
                    (
                        sky,
                        light,
                        light_style,
                        dist_name,
                        dist_min,
                        dist_max,
                        resolution_name,
                        width,
                        height,
                        variant_idx,
                    )
                )
        return combos

    for sky in SKY_PRESETS:
        for light in LIGHT_POSITION_PRESETS:
            for light_style in LIGHT_STYLE_PRESETS:
                for dist_name, dist_min, dist_max in distance_bins:
                    for resolution_name, width, height in resolutions:
                        for variant_idx in range(render_config.variants_per_combo):
                            combos.append(
                                (
                                    sky,
                                    light,
                                    light_style,
                                    dist_name,
                                    dist_min,
                                    dist_max,
                                    resolution_name,
                                    width,
                                    height,
                                    variant_idx,
                                )
                            )
    return combos


def build_pose_grid(render_config: RenderConfig) -> list[tuple[str, float, int, int]]:
    azimuth_values = list(range(0, 360, render_config.azimuth_step_deg))
    elevation_values = render_config.elevation_degs
    per_orientation: list[list[tuple[str, float, int, int]]] = []
    for orientation_name, roll_deg in render_config.orientation_modes:
        orientation_poses: list[tuple[str, float, int, int]] = []
        for azimuth_deg in azimuth_values:
            for elevation_deg in elevation_values:
                orientation_poses.append((orientation_name, roll_deg, azimuth_deg, elevation_deg))
        per_orientation.append(orientation_poses)

    interleaved: list[tuple[str, float, int, int]] = []
    max_len = max((len(bucket) for bucket in per_orientation), default=0)
    for idx in range(max_len):
        for bucket in per_orientation:
            if idx < len(bucket):
                interleaved.append(bucket[idx])
    return interleaved


def select_even_indices(total: int, count: int, offset: float = 0.0) -> list[int]:
    if count <= 0 or total <= 0:
        return []
    if count >= total:
        return list(range(total))
    step = total / float(count)
    indices = []
    for i in range(count):
        value = ((i + 0.5) * step + offset) % total
        indices.append(int(math.floor(value)))
    return indices


def build_pose_schedule(
    poses: list[tuple[str, float, int, int]],
    total_frames: int,
    model_seed: int,
) -> list[tuple[str, float, int, int]]:
    if not poses:
        raise ValueError("No camera poses available for scheduling.")
    if total_frames <= 0:
        return []

    offset_idx = int(model_seed % len(poses))
    rotated = poses[offset_idx:] + poses[:offset_idx]
    if total_frames <= len(rotated):
        extra_offset = float((model_seed >> 8) % len(rotated))
        return [rotated[idx] for idx in select_even_indices(len(rotated), total_frames, offset=extra_offset)]

    repeats = total_frames // len(rotated)
    remainder = total_frames % len(rotated)
    schedule = rotated * repeats
    if remainder > 0:
        extra_offset = float((model_seed >> 8) % len(rotated))
        schedule.extend(rotated[idx] for idx in select_even_indices(len(rotated), remainder, offset=extra_offset))
    return schedule


def combo_factor_keys(combo: tuple[dict, dict, dict, str, float, float, str, int, int, int]) -> tuple[str, str, str, str, str]:
    sky, light, light_style, distance_name, _, _, resolution_name, _, _, _ = combo
    return sky["name"], light["name"], light_style["name"], distance_name, resolution_name


def combo_pair_keys(combo: tuple[dict, dict, dict, str, float, float, str, int, int, int]) -> list[tuple[str, str, str]]:
    factor_names = ("sky", "light", "style", "distance", "resolution")
    factor_values = combo_factor_keys(combo)
    pairs: list[tuple[str, str, str]] = []
    for i, j in combinations(range(len(factor_values)), 2):
        pairs.append((f"{factor_names[i]}_{factor_names[j]}", factor_values[i], factor_values[j]))
    return pairs


def build_combo_schedule(
    frame_combinations: list[tuple[dict, dict, dict, str, float, float, str, int, int, int]],
    total_frames: int,
    selection_mode: str,
    model_seed: int,
) -> list[tuple[dict, dict, dict, str, float, float, str, int, int, int]]:
    if not frame_combinations:
        raise ValueError("No environment combinations available for scheduling.")
    if total_frames <= 0:
        return []

    if selection_mode == "round_robin":
        start = int((model_seed >> 12) % len(frame_combinations))
        ordered = frame_combinations[start:] + frame_combinations[:start]
        return [ordered[i % len(ordered)] for i in range(total_frames)]

    rng = np.random.default_rng((model_seed ^ 0xA5A5A5A5A5A5A5A5) & 0xFFFFFFFFFFFFFFFF)
    tie_breakers = rng.random(len(frame_combinations))
    level_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    pair_counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    scheduled: list[tuple[dict, dict, dict, str, float, float, str, int, int, int]] = []
    last_idx = -1

    factor_cache = [combo_factor_keys(combo) for combo in frame_combinations]
    pair_cache = [combo_pair_keys(combo) for combo in frame_combinations]

    for _ in range(total_frames):
        best_idx = 0
        best_score = float("-inf")
        for idx, combo in enumerate(frame_combinations):
            factors = factor_cache[idx]
            pairs = pair_cache[idx]
            pair_gain = sum(1.0 / (1.0 + pair_counts[pair]) for pair in pairs)
            factor_gain = sum(1.0 / (1.0 + level_counts[(f"group_{i}", value)]) for i, value in enumerate(factors))
            repeat_penalty = 1.0 if idx == last_idx else 0.0
            score = (3.4 * pair_gain) + factor_gain - (0.7 * repeat_penalty) + (1e-6 * tie_breakers[idx])
            if score > best_score:
                best_score = score
                best_idx = idx

        picked = frame_combinations[best_idx]
        scheduled.append(picked)
        for i, value in enumerate(factor_cache[best_idx]):
            level_counts[(f"group_{i}", value)] += 1
        for pair in pair_cache[best_idx]:
            pair_counts[pair] += 1
        last_idx = best_idx
    return scheduled


def sample_frame_variant(
    rng: np.random.Generator,
    mesh_extents: np.ndarray,
    render_config: RenderConfig,
    distance_lo: float,
    distance_hi: float,
) -> tuple[float, np.ndarray]:
    distance_scale = float(rng.uniform(distance_lo, distance_hi))
    if rng.random() < render_config.center_hold_probability:
        return distance_scale, np.zeros(3, dtype=np.float64)

    xy_raw = rng.uniform(-1.0, 1.0, size=2)
    xy_norm = float(np.linalg.norm(xy_raw))
    min_norm = 0.22
    if xy_norm < min_norm and render_config.offcenter_xy_scale > 1e-6:
        scale = min_norm / max(xy_norm, 1e-6)
        xy_raw *= scale

    xy_bias = np.clip(xy_raw, -1.0, 1.0) * render_config.offcenter_xy_scale
    z_bias = float(rng.uniform(-render_config.offcenter_z_scale, render_config.offcenter_z_scale))
    target_bias = np.array(
        [
            xy_bias[0] * mesh_extents[0],
            xy_bias[1] * mesh_extents[1],
            z_bias * mesh_extents[2],
        ],
        dtype=np.float64,
    )
    return distance_scale, target_bias


def maybe_apply_fake_clouds(
    color_buffer: np.ndarray,
    rng: np.random.Generator,
    cloud_probability: float,
    cloud_max_layers: int,
) -> np.ndarray:
    if cloud_probability <= 0.0 or cloud_max_layers <= 0:
        return color_buffer
    if float(rng.random()) > cloud_probability:
        return color_buffer

    base = Image.fromarray(color_buffer).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    width, height = base.size

    try:
        layer_count = int(rng.integers(1, cloud_max_layers + 1))
        for _ in range(layer_count):
            layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer, "RGBA")
            puff_count = int(rng.integers(2, 7))
            for _ in range(puff_count):
                cloud_w = int(rng.uniform(0.12, 0.45) * width)
                cloud_h = int(rng.uniform(0.06, 0.24) * height)
                x0 = int(rng.uniform(-0.2, 1.0) * width)
                y0 = int(rng.uniform(-0.2, 0.85) * height)
                x1 = x0 + max(cloud_w, 8)
                y1 = y0 + max(cloud_h, 6)
                tint = int(rng.integers(224, 256))
                alpha = int(rng.integers(22, 76))
                draw.ellipse((x0, y0, x1, y1), fill=(tint, tint, tint, alpha))

            blur_radius = float(rng.uniform(8.0, 28.0))
            blurred = layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            overlay = Image.alpha_composite(overlay, blurred)
            layer.close()
            blurred.close()

        haze_alpha = int(rng.integers(0, 28))
        if haze_alpha > 0:
            haze = Image.new("RGBA", base.size, (255, 255, 255, haze_alpha))
            overlay = Image.alpha_composite(overlay, haze)
            haze.close()

        merged = Image.alpha_composite(base, overlay).convert("RGB")
        output = np.array(merged, dtype=np.uint8)
        merged.close()
        return output
    finally:
        overlay.close()
        base.close()


def resolve_noise_settings(noise_profile: str) -> dict[str, float]:
    profile = str(noise_profile).strip().lower()
    settings = NOISE_PROFILE_SETTINGS.get(profile)
    if settings is None:
        choices = ", ".join(sorted(NOISE_PROFILE_SETTINGS))
        raise ValueError(f"Unsupported noise profile: {noise_profile}. Expected one of: {choices}")
    return settings


def _background_mask_from_depth(depth_buffer: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_buffer)
    return (~np.isfinite(depth)) | (depth <= 0.0)


def _build_airport_background(
    width: int,
    height: int,
    rng: np.random.Generator,
    settings: dict[str, float],
) -> np.ndarray:
    horizon = int(rng.uniform(0.38, 0.68) * height)
    horizon = int(np.clip(horizon, max(14, int(0.2 * height)), max(16, height - 16)))

    sky_top = np.array(
        [
            rng.uniform(86, 148),
            rng.uniform(126, 186),
            rng.uniform(154, 232),
        ],
        dtype=np.float32,
    )
    sky_horizon = np.array(
        [
            rng.uniform(164, 216),
            rng.uniform(172, 224),
            rng.uniform(182, 238),
        ],
        dtype=np.float32,
    )
    ground_far = np.array(
        [
            rng.uniform(108, 158),
            rng.uniform(104, 148),
            rng.uniform(92, 136),
        ],
        dtype=np.float32,
    )
    ground_near = np.array(
        [
            rng.uniform(58, 104),
            rng.uniform(58, 98),
            rng.uniform(58, 96),
        ],
        dtype=np.float32,
    )

    image = np.zeros((height, width, 3), dtype=np.float32)
    if horizon > 0:
        sky_grad = np.linspace(0.0, 1.0, num=horizon, dtype=np.float32)[:, None]
        image[:horizon, :, :] = ((1.0 - sky_grad) * sky_top)[:, None, :] + (sky_grad * sky_horizon)[:, None, :]
    if horizon < height:
        ground_rows = max(height - horizon, 1)
        ground_grad = np.linspace(0.0, 1.0, num=ground_rows, dtype=np.float32)[:, None]
        image[horizon:, :, :] = ((1.0 - ground_grad) * ground_far)[:, None, :] + (ground_grad * ground_near)[:, None, :]

    canvas = Image.fromarray(np.clip(image, 0.0, 255.0).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    structure_density = float(settings["structure_density"])

    runway_exists = rng.random() < 0.94
    if runway_exists and horizon < height - 4:
        top_y = int(horizon + rng.uniform(0.03, 0.2) * max(8, height - horizon))
        top_y = int(np.clip(top_y, horizon, height - 6))

        left_top = int(width * rng.uniform(0.40, 0.48))
        right_top = int(width * rng.uniform(0.52, 0.60))
        left_bottom = int(width * rng.uniform(0.06, 0.24))
        right_bottom = int(width * rng.uniform(0.76, 0.94))
        runway_color = (
            int(rng.integers(56, 92)),
            int(rng.integers(56, 92)),
            int(rng.integers(56, 94)),
        )
        draw.polygon(
            [(left_bottom, height), (left_top, top_y), (right_top, top_y), (right_bottom, height)],
            fill=runway_color,
        )

        stripe_segments = int(rng.integers(7, 14))
        for seg_idx in range(stripe_segments):
            t0 = seg_idx / max(stripe_segments, 1)
            t1 = min(1.0, t0 + rng.uniform(0.03, 0.09))
            y0 = int(height - (height - top_y) * (t0**1.15))
            y1 = int(height - (height - top_y) * (t1**1.15))
            y_top = min(y0, y1)
            y_bottom = max(y0, y1)
            if y_bottom - y_top < 1:
                continue
            width_scale = max(2, int((1.0 - t0) * width * 0.018))
            x_mid = int((left_bottom + right_bottom) * 0.5)
            stripe_color = (
                int(rng.integers(212, 248)),
                int(rng.integers(212, 248)),
                int(rng.integers(208, 242)),
            )
            draw.rectangle(
                [x_mid - width_scale, y_top, x_mid + width_scale, y_bottom],
                fill=stripe_color,
            )

        taxi_line_count = int(max(1, round(2 + (3 * structure_density))))
        for _ in range(taxi_line_count):
            x0 = int(rng.uniform(0.12, 0.42) * width)
            x1 = int(rng.uniform(0.58, 0.88) * width)
            y = int(rng.uniform(top_y, height) * 0.98)
            color = (
                int(rng.integers(168, 232)),
                int(rng.integers(152, 208)),
                int(rng.integers(78, 126)),
            )
            draw.line([(x0, y), (x1, y)], fill=color, width=max(1, int(width * 0.0022)))

    structure_count = int(max(3, round(6 + (28 * structure_density))))
    for _ in range(structure_count):
        struct_w = int(max(3, rng.uniform(0.01, 0.11) * width))
        struct_h = int(max(3, rng.uniform(0.015, 0.12) * height))
        x0 = int(rng.uniform(-0.02, 1.02) * width) - struct_w // 2
        y_base = int(horizon + rng.uniform(-0.04, 0.09) * height)
        y1 = int(np.clip(y_base, 0, height - 1))
        y0 = int(np.clip(y1 - struct_h, 0, height - 1))
        shade = int(rng.integers(34, 88))
        draw.rectangle([x0, y0, x0 + struct_w, y1], fill=(shade, shade, shade + int(rng.integers(0, 18))))

    pole_count = int(max(2, round(8 + (22 * structure_density))))
    for _ in range(pole_count):
        x = int(rng.uniform(0.0, 1.0) * width)
        y_top = int(horizon + rng.uniform(-0.06, 0.02) * height)
        y_bottom = int(horizon + rng.uniform(0.02, 0.12) * height)
        pole_shade = int(rng.integers(24, 70))
        draw.line([(x, y_top), (x, y_bottom)], fill=(pole_shade, pole_shade, pole_shade), width=max(1, int(width * 0.0016)))

    output = np.array(canvas, dtype=np.float32)
    canvas.close()

    yy = np.arange(height, dtype=np.float32)[:, None]
    fog_sigma = max(5.0, 0.13 * height)
    fog_band = np.exp(-((yy - float(horizon)) ** 2) / (2.0 * (fog_sigma**2)))
    fog_strength = float(settings["fog_strength"])
    output = output * (1.0 - (fog_strength * fog_band[..., None])) + (255.0 * fog_strength * fog_band[..., None])
    return np.clip(output, 0.0, 255.0).astype(np.uint8)


def _blend_background(
    rendered: np.ndarray,
    synthetic_bg: np.ndarray,
    background_mask: np.ndarray,
    blend_weight: float,
) -> np.ndarray:
    source = rendered.astype(np.float32)
    background = synthetic_bg.astype(np.float32)
    if np.any(background_mask):
        alpha = (background_mask.astype(np.float32) * float(blend_weight))[..., None]
        blended = (source * (1.0 - alpha)) + (background * alpha)
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)
    alpha = 0.14 * float(blend_weight)
    blended = (source * (1.0 - alpha)) + (background * alpha)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _apply_vignette(image: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return image
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    nx = (xx - cx) / max(cx, 1.0)
    ny = (yy - cy) / max(cy, 1.0)
    radius = np.sqrt(nx**2 + ny**2)
    vignette = 1.0 - (strength * np.clip(radius, 0.0, 1.6) ** 1.65)
    out = image.astype(np.float32) * vignette[..., None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _apply_foreground_overlay(
    image: np.ndarray,
    rng: np.random.Generator,
    alpha_cap: float,
) -> np.ndarray:
    if alpha_cap <= 0.0:
        return image

    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    width, height = base.size
    draw = ImageDraw.Draw(overlay, "RGBA")

    streaks = int(rng.integers(2, 6))
    for _ in range(streaks):
        x0 = int(rng.uniform(-0.2, 1.2) * width)
        y0 = int(rng.uniform(0.0, 1.0) * height)
        x1 = int(rng.uniform(-0.2, 1.2) * width)
        y1 = int(y0 + rng.uniform(-0.18, 0.18) * height)
        alpha = int(min(255, max(0, rng.uniform(0.12, 0.45) * alpha_cap)))
        tone = int(rng.integers(120, 220))
        draw.line([(x0, y0), (x1, y1)], fill=(tone, tone, tone, alpha), width=max(1, int(width * rng.uniform(0.002, 0.007))))

    speck_count = int(rng.integers(20, 72))
    for _ in range(speck_count):
        cx = int(rng.uniform(0.0, 1.0) * width)
        cy = int(rng.uniform(0.0, 1.0) * height)
        r = int(max(1, rng.uniform(0.0015, 0.008) * min(width, height)))
        alpha = int(min(255, max(0, rng.uniform(0.08, 0.35) * alpha_cap)))
        tone = int(rng.integers(94, 236))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(tone, tone, tone, alpha))

    blur_radius = float(rng.uniform(0.6, 2.2))
    soft_overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    try:
        merged = Image.alpha_composite(base, soft_overlay).convert("RGB")
        output = np.array(merged, dtype=np.uint8)
        merged.close()
        return output
    finally:
        soft_overlay.close()
        overlay.close()
        base.close()


def _apply_codec_artifacts(
    image: np.ndarray,
    rng: np.random.Generator,
    settings: dict[str, float],
) -> np.ndarray:
    if float(rng.random()) > float(settings["compression_prob"]):
        return image
    quality = int(
        rng.uniform(
            float(settings["compression_quality_min"]),
            float(settings["compression_quality_max"]),
        )
    )
    quality = int(np.clip(quality, 20, 95))
    buffer = io.BytesIO()
    source = Image.fromarray(image)
    decoded = None
    try:
        source.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
        buffer.seek(0)
        decoded = Image.open(buffer).convert("RGB")
        return np.array(decoded, dtype=np.uint8)
    finally:
        if decoded is not None:
            decoded.close()
        source.close()
        buffer.close()


def apply_background_and_noise(
    color_buffer: np.ndarray,
    depth_buffer: np.ndarray,
    rng: np.random.Generator,
    render_config: RenderConfig,
) -> np.ndarray:
    settings = resolve_noise_settings(render_config.noise_profile)
    profile = str(render_config.background_scene_profile).strip().lower()
    if profile not in BACKGROUND_SCENE_PROFILES:
        choices = ", ".join(BACKGROUND_SCENE_PROFILES)
        raise ValueError(f"Unsupported background scene profile: {render_config.background_scene_profile}. Expected one of: {choices}")

    height, width = color_buffer.shape[:2]
    background_mask = _background_mask_from_depth(depth_buffer)
    noise_scale = float(
        rng.uniform(
            float(settings["noise_scale_min"]),
            float(settings["noise_scale_max"]),
        )
    )
    airport_visible_probability = float(np.clip(settings["airport_visible_probability"], 0.0, 1.0))
    airport_visible = bool(float(rng.random()) <= airport_visible_probability)

    output = color_buffer.copy()
    if airport_visible:
        synthetic_bg = _build_airport_background(width=width, height=height, rng=rng, settings=settings)
        blend_min = float(settings["background_replace_weight_min"])
        blend_max = float(settings["background_replace_weight_max"])
        sampled_blend = float(rng.uniform(blend_min, blend_max))
        blend_weight = float(np.clip(sampled_blend * (0.82 + (0.28 * noise_scale)), 0.0, 1.0))
        output = _blend_background(
            rendered=color_buffer,
            synthetic_bg=synthetic_bg,
            background_mask=background_mask,
            blend_weight=blend_weight,
        )

    cloud_prob_scale = float(np.clip(0.48 + (0.52 * noise_scale) + rng.uniform(-0.16, 0.22), 0.15, 1.35))
    cloud_probability = float(np.clip(render_config.cloud_probability * cloud_prob_scale, 0.0, 1.0))
    output = maybe_apply_fake_clouds(
        color_buffer=output,
        rng=rng,
        cloud_probability=cloud_probability,
        cloud_max_layers=render_config.cloud_max_layers,
    )

    output_f = output.astype(np.float32)
    bg_sigma = float(settings["bg_grain_sigma"]) * noise_scale
    fg_sigma = float(settings["fg_grain_sigma"]) * (0.72 + (0.38 * noise_scale))
    bg_noise = rng.normal(0.0, bg_sigma, size=output_f.shape).astype(np.float32)
    fg_noise = rng.normal(0.0, fg_sigma, size=output_f.shape).astype(np.float32)
    if np.any(background_mask):
        mask = background_mask[..., None].astype(np.float32)
        output_f = output_f + (mask * bg_noise) + ((1.0 - mask) * fg_noise)
    else:
        output_f = output_f + (0.65 * bg_noise)

    color_drift = float(settings["color_drift"]) * (0.70 + (0.45 * noise_scale))
    drift_factors = np.array(
        [
            rng.uniform(1.0 - color_drift, 1.0 + color_drift),
            rng.uniform(1.0 - color_drift, 1.0 + color_drift),
            rng.uniform(1.0 - color_drift, 1.0 + color_drift),
        ],
        dtype=np.float32,
    )
    output_f = output_f * drift_factors[None, None, :]
    output = np.clip(output_f, 0.0, 255.0).astype(np.uint8)

    artifact_prob = float(np.clip(float(settings["artifact_prob"]) * (0.65 + (0.55 * noise_scale)), 0.0, 1.0))
    artifact_strength = float(settings["artifact_strength"]) * (0.72 + (0.52 * noise_scale))
    if float(rng.random()) < artifact_prob:
        base = Image.fromarray(output).convert("RGB")
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        line_count = int(rng.integers(5, 22))
        for _ in range(line_count):
            y = int(rng.uniform(0.0, 1.0) * height)
            alpha = int(min(255, 255.0 * artifact_strength * rng.uniform(0.05, 0.22)))
            tone = int(rng.integers(90, 210))
            draw.line([(0, y), (width, y)], fill=(tone, tone, tone, alpha), width=max(1, int(height * 0.0014)))
        merged = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        output = np.array(merged, dtype=np.uint8)
        merged.close()
        overlay.close()
        base.close()

    foreground_alpha = float(settings["foreground_overlay_alpha"]) * (0.70 + (0.55 * noise_scale))
    output = _apply_foreground_overlay(output, rng=rng, alpha_cap=foreground_alpha)
    blur_max = float(settings["blur_max"]) * (0.75 + (0.55 * noise_scale))
    blur_radius = float(rng.uniform(0.0, blur_max))
    if blur_radius > 0.08:
        blurred = Image.fromarray(output).filter(ImageFilter.GaussianBlur(radius=blur_radius))
        output = np.array(blurred, dtype=np.uint8)
        blurred.close()
    codec_settings = dict(settings)
    codec_settings["compression_prob"] = float(
        np.clip(float(settings["compression_prob"]) * (0.55 + (0.55 * noise_scale)), 0.0, 1.0)
    )
    quality_shift = (noise_scale - 1.0) * 15.0
    codec_settings["compression_quality_min"] = float(
        np.clip(float(settings["compression_quality_min"]) - quality_shift, 20.0, 95.0)
    )
    codec_settings["compression_quality_max"] = float(
        np.clip(float(settings["compression_quality_max"]) - (0.7 * quality_shift), 20.0, 95.0)
    )
    if codec_settings["compression_quality_max"] < codec_settings["compression_quality_min"]:
        codec_settings["compression_quality_max"] = codec_settings["compression_quality_min"]
    output = _apply_codec_artifacts(output, rng=rng, settings=codec_settings)
    vignette_strength = float(settings["vignette_strength"]) * (0.72 + (0.52 * noise_scale))
    output = _apply_vignette(output, strength=vignette_strength)
    return output


def release_scene(scene: pyrender.Scene | None) -> None:
    if scene is None:
        return
    for node in list(scene.get_nodes()):
        try:
            scene.remove_node(node)
        except Exception:
            pass


def safe_remove_node(scene: pyrender.Scene | None, node: pyrender.Node | None) -> None:
    if scene is None or node is None:
        return
    try:
        scene.remove_node(node)
    except Exception:
        pass


def build_scene(mesh_path: Path, force_untextured: bool = False):
    check_missing_textures(mesh_path)
    loaded = load_mesh_asset(mesh_path)
    renderable = to_renderable_meshes(loaded)
    mesh_center, mesh_extents, mesh_diagonal = compute_bounds(renderable)

    scene = pyrender.Scene(bg_color=SKY_PRESETS[0]["bg_color"], ambient_light=SKY_PRESETS[0]["ambient"])
    scene.add(build_pyrender_mesh(mesh_path, renderable, force_untextured=force_untextured))

    del renderable
    del loaded

    camera_yfov = np.pi / 3.0
    object_radius = max(mesh_diagonal * 0.5, 1e-3)
    fit_distance = object_radius / math.sin(camera_yfov * 0.5)
    base_radius = max(FIT_MARGIN * fit_distance, np.max(mesh_extents) * EXTENT_MARGIN)

    return scene, mesh_center, mesh_extents, base_radius, camera_yfov


def is_ctypes_array_handler_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "no array-type handler" in msg and "_ctypes.type" in msg


def is_eigenvalue_convergence_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "eigenvalues did not converge" in msg


def is_finite_pose(pose: np.ndarray | None) -> bool:
    return pose is not None and np.asarray(pose).shape == (4, 4) and np.all(np.isfinite(pose))


def compute_camera_pose(
    mesh_center: np.ndarray,
    mesh_extents: np.ndarray,
    base_radius: float,
    azimuth_deg: int,
    elevation_deg: int,
    roll_deg: float,
    phase: float | None,
    distance_scale: float = 1.0,
    target_bias: np.ndarray | None = None,
):
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)

    if phase is None:
        dynamic_target = mesh_center
        dynamic_radius = base_radius
    else:
        target_offset = np.array(
            [
                0.04 * mesh_extents[0] * math.sin(phase),
                0.04 * mesh_extents[1] * math.cos(1.37 * phase),
                0.025 * mesh_extents[2] * math.sin(0.73 * phase),
            ]
        )
        dynamic_target = mesh_center + target_offset
        dynamic_radius = base_radius * (1.0 + 0.06 * math.sin(1.17 * phase))

    if target_bias is not None:
        dynamic_target = dynamic_target + target_bias
    dynamic_radius = max(base_radius * 0.55, dynamic_radius * max(distance_scale, 0.35))

    x = dynamic_radius * math.cos(elevation) * math.cos(azimuth)
    y = dynamic_radius * math.cos(elevation) * math.sin(azimuth)
    z = dynamic_radius * math.sin(elevation)

    cam_pos = dynamic_target + np.array([x, y, z])
    return look_at(cam_pos, target=dynamic_target, roll_deg=roll_deg)


def select_environment_variant(frame_index: int) -> tuple[dict, dict]:
    sky = SKY_PRESETS[frame_index % len(SKY_PRESETS)]
    light = LIGHT_POSITION_PRESETS[(frame_index // len(SKY_PRESETS)) % len(LIGHT_POSITION_PRESETS)]
    return sky, light


def apply_environment_lighting(
    scene: pyrender.Scene,
    mesh_center: np.ndarray,
    base_radius: float,
    sky: dict,
    light: dict,
    light_style: dict | None = None,
    dir_node: pyrender.Node | None = None,
    fill_node: pyrender.Node | None = None,
) -> tuple[pyrender.Node, pyrender.Node]:
    scene.bg_color = sky["bg_color"]

    ambient_scale = float(light_style["ambient_scale"]) if light_style is not None else 1.0
    dir_scale = float(light_style["dir_scale"]) if light_style is not None else 1.0
    fill_scale = float(light_style["fill_scale"]) if light_style is not None else 1.0

    ambient = np.clip(np.array(sky["ambient"], dtype=np.float64) * ambient_scale, 0.0, 1.0)
    scene.ambient_light = ambient.tolist()

    dir_offset = np.array(light["dir_offset"], dtype=np.float64) * base_radius
    fill_offset = np.array(light["fill_offset"], dtype=np.float64) * base_radius
    dir_pos = mesh_center + dir_offset
    fill_pos = mesh_center + fill_offset

    dir_pose = look_at(dir_pos, target=mesh_center)
    fill_pose = np.eye(4)
    fill_pose[:3, 3] = fill_pos

    if dir_node is None:
        dir_node = scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=float(sky["dir_intensity"]) * dir_scale),
            pose=dir_pose,
        )
    else:
        scene.set_pose(dir_node, pose=dir_pose)
        dir_light = dir_node.light
        if isinstance(dir_light, pyrender.DirectionalLight):
            dir_light.intensity = float(sky["dir_intensity"]) * dir_scale

    if fill_node is None:
        fill_node = scene.add(
            pyrender.PointLight(color=np.ones(3), intensity=float(sky["fill_intensity"]) * fill_scale),
            pose=fill_pose,
        )
    else:
        scene.set_pose(fill_node, pose=fill_pose)
        fill_light = fill_node.light
        if isinstance(fill_light, pyrender.PointLight):
            fill_light.intensity = float(sky["fill_intensity"]) * fill_scale

    return dir_node, fill_node


def mesh_signature(mesh_path: Path) -> str:
    resolved = mesh_path.resolve()
    stat = resolved.stat()
    return f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}"


def approval_record_path(approvals_root: Path, model_key: str) -> Path:
    return approvals_root / "records" / f"{slugify(model_key)}.json"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_review_sheet(images: list[Path], sheet_path: Path, columns: int = 4, thumb_size: int = 280) -> None:
    if not images:
        raise ValueError("Cannot build a review sheet with no images.")

    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * thumb_size, rows * thumb_size), color=(245, 245, 245))

    for index, image_path in enumerate(images):
        row = index // columns
        col = index % columns
        with Image.open(image_path) as src:
            rgb = src.convert("RGB")
            resized = rgb.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            canvas.paste(resized, (col * thumb_size, row * thumb_size))

    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(sheet_path)


def render_review_pack(
    mesh_path: Path,
    render_root: Path,
    approvals_root: Path,
    preview_size: int = 512,
    gc_every_frames: int = 40,
) -> Path:
    model_key = model_key_from_mesh_path(mesh_path, render_root)
    preview_dir = approvals_root / "previews" / slugify(model_key)
    preview_dir.mkdir(parents=True, exist_ok=True)

    for old_file in preview_dir.glob("review_*.png"):
        old_file.unlink(missing_ok=True)

    scene: pyrender.Scene | None = None
    camera = None
    renderer: pyrender.OffscreenRenderer | None = None
    camera_node: pyrender.Node | None = None
    dir_node: pyrender.Node | None = None
    fill_node: pyrender.Node | None = None

    rendered: list[Path] = []
    for force_untextured in (False, True):
        try:
            scene, mesh_center, mesh_extents, base_radius, camera_yfov = build_scene(
                mesh_path,
                force_untextured=force_untextured,
            )
            camera = pyrender.PerspectiveCamera(yfov=camera_yfov)
            renderer = pyrender.OffscreenRenderer(viewport_width=preview_size, viewport_height=preview_size)
            camera_node = scene.add(camera, pose=np.eye(4))

            rendered.clear()
            for index, (label, azimuth_deg, elevation_deg, roll_deg) in enumerate(REVIEW_VIEW_SPECS):
                sky, light = select_environment_variant(index)
                dir_node, fill_node = apply_environment_lighting(
                    scene=scene,
                    mesh_center=mesh_center,
                    base_radius=base_radius,
                    sky=sky,
                    light=light,
                    dir_node=dir_node,
                    fill_node=fill_node,
                )
                cam_pose = compute_camera_pose(
                    mesh_center=mesh_center,
                    mesh_extents=mesh_extents,
                    base_radius=base_radius,
                    azimuth_deg=azimuth_deg,
                    elevation_deg=elevation_deg,
                    roll_deg=roll_deg,
                    phase=None,
                )
                scene.set_pose(camera_node, pose=cam_pose)
                color, _ = renderer.render(scene)

                out_file = preview_dir / f"review_{index:02d}_{label}_{sky['name']}_{light['name']}.png"
                save_png_array(color, out_file)
                rendered.append(out_file)
                del color
                if gc_every_frames > 0 and (index + 1) % gc_every_frames == 0:
                    gc.collect()
            break
        except Exception as exc:
            if force_untextured or not is_ctypes_array_handler_error(exc):
                raise
            for file_path in rendered:
                file_path.unlink(missing_ok=True)
            rendered.clear()
            print(f"[warn] {mesh_path.name}: texture/OpenGL path failed during preview; retrying without textures.")
        finally:
            safe_remove_node(scene, camera_node)
            safe_remove_node(scene, dir_node)
            safe_remove_node(scene, fill_node)
            if renderer is not None:
                renderer.delete()
                del renderer
            release_scene(scene)
            del scene
            del camera
            gc.collect()

    sheet_path = preview_dir / "review_sheet.png"
    save_review_sheet(rendered, sheet_path)
    return sheet_path


def prompt_for_approval(model_key: str, mesh_path: Path, sheet_path: Path) -> tuple[str, str]:
    print(f"[review] model={model_key}")
    print(f"[review] mesh={mesh_path}")
    print(f"[review] preview={sheet_path}")

    if not sys.stdin.isatty():
        return "pending", "Interactive approval required (non-interactive shell)."

    while True:
        answer = input("Approve model? [a]pprove / [r]eject / [s]kip: ").strip().lower()
        if answer in {"a", "approve"}:
            note = input("Optional approval note (Enter to skip): ").strip()
            return "approved", note
        if answer in {"r", "reject"}:
            reason = input("Reason for rejection (optional): ").strip()
            return "rejected", reason
        if answer in {"s", "skip"}:
            return "pending", "Skipped during manual review."
        print("Please enter a, r, or s.")


def check_or_update_approval(
    mesh_path: Path,
    render_root: Path,
    approvals_root: Path,
    approval_mode: str,
    force_review: bool,
    auto_approve: bool,
    preview_size: int,
    gc_every_frames: int,
) -> tuple[str, Path | None]:
    model_key = model_key_from_mesh_path(mesh_path, render_root)
    record_path = approval_record_path(approvals_root, model_key)
    current_signature = mesh_signature(mesh_path)

    record = load_json(record_path) or {}
    status = str(record.get("status", "pending"))
    is_signature_current = record.get("mesh_signature") == current_signature
    has_valid_approval = status == "approved" and is_signature_current and not force_review

    if approval_mode == "off":
        return "approved", None

    if has_valid_approval:
        return "approved", Path(record["preview_sheet"]) if record.get("preview_sheet") else None

    if approval_mode == "check":
        return "pending", Path(record["preview_sheet"]) if record.get("preview_sheet") else None

    sheet_path = render_review_pack(
        mesh_path,
        render_root,
        approvals_root,
        preview_size=preview_size,
        gc_every_frames=gc_every_frames,
    )
    if auto_approve:
        status, note = "approved", "Auto-approved by CLI flag."
    else:
        status, note = prompt_for_approval(model_key, mesh_path, sheet_path)

    payload = {
        "model_key": model_key,
        "mesh_path": str(mesh_path),
        "mesh_signature": current_signature,
        "status": status,
        "note": note,
        "preview_sheet": str(sheet_path),
        "updated_at": utc_now_iso(),
    }
    write_json(record_path, payload)
    return status, sheet_path


def bulk_auto_approve(
    mesh_paths: list[Path],
    render_root: Path,
    approvals_root: Path,
    force_review: bool,
) -> tuple[int, int]:
    newly_approved = 0
    already_approved = 0

    for mesh_path in mesh_paths:
        model_key = model_key_from_mesh_path(mesh_path, render_root)
        record_path = approval_record_path(approvals_root, model_key)
        current_signature = mesh_signature(mesh_path)

        record = load_json(record_path) or {}
        is_signature_current = record.get("mesh_signature") == current_signature
        has_valid_approval = record.get("status") == "approved" and is_signature_current

        if has_valid_approval and not force_review:
            already_approved += 1
            continue

        payload = {
            "model_key": model_key,
            "mesh_path": str(mesh_path),
            "mesh_signature": current_signature,
            "status": "approved",
            "note": "Bulk auto-approved before render run.",
            "preview_sheet": record.get("preview_sheet"),
            "updated_at": utc_now_iso(),
        }
        write_json(record_path, payload)
        newly_approved += 1

    return newly_approved, already_approved


def write_render_manifest(
    manifest_path: Path,
    mesh_path: Path,
    output_dir: Path,
    frame_count: int,
    removed_existing: int,
    render_config: RenderConfig,
    resolution_counts: dict[str, int],
) -> None:
    pose_count = len(build_pose_grid(render_config))
    combinations_per_pose = len(build_frame_combinations(render_config))
    resolution_options = [
        {"name": name, "width": width, "height": height}
        for name, width, height in render_config.resolution_options
    ]
    payload = {
        "mesh_path": str(mesh_path),
        "output_dir": str(output_dir),
        "frame_count": frame_count,
        "removed_existing": removed_existing,
        "render_size": render_config.render_size,
        "resolution_profile": render_config.resolution_profile,
        "resolution_options": resolution_options,
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "azimuth_step_deg": render_config.azimuth_step_deg,
        "elevation_min_deg": render_config.elevation_min_deg,
        "elevation_max_deg": render_config.elevation_max_deg,
        "elevation_step_deg": render_config.elevation_step_deg,
        "orientation_modes": [mode for mode, _ in render_config.orientation_modes],
        "distance_min_scale": render_config.distance_min_scale,
        "distance_max_scale": render_config.distance_max_scale,
        "offcenter_xy_scale": render_config.offcenter_xy_scale,
        "offcenter_z_scale": render_config.offcenter_z_scale,
        "center_hold_probability": render_config.center_hold_probability,
        "cloud_probability": render_config.cloud_probability,
        "cloud_max_layers": render_config.cloud_max_layers,
        "frame_seed": render_config.frame_seed,
        "output_format": render_config.output_format,
        "output_quality": render_config.output_quality,
        "noise_profile": render_config.noise_profile,
        "noise_profile_settings": resolve_noise_settings(render_config.noise_profile),
        "background_scene_profile": render_config.background_scene_profile,
        "combination_strategy": render_config.combination_strategy,
        "variants_per_combo": render_config.variants_per_combo,
        "distance_bins": render_config.distance_bins,
        "pose_count": pose_count,
        "combinations_per_pose": combinations_per_pose,
        "target_images_per_model": render_config.target_images_per_model,
        "combo_selection": render_config.combo_selection,
        "frame_count_is_successful_writes": True,
        "updated_at": utc_now_iso(),
    }
    write_json(manifest_path, payload)


def estimate_frame_counts(render_config: RenderConfig, model_count: int) -> dict[str, int]:
    pose_count = len(build_pose_grid(render_config))
    combination_count = len(build_frame_combinations(render_config))
    all_combos_per_model = int(pose_count * combination_count)
    if render_config.target_images_per_model > 0:
        planned_per_model = int(min(render_config.target_images_per_model, all_combos_per_model))
    else:
        planned_per_model = all_combos_per_model
    models = int(max(0, model_count))
    return {
        "pose_count": int(pose_count),
        "combination_count": int(combination_count),
        "all_combos_per_model": all_combos_per_model,
        "planned_per_model": planned_per_model,
        "all_combos_total": int(all_combos_per_model * models),
        "planned_total": int(planned_per_model * models),
    }


def render_one_mesh(
    mesh_path: Path,
    render_root: Path,
    images_root: Path,
    images_index: dict[str, Path],
    render_config: RenderConfig,
    clean_output: bool,
) -> int:
    output_dir = resolve_output_dir(mesh_path, render_root, images_root, images_index)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = slugify(mesh_path.stem)
    prefix = f"render_{safe_stem}_"
    removed_existing = clear_existing_renders(output_dir, prefix) if clean_output else 0
    extension = output_extension(render_config.output_format)
    model_seed = stable_seed(mesh_signature(mesh_path), render_config.frame_seed)

    poses = build_pose_grid(render_config)
    if not poses:
        raise ValueError("No camera poses generated. Check angle/orientation settings.")
    frame_combinations = build_frame_combinations(render_config)
    if not frame_combinations:
        raise ValueError("No frame combinations were generated. Check combination settings.")
    max_frames = len(poses) * len(frame_combinations)
    if render_config.target_images_per_model > 0:
        total_frames = min(render_config.target_images_per_model, max_frames)
    else:
        total_frames = max_frames
    pose_schedule = build_pose_schedule(poses=poses, total_frames=total_frames, model_seed=model_seed)
    combo_schedule = build_combo_schedule(
        frame_combinations=frame_combinations,
        total_frames=total_frames,
        selection_mode=render_config.combo_selection,
        model_seed=model_seed,
    )
    if len(pose_schedule) != len(combo_schedule):
        raise RuntimeError("Pose/combo scheduling mismatch.")

    frame_idx = 0
    generated_paths: list[Path] = []
    skipped_render_frames = 0
    resolution_counts: defaultdict[str, int] = defaultdict(int)
    for force_untextured in (False, True):
        scene: pyrender.Scene | None = None
        camera = None
        renderer: pyrender.OffscreenRenderer | None = None
        camera_node: pyrender.Node | None = None
        dir_node: pyrender.Node | None = None
        fill_node: pyrender.Node | None = None

        frame_idx = 0
        generated_paths.clear()
        resolution_counts.clear()
        try:
            scene, mesh_center, mesh_extents, base_radius, camera_yfov = build_scene(
                mesh_path,
                force_untextured=force_untextured,
            )
            camera = pyrender.PerspectiveCamera(yfov=camera_yfov)
            camera_node = scene.add(camera, pose=np.eye(4))
            if combo_schedule:
                _, _, _, _, _, _, _, initial_width, initial_height, _ = combo_schedule[0]
            else:
                initial_width = render_config.render_size
                initial_height = render_config.render_size
            renderer = pyrender.OffscreenRenderer(
                viewport_width=int(initial_width),
                viewport_height=int(initial_height),
            )

            for frame_idx, ((orientation_name, roll_deg, azimuth_deg, elevation_deg), combo) in enumerate(
                zip(pose_schedule, combo_schedule)
            ):
                sky, light, light_style, distance_name, dist_min, dist_max, resolution_name, width, height, variant_idx = combo
                rng = frame_rng(model_seed=model_seed, frame_index=frame_idx)
                phase = (2.0 * math.pi * frame_idx) / max(total_frames - 1, 1)
                distance_scale, target_bias = sample_frame_variant(
                    rng=rng,
                    mesh_extents=mesh_extents,
                    render_config=render_config,
                    distance_lo=dist_min,
                    distance_hi=dist_max,
                )
                dir_node, fill_node = apply_environment_lighting(
                    scene=scene,
                    mesh_center=mesh_center,
                    base_radius=base_radius,
                    sky=sky,
                    light=light,
                    light_style=light_style,
                    dir_node=dir_node,
                    fill_node=fill_node,
                )
                cam_pose = compute_camera_pose(
                    mesh_center=mesh_center,
                    mesh_extents=mesh_extents,
                    base_radius=base_radius,
                    azimuth_deg=azimuth_deg,
                    elevation_deg=elevation_deg,
                    roll_deg=roll_deg,
                    phase=phase,
                    distance_scale=distance_scale,
                    target_bias=target_bias,
                )

                if not is_finite_pose(cam_pose):
                    skipped_render_frames += 1
                    continue

                scene.set_pose(camera_node, pose=cam_pose)
                renderer.viewport_width = int(width)
                renderer.viewport_height = int(height)
                try:
                    color, depth = renderer.render(scene)
                except Exception as exc:
                    if not is_eigenvalue_convergence_error(exc):
                        raise

                    # Retry once with a numerically safe baseline lighting/camera setup.
                    safe_sky = SKY_PRESETS[0]
                    safe_light = LIGHT_POSITION_PRESETS[0]
                    safe_style = LIGHT_STYLE_PRESETS[1] if len(LIGHT_STYLE_PRESETS) > 1 else LIGHT_STYLE_PRESETS[0]
                    dir_node, fill_node = apply_environment_lighting(
                        scene=scene,
                        mesh_center=mesh_center,
                        base_radius=base_radius,
                        sky=safe_sky,
                        light=safe_light,
                        light_style=safe_style,
                        dir_node=dir_node,
                        fill_node=fill_node,
                    )
                    safe_pose = compute_camera_pose(
                        mesh_center=mesh_center,
                        mesh_extents=mesh_extents,
                        base_radius=base_radius,
                        azimuth_deg=azimuth_deg,
                        elevation_deg=elevation_deg,
                        roll_deg=roll_deg,
                        phase=None,
                        distance_scale=1.0,
                        target_bias=np.zeros(3, dtype=np.float64),
                    )
                    if not is_finite_pose(safe_pose):
                        skipped_render_frames += 1
                        continue

                    scene.set_pose(camera_node, pose=safe_pose)
                    try:
                        color, depth = renderer.render(scene)
                    except Exception as retry_exc:
                        if not is_eigenvalue_convergence_error(retry_exc):
                            raise
                        skipped_render_frames += 1
                        continue

                color = apply_background_and_noise(
                    color_buffer=color,
                    depth_buffer=depth,
                    rng=rng,
                    render_config=render_config,
                )

                filename = (
                    f"{prefix}{orientation_name}_"
                    f"{sky['name']}_{light['name']}_{light_style['name']}_{distance_name}_{resolution_name}_"
                    f"v{variant_idx:02d}_"
                    f"az{azimuth_deg:03d}_el{elevation_deg:+03d}_{frame_idx:05d}{extension}"
                )
                out_path = output_dir / filename
                save_render_array(
                    color_buffer=color,
                    output_path=out_path,
                    output_format=render_config.output_format,
                    output_quality=render_config.output_quality,
                )
                resolution_counts[f"{int(width)}x{int(height)}"] += 1
                generated_paths.append(out_path)
                del color
                del depth
                if render_config.gc_every_frames > 0 and (frame_idx + 1) % render_config.gc_every_frames == 0:
                    gc.collect()
            frame_idx = len(generated_paths)
            break
        except Exception as exc:
            if force_untextured or not is_ctypes_array_handler_error(exc):
                raise
            for file_path in generated_paths:
                file_path.unlink(missing_ok=True)
            generated_paths.clear()
            print(f"[warn] {mesh_path.name}: texture/OpenGL path failed; retrying without textures.")
        finally:
            safe_remove_node(scene, camera_node)
            safe_remove_node(scene, dir_node)
            safe_remove_node(scene, fill_node)
            if renderer is not None:
                renderer.delete()
            release_scene(scene)
            del scene
            del camera
            gc.collect()

    manifest_path = output_dir / f"{safe_stem}_render_manifest.json"
    write_render_manifest(
        manifest_path=manifest_path,
        mesh_path=mesh_path,
        output_dir=output_dir,
        frame_count=frame_idx,
        removed_existing=removed_existing,
        render_config=render_config,
        resolution_counts=dict(resolution_counts),
    )

    print(
        f"[ok] {mesh_path.name}: wrote {frame_idx} images to {output_dir} "
        f"(removed_existing={removed_existing}, skipped_frames={skipped_render_frames})"
    )
    return frame_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render dense aircraft photo datasets from mesh files, with optional manual approval gating."
    )
    parser.add_argument("--render-root", type=str, default="data/renders", help="Folder containing mesh models.")
    parser.add_argument("--images-root", type=str, default="data/images", help="Output folder for training image sets.")
    parser.add_argument(
        "--approvals-root",
        type=str,
        default="data/model_approvals",
        help="Folder where approval records and review sheets are stored.",
    )
    parser.add_argument("--render-size", type=int, default=640, help="Render side length in pixels.")
    parser.add_argument(
        "--resolution-profile",
        type=str,
        default="balanced_mix",
        choices=tuple(sorted(RESOLUTION_PROFILES.keys())),
        help="Render resolution schedule profile; fixed uses --render-size, others include non-square frames.",
    )
    parser.add_argument(
        "--noise-profile",
        type=str,
        default="aggressive_background",
        choices=tuple(sorted(NOISE_PROFILE_SETTINGS.keys())),
        help="Noise strength profile applied after rendering.",
    )
    parser.add_argument(
        "--background-scene-profile",
        type=str,
        default="airport_heavy",
        choices=BACKGROUND_SCENE_PROFILES,
        help="Synthetic background scene mix used during post-processing.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="webp",
        help="Image format for training renders: png|jpeg|jpg|webp.",
    )
    parser.add_argument(
        "--output-quality",
        type=int,
        default=78,
        help="Compression quality for jpeg/webp (1-100). Ignored for png.",
    )
    parser.add_argument(
        "--preview-size",
        type=int,
        default=512,
        help="Preview render side length for manual approval sheets.",
    )
    parser.add_argument("--azimuth-step", type=int, default=20, help="Azimuth angle step in degrees.")
    parser.add_argument("--elevation-min", type=int, default=-30, help="Minimum elevation in degrees.")
    parser.add_argument("--elevation-max", type=int, default=30, help="Maximum elevation in degrees.")
    parser.add_argument("--elevation-step", type=int, default=15, help="Elevation angle step in degrees.")
    parser.add_argument(
        "--approval-mode",
        type=str,
        choices=("off", "check", "interactive", "only"),
        default="interactive",
        help=(
            "off=render everything, check=render only previously approved models, "
            "interactive=prompt for manual approval, only=run approval prompts without rendering"
        ),
    )
    parser.add_argument("--force-review", action="store_true", help="Force fresh review even if already approved.")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve models in interactive/only mode.")
    parser.add_argument(
        "--auto-approve-all-first",
        action="store_true",
        help="Approve every discovered model at startup before per-model processing.",
    )
    parser.add_argument("--clean-output", action="store_true", help="Delete previous renders for each mesh before writing.")
    parser.add_argument("--limit-models", type=int, default=0, help="Process only the first N meshes (0 means all).")
    parser.add_argument(
        "--combination-strategy",
        type=str,
        choices=("exhaustive", "cyclic"),
        default="exhaustive",
        help="exhaustive=full sky/light/style/distance cross-product per camera pose; cyclic=lighter cycling pass.",
    )
    parser.add_argument(
        "--combo-selection",
        type=str,
        choices=("pairwise", "round_robin"),
        default="pairwise",
        help="pairwise=greedy high-coverage sampling across factor pairs; round_robin=simple deterministic cycling.",
    )
    parser.add_argument(
        "--target-images-per-model",
        type=int,
        default=800,
        help="Target render count per model (0 means full pose x combination expansion).",
    )
    parser.add_argument(
        "--variants-per-combo",
        type=int,
        default=1,
        help="Additional jittered samples per environment combination.",
    )
    parser.add_argument(
        "--distance-bins",
        type=int,
        default=3,
        help="How many distance bands to render (near/mid/far...).",
    )
    parser.add_argument(
        "--distance-min-scale",
        type=float,
        default=0.55,
        help="Minimum camera distance multiplier relative to base fit distance.",
    )
    parser.add_argument(
        "--distance-max-scale",
        type=float,
        default=1.95,
        help="Maximum camera distance multiplier relative to base fit distance.",
    )
    parser.add_argument(
        "--offcenter-xy-scale",
        type=float,
        default=0.34,
        help="Max horizontal/vertical framing offset as a fraction of mesh extents.",
    )
    parser.add_argument(
        "--offcenter-z-scale",
        type=float,
        default=0.20,
        help="Max depth-axis framing offset as a fraction of mesh extents.",
    )
    parser.add_argument(
        "--center-hold-probability",
        type=float,
        default=0.22,
        help="Chance that a frame keeps centered framing (0-1).",
    )
    parser.add_argument(
        "--cloud-probability",
        type=float,
        default=0.55,
        help="Chance to inject synthetic cloud noise into each frame (0-1).",
    )
    parser.add_argument(
        "--cloud-max-layers",
        type=int,
        default=3,
        help="Maximum fake cloud overlay layers per frame (0 disables clouds).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=24,
        help="Base seed for deterministic random frame variations.",
    )
    parser.add_argument(
        "--gc-every-frames",
        type=int,
        default=40,
        help="Run Python garbage collection every N rendered frames (0 disables periodic GC).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    render_root = resolve_path(args.render_root, project_root)
    images_root = resolve_path(args.images_root, project_root)
    approvals_root = resolve_path(args.approvals_root, project_root)

    if not render_root.exists():
        raise FileNotFoundError(f"Render root does not exist: {render_root}")

    images_root.mkdir(parents=True, exist_ok=True)
    approvals_root.mkdir(parents=True, exist_ok=True)

    mesh_paths = discover_mesh_files(render_root)
    if not mesh_paths:
        raise FileNotFoundError(f"No .glb/.gltf/.obj files found under {render_root}")

    if args.limit_models > 0:
        mesh_paths = mesh_paths[: args.limit_models]
    output_format = normalize_output_format(args.output_format)
    if args.render_size < 64:
        raise ValueError("--render-size must be >= 64")
    if args.preview_size < 64:
        raise ValueError("--preview-size must be >= 64")
    if args.gc_every_frames < 0:
        raise ValueError("--gc-every-frames must be >= 0")
    if not (1 <= args.output_quality <= 100):
        raise ValueError("--output-quality must be between 1 and 100")
    if args.distance_min_scale <= 0:
        raise ValueError("--distance-min-scale must be > 0")
    if args.distance_max_scale < args.distance_min_scale:
        raise ValueError("--distance-max-scale must be >= --distance-min-scale")
    if args.offcenter_xy_scale < 0.0:
        raise ValueError("--offcenter-xy-scale must be >= 0")
    if args.offcenter_z_scale < 0.0:
        raise ValueError("--offcenter-z-scale must be >= 0")
    if not (0.0 <= args.center_hold_probability <= 1.0):
        raise ValueError("--center-hold-probability must be in [0, 1]")
    if not (0.0 <= args.cloud_probability <= 1.0):
        raise ValueError("--cloud-probability must be in [0, 1]")
    if args.cloud_max_layers < 0:
        raise ValueError("--cloud-max-layers must be >= 0")
    if args.variants_per_combo < 1:
        raise ValueError("--variants-per-combo must be >= 1")
    if args.distance_bins < 1:
        raise ValueError("--distance-bins must be >= 1")
    if args.target_images_per_model < 0:
        raise ValueError("--target-images-per-model must be >= 0")
    if args.resolution_profile == "fixed" and args.render_size < 64:
        raise ValueError("--render-size must be >= 64 when --resolution-profile=fixed")

    # Validate non-fixed resolution profiles early so CLI feedback is immediate.
    dummy_cfg = RenderConfig(
        render_size=args.render_size,
        azimuth_step_deg=args.azimuth_step,
        elevation_min_deg=args.elevation_min,
        elevation_max_deg=args.elevation_max,
        elevation_step_deg=args.elevation_step,
        resolution_profile=args.resolution_profile,
        noise_profile=args.noise_profile,
        background_scene_profile=args.background_scene_profile,
    )
    _ = dummy_cfg.resolution_options
    _ = resolve_noise_settings(args.noise_profile)

    if args.auto_approve_all_first and args.approval_mode != "off":
        newly_approved, already_approved = bulk_auto_approve(
            mesh_paths=mesh_paths,
            render_root=render_root,
            approvals_root=approvals_root,
            force_review=args.force_review,
        )
        print(
            "[auto-approve] "
            f"newly_approved={newly_approved} already_approved={already_approved} "
            f"models={len(mesh_paths)}"
        )

    render_config = RenderConfig(
        render_size=args.render_size,
        azimuth_step_deg=args.azimuth_step,
        elevation_min_deg=args.elevation_min,
        elevation_max_deg=args.elevation_max,
        elevation_step_deg=args.elevation_step,
        gc_every_frames=args.gc_every_frames,
        distance_min_scale=args.distance_min_scale,
        distance_max_scale=args.distance_max_scale,
        offcenter_xy_scale=args.offcenter_xy_scale,
        offcenter_z_scale=args.offcenter_z_scale,
        center_hold_probability=args.center_hold_probability,
        cloud_probability=args.cloud_probability,
        cloud_max_layers=args.cloud_max_layers,
        frame_seed=args.seed,
        output_format=output_format,
        output_quality=args.output_quality,
        resolution_profile=args.resolution_profile,
        noise_profile=args.noise_profile,
        background_scene_profile=args.background_scene_profile,
        combination_strategy=args.combination_strategy,
        combo_selection=args.combo_selection,
        variants_per_combo=args.variants_per_combo,
        distance_bins=args.distance_bins,
        target_images_per_model=args.target_images_per_model,
    )
    estimates = estimate_frame_counts(render_config=render_config, model_count=len(mesh_paths))
    print(
        "[estimate] "
        f"poses={estimates['pose_count']} combos={estimates['combination_count']} "
        f"all_per_model={estimates['all_combos_per_model']:,} "
        f"planned_per_model={estimates['planned_per_model']:,} "
        f"all_total={estimates['all_combos_total']:,} "
        f"planned_total={estimates['planned_total']:,}"
    )
    images_index = build_existing_images_index(images_root)

    rendered_models = 0
    skipped_models = 0
    failed_models = 0
    total_images = 0

    for mesh_path in mesh_paths:
        try:
            approval_status, _ = check_or_update_approval(
                mesh_path=mesh_path,
                render_root=render_root,
                approvals_root=approvals_root,
                approval_mode=args.approval_mode,
                force_review=args.force_review,
                auto_approve=args.auto_approve,
                preview_size=args.preview_size,
                gc_every_frames=args.gc_every_frames,
            )

            if args.approval_mode == "only":
                print(f"[review-only] {mesh_path.name}: status={approval_status}")
                skipped_models += 1
                continue

            if approval_status != "approved":
                print(f"[skip] {mesh_path.name}: status={approval_status} (not approved for rendering)")
                skipped_models += 1
                continue

            image_count = render_one_mesh(
                mesh_path=mesh_path,
                render_root=render_root,
                images_root=images_root,
                images_index=images_index,
                render_config=render_config,
                clean_output=args.clean_output,
            )
            rendered_models += 1
            total_images += image_count
        except Exception as exc:
            failed_models += 1
            print(f"[error] {mesh_path}: {exc}")

    print(
        "[summary] "
        f"models={len(mesh_paths)} rendered={rendered_models} skipped={skipped_models} "
        f"failed={failed_models} images={total_images}"
    )


if __name__ == "__main__":
    main()
