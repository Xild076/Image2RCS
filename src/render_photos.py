from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
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
    distance_min_scale: float = 0.78
    distance_max_scale: float = 1.45
    offcenter_xy_scale: float = 0.34
    offcenter_z_scale: float = 0.20
    center_hold_probability: float = 0.22
    cloud_probability: float = 0.55
    cloud_max_layers: int = 3
    frame_seed: int = 23
    output_format: str = "webp"
    output_quality: int = 78

    @property
    def elevation_degs(self) -> list[int]:
        if self.elevation_step_deg <= 0:
            raise ValueError("Elevation step must be greater than 0.")
        if self.elevation_max_deg < self.elevation_min_deg:
            raise ValueError("Elevation max must be >= elevation min.")
        return list(range(self.elevation_min_deg, self.elevation_max_deg + 1, self.elevation_step_deg))


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


def look_at(
    camera_position,
    target=np.array([0.0, 0.0, 0.0]),
    up=np.array([0.0, 0.0, 1.0]),
    roll_deg: float = 0.0,
):
    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    roll = math.radians(roll_deg)
    cos_r = math.cos(roll)
    sin_r = math.sin(roll)
    right_r = cos_r * right + sin_r * true_up
    up_r = -sin_r * right + cos_r * true_up

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


def sample_frame_variant(
    rng: np.random.Generator,
    mesh_extents: np.ndarray,
    render_config: RenderConfig,
) -> tuple[float, np.ndarray, dict]:
    distance_scale = float(rng.uniform(render_config.distance_min_scale, render_config.distance_max_scale))
    light_style = LIGHT_STYLE_PRESETS[int(rng.integers(0, len(LIGHT_STYLE_PRESETS)))]

    if rng.random() < render_config.center_hold_probability:
        return distance_scale, np.zeros(3, dtype=np.float64), light_style

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
    return distance_scale, target_bias, light_style


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
) -> None:
    payload = {
        "mesh_path": str(mesh_path),
        "output_dir": str(output_dir),
        "frame_count": frame_count,
        "removed_existing": removed_existing,
        "render_size": render_config.render_size,
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
        "updated_at": utc_now_iso(),
    }
    write_json(manifest_path, payload)


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

    elevation_degs = render_config.elevation_degs
    total_frames = (
        len(render_config.orientation_modes)
        * len(range(0, 360, render_config.azimuth_step_deg))
        * len(elevation_degs)
    )

    frame_idx = 0
    generated_paths: list[Path] = []
    for force_untextured in (False, True):
        scene: pyrender.Scene | None = None
        camera = None
        renderer: pyrender.OffscreenRenderer | None = None
        camera_node: pyrender.Node | None = None
        dir_node: pyrender.Node | None = None
        fill_node: pyrender.Node | None = None

        frame_idx = 0
        generated_paths.clear()
        try:
            scene, mesh_center, mesh_extents, base_radius, camera_yfov = build_scene(
                mesh_path,
                force_untextured=force_untextured,
            )
            camera = pyrender.PerspectiveCamera(yfov=camera_yfov)
            renderer = pyrender.OffscreenRenderer(
                viewport_width=render_config.render_size,
                viewport_height=render_config.render_size,
            )
            camera_node = scene.add(camera, pose=np.eye(4))

            for orientation_name, roll_deg in render_config.orientation_modes:
                for azimuth_deg in range(0, 360, render_config.azimuth_step_deg):
                    for elevation_deg in elevation_degs:
                        rng = frame_rng(model_seed=model_seed, frame_index=frame_idx)
                        phase = (2.0 * math.pi * frame_idx) / max(total_frames - 1, 1)
                        sky, light = select_environment_variant(frame_idx)
                        distance_scale, target_bias, light_style = sample_frame_variant(
                            rng=rng,
                            mesh_extents=mesh_extents,
                            render_config=render_config,
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

                        scene.set_pose(camera_node, pose=cam_pose)
                        color, _ = renderer.render(scene)
                        color = maybe_apply_fake_clouds(
                            color_buffer=color,
                            rng=rng,
                            cloud_probability=render_config.cloud_probability,
                            cloud_max_layers=render_config.cloud_max_layers,
                        )

                        filename = (
                            f"{prefix}{orientation_name}_"
                            f"{sky['name']}_{light['name']}_{light_style['name']}_"
                            f"az{azimuth_deg:03d}_el{elevation_deg:+03d}_{frame_idx:05d}{extension}"
                        )
                        out_path = output_dir / filename
                        save_render_array(
                            color_buffer=color,
                            output_path=out_path,
                            output_format=render_config.output_format,
                            output_quality=render_config.output_quality,
                        )
                        generated_paths.append(out_path)
                        del color
                        frame_idx += 1
                        if render_config.gc_every_frames > 0 and frame_idx % render_config.gc_every_frames == 0:
                            gc.collect()
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
                del renderer
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
    )

    print(
        f"[ok] {mesh_path.name}: wrote {frame_idx} images to {output_dir} "
        f"(removed_existing={removed_existing})"
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
        "--distance-min-scale",
        type=float,
        default=0.78,
        help="Minimum camera distance multiplier relative to base fit distance.",
    )
    parser.add_argument(
        "--distance-max-scale",
        type=float,
        default=1.45,
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
        default=23,
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
