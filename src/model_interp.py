from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

try:
    from captum.attr import Occlusion
except ImportError as exc:
    raise SystemExit(
        "Captum is not installed. Install dependencies with: pip install -r requirements.txt"
    ) from exc

from dataset import ImageTensorCache, inverse_transform_target_tensor
from device_utils import resolve_device
from inference import load_checkpoint


def load_and_preprocess(image_path: Path, transform) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    cache = ImageTensorCache(mode="off")
    image_tensor = cache.get(image_path)
    model_input = transform(image_tensor).unsqueeze(0)

    original_image = image_tensor.permute(1, 2, 0).float().div(255.0).clamp(0.0, 1.0).numpy()

    # Keep display image at model-input resolution so attribution coordinates align exactly.
    target_h, target_w = int(model_input.shape[-2]), int(model_input.shape[-1])
    resized = F.interpolate(
        image_tensor.unsqueeze(0).float().div(255.0),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    model_res_image = resized.permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    return model_input, model_res_image, original_image


def compute_heatmap(attributions: torch.Tensor) -> np.ndarray:
    arr = attributions.detach().cpu().squeeze(0).numpy()
    heatmap = np.abs(arr).sum(axis=0)
    min_val = float(heatmap.min())
    max_val = float(heatmap.max())
    denom = max(max_val - min_val, 1e-12)
    return (heatmap - min_val) / denom


def select_tile_points(
    heatmap: np.ndarray,
    threshold: float,
    max_tiles: int,
    min_distance: float,
) -> list[tuple[float, float, float]]:
    ys, xs = np.where(heatmap >= threshold)
    if xs.size == 0:
        # Fallback: always show at least one tile at the strongest point.
        flat_index = int(np.argmax(heatmap))
        y_peak, x_peak = np.unravel_index(flat_index, heatmap.shape)
        return [(float(x_peak), float(y_peak), float(heatmap[y_peak, x_peak]))]

    scores = heatmap[ys, xs]
    order = np.argsort(scores)[::-1]
    selected: list[tuple[float, float, float]] = []
    min_dist_sq = float(min_distance * min_distance)

    for idx in order:
        x = float(xs[idx])
        y = float(ys[idx])
        score = float(scores[idx])
        too_close = any(((x - sx) ** 2 + (y - sy) ** 2) < min_dist_sq for sx, sy, _ in selected)
        if too_close:
            continue
        selected.append((x, y, score))
        if len(selected) >= max_tiles:
            break
    return selected


def rasterize_tiles(
    height: int,
    width: int,
    tiles: list[tuple[float, float, float]],
    threshold: float,
    tile_size: float,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    for x, y, score in tiles:
        strength = (score - threshold) / max(1.0 - threshold, 1e-9)
        strength = float(np.clip(strength, 0.0, 1.0))
        side = tile_size * (0.85 + 0.7 * strength)
        half = side / 2.0

        x0 = max(0, int(np.floor(x - half)))
        y0 = max(0, int(np.floor(y - half)))
        x1 = min(width, int(np.ceil(x + half)))
        y1 = min(height, int(np.ceil(y + half)))
        if x1 <= x0 or y1 <= y0:
            continue

        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], float(score))
    return mask


def _neighbors4(y: int, x: int, h: int, w: int):
    if y > 0:
        yield y - 1, x
    if y + 1 < h:
        yield y + 1, x
    if x > 0:
        yield y, x - 1
    if x + 1 < w:
        yield y, x + 1


def mask_to_merged_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    h, w = mask.shape
    active = mask > 0.0
    visited = np.zeros_like(active, dtype=bool)
    boxes: list[tuple[int, int, int, int, float]] = []

    ys, xs = np.where(active)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue

        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        min_y = max_y = start_y
        min_x = max_x = start_x
        peak = float(mask[start_y, start_x])

        while stack:
            y, x = stack.pop()
            value = float(mask[y, x])
            if value > peak:
                peak = value
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x

            for ny, nx in _neighbors4(y, x, h, w):
                if visited[ny, nx] or not active[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))

        # +1 on max to make right/bottom edges exclusive for Rectangle width/height math.
        boxes.append((min_x, min_y, max_x + 1, max_y + 1, peak))

    return boxes


def draw_green_tiles(
    ax,
    merged_boxes: list[tuple[float, float, float, float, float]],
    threshold: float,
    alpha: float,
) -> None:
    for x0, y0, x1, y1, score in merged_boxes:
        strength = (score - threshold) / max(1.0 - threshold, 1e-9)
        strength = float(np.clip(strength, 0.0, 1.0))

        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)
        fill_alpha = alpha * (0.35 + 0.65 * strength)

        tile = Rectangle(
            (x0, y0),
            width=width,
            height=height,
            facecolor=(0.15, 0.95, 0.35, fill_alpha),
            edgecolor=(0.06, 0.58, 0.22, min(1.0, alpha + 0.3)),
            linewidth=1.1,
        )
        ax.add_patch(tile)


def remap_boxes_to_size(
    merged_boxes: list[tuple[int, int, int, int, float]],
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> list[tuple[float, float, float, float, float]]:
    if src_h <= 0 or src_w <= 0:
        return []
    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)

    remapped: list[tuple[float, float, float, float, float]] = []
    for x0, y0, x1, y1, peak in merged_boxes:
        remapped.append((x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y, peak))
    return remapped


def generate_interpretation(
    image_path: Path,
    model: torch.nn.Module,
    transform,
    target_mode: str,
    device: torch.device,
    patch_size: int = 15,
    stride: int = 8,
    tile_threshold: float = 0.65,
    max_tiles: int = 40,
    tile_alpha: float = 0.42,
) -> tuple[plt.Figure, float]:
    input_image, model_res_image, original_image = load_and_preprocess(image_path, transform=transform)
    input_image = input_image.to(device)

    with torch.inference_mode():
        pred_transformed = model(input_image).squeeze(1)
        pred_rcs = float(inverse_transform_target_tensor(pred_transformed, mode=target_mode).item())

    occ = Occlusion(model)
    attributions = occ.attribute(
        input_image,
        strides=(3, stride, stride),
        sliding_window_shapes=(3, patch_size, patch_size),
        baselines=0.0,
        target=None,
    )
    heatmap = compute_heatmap(attributions)

    tiles = select_tile_points(
        heatmap=heatmap,
        threshold=float(tile_threshold),
        max_tiles=int(max_tiles),
        min_distance=max(3.0, float(patch_size) * 0.6),
    )
    tile_size = max(4.0, float(patch_size) * 0.95)

    tile_mask = rasterize_tiles(
        height=int(heatmap.shape[0]),
        width=int(heatmap.shape[1]),
        tiles=tiles,
        threshold=float(tile_threshold),
        tile_size=tile_size,
    )
    merged_boxes_model = mask_to_merged_boxes(tile_mask)
    merged_boxes_original = remap_boxes_to_size(
        merged_boxes=merged_boxes_model,
        src_h=int(model_res_image.shape[0]),
        src_w=int(model_res_image.shape[1]),
        dst_h=int(original_image.shape[0]),
        dst_w=int(original_image.shape[1]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    # Give the figure a transparent background for nice UI integration
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.set_facecolor('none')
        ax.patch.set_alpha(0.0)

    # Note: To scale occlusion attribution height similarly to the original image, 
    # matplotlib constrained_layout handles the scaling aspect ratios naturally.
    hm = axes[0].imshow(heatmap, cmap="inferno")
    draw_green_tiles(
        axes[0],
        merged_boxes=merged_boxes_model,
        threshold=float(tile_threshold),
        alpha=float(tile_alpha),
    )
    axes[0].set_title(f"Occlusion Attribution Map", color="#1c252e", fontsize=14)
    axes[0].axis("off")
    cb = fig.colorbar(hm, ax=axes[0], fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color='#1c252e')
    cb.outline.set_edgecolor('#1c252e')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#1c252e')

    axes[1].imshow(original_image)
    draw_green_tiles(
        axes[1],
        merged_boxes=merged_boxes_original,
        threshold=float(tile_threshold),
        alpha=float(tile_alpha),
    )
    axes[1].set_title(f"Key Feature Overlay", color="#1c252e", fontsize=14)
    axes[1].axis("off")

    return fig, pred_rcs

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Captum Occlusion heatmap for Image2RCS")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="Device for attribution (cpu|mps|cuda|auto)")
    parser.add_argument("--patch-size", type=int, default=15, help="Occlusion patch size in pixels")
    parser.add_argument("--stride", type=int, default=8, help="Occlusion stride in pixels")
    parser.add_argument("--tile-threshold", type=float, default=0.65, help="Heatmap threshold for tile placement (0..1)")
    parser.add_argument("--max-tiles", type=int, default=40, help="Maximum number of tiles to draw")
    parser.add_argument("--tile-alpha", type=float, default=0.42, help="Tile transparency strength (0..1)")
    parser.add_argument("--output", type=str, default="data/model_interp_occlusion/model_interp_occlusion.png", help="Output figure path")
    parser.add_argument("--show", action="store_true", help="Display the plot window")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if args.patch_size < 1:
        raise ValueError("--patch-size must be >= 1")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if not (0.0 <= args.tile_threshold <= 1.0):
        raise ValueError("--tile-threshold must be between 0 and 1")
    if args.max_tiles < 1:
        raise ValueError("--max-tiles must be >= 1")
    if not (0.0 <= args.tile_alpha <= 1.0):
        raise ValueError("--tile-alpha must be between 0 and 1")

    device = resolve_device(args.device)
    model, transform, target_mode = load_checkpoint(args.checkpoint, device=device)

    fig, pred_rcs = generate_interpretation(
        image_path=image_path,
        model=model,
        transform=transform,
        target_mode=target_mode,
        device=device,
        patch_size=args.patch_size,
        stride=args.stride,
        tile_threshold=args.tile_threshold,
        max_tiles=args.max_tiles,
        tile_alpha=args.tile_alpha,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Saved interpretation to: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()