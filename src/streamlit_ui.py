from __future__ import annotations

import inspect
import io
import math
import shutil
import subprocess
import tempfile
from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps

from device_utils import describe_device, mps_is_available, resolve_device
from inference import load_checkpoint, predict_images


DEFAULT_CHECKPOINT = Path("checkpoints") / "best_model.pt"
DEFAULT_CSV = Path("data") / "aircraft_rcs.csv"
RCS_EPSILON = 1e-8


def safe_rcs(value: float, eps: float = RCS_EPSILON) -> float:
    return float(max(float(value), eps))


def to_dbsm(value_m2: float) -> float:
    return float(10.0 * math.log10(safe_rcs(value_m2)))


def format_rcs(value: float) -> str:
    val = float(value)
    abs_val = abs(val)
    if abs_val == 0.0:
        return "0.0000 m^2"
    if abs_val < 1e-3 or abs_val >= 1e3:
        return f"{val:.3e} m^2"
    if abs_val >= 100:
        return f"{val:.2f} m^2"
    if abs_val >= 10:
        return f"{val:.3f} m^2"
    return f"{val:.4f} m^2"


def compute_percentile(reference_rcs: pd.Series, predicted_rcs: float) -> float:
    values = pd.to_numeric(reference_rcs, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return 0.0
    lower = float(np.sum(values < predicted_rcs))
    equal = float(np.sum(values == predicted_rcs))
    return ((lower + (0.5 * equal)) / float(values.size)) * 100.0


def _supports_kwarg(fn, kwarg: str) -> bool:
    try:
        return kwarg in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def render_responsive_image(image: Image.Image, caption: str) -> None:
    kwargs: dict[str, object] = {"caption": caption}
    if _supports_kwarg(st.image, "use_container_width"):
        kwargs["use_container_width"] = True
    elif _supports_kwarg(st.image, "use_column_width"):
        kwargs["use_column_width"] = True
    elif _supports_kwarg(st.image, "width"):
        kwargs["width"] = 520
    st.image(image, **kwargs)


def load_uploaded_preview(uploaded_file) -> Image.Image:
    raw_bytes = uploaded_file.getvalue()
    suffix = Path(getattr(uploaded_file, "name", "")).suffix.lower()
    try:
        with Image.open(io.BytesIO(raw_bytes)) as raw_preview:
            return ImageOps.exif_transpose(raw_preview).convert("RGB")
    except (OSError, ValueError):
        if suffix not in {".heic", ".heif"} or shutil.which("sips") is None:
            raise

        with tempfile.TemporaryDirectory(prefix="image2rcs_preview_") as tmp_dir:
            src_path = Path(tmp_dir) / f"upload{suffix or '.heic'}"
            dst_path = Path(tmp_dir) / "converted_preview.jpg"
            src_path.write_bytes(raw_bytes)
            subprocess.run(
                ["sips", "-s", "format", "jpeg", str(src_path), "--out", str(dst_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with Image.open(dst_path) as converted:
                return ImageOps.exif_transpose(converted).convert("RGB")


def render_full_width_button(label: str, **kwargs) -> bool:
    if _supports_kwarg(st.button, "use_container_width"):
        return bool(st.button(label, use_container_width=True, **kwargs))
    if _supports_kwarg(st.button, "width"):
        return bool(st.button(label, width="stretch", **kwargs))
    return bool(st.button(label, **kwargs))


def render_responsive_altair_chart(chart: alt.Chart) -> None:
    if _supports_kwarg(st.altair_chart, "use_container_width"):
        st.altair_chart(chart, use_container_width=True)
    else:
        st.altair_chart(chart)


def render_responsive_dataframe(table: pd.DataFrame, *, hide_index: bool = True) -> None:
    kwargs: dict[str, object] = {}
    if _supports_kwarg(st.dataframe, "use_container_width"):
        kwargs["use_container_width"] = True
    if _supports_kwarg(st.dataframe, "hide_index"):
        kwargs["hide_index"] = hide_index
    if _supports_kwarg(st.dataframe, "height"):
        visible_rows = min(max(len(table.index), 6), 16)
        kwargs["height"] = 42 + (visible_rows * 35)
    st.dataframe(table, **kwargs)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-color: #f0f4f8;
            --panel-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #475569;
            --text-light: #64748b;
            --border: #cbd5e1;
            --border-light: #e2e8f0;
            --accent: #0f172a;
            --accent-hover: #1e293b;
            --accent-blue: #0284c7;
            --predict: #ea580c;
            --predict-soft: #fed7aa;
            --chart-bg: #ffffff;
            --chart-grid: #f1f5f9;
            --shadow-subtle: 0 2px 8px rgba(0, 0, 0, 0.06);
            --shadow-floating: 0 8px 16px rgba(0, 0, 0, 0.1);
        }

        * { box-sizing: border-box; }

        /* Base setup */
        html, body {
            background-color: var(--bg-color);
            color: var(--text-main);
        }

        .stApp {
            background-color: var(--bg-color);
            font-family: 'Inter', sans-serif;
            color: var(--text-main);
        }

        .main .block-container {
            max-width: 1400px;
            padding: 2rem 3rem;
            background-color: var(--bg-color);
        }
        
        [data-testid="stAppViewContainer"] {
            background-color: var(--bg-color);
        }
        
        /* Column and container spacing */
        [data-testid="column"] {
            gap: 0 !important;
        }
        
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
            gap: 0.5rem !important;
        }

        /* Hero Container */
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 20px;
            padding: 3rem 4rem;
            color: #ffffff;
            box-shadow: var(--shadow-floating);
            margin-bottom: 2.5rem;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .hero::before {
            content: "";
            position: absolute;
            top: -50%; left: -50%;
            width: 200%; height: 200%;
            background: radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.05) 0%, transparent 40%);
            pointer-events: none;
        }
        .hero-eyebrow {
            display: inline-block;
            margin-bottom: 1.2rem;
            padding: 0.35rem 1rem;
            border-radius: 100px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #e0f2fe;
            border: 1px solid rgba(255, 255, 255, 0.15);
            z-index: 1;
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(2.5rem, 4vw, 3.5rem);
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.02em;
            color: #ffffff !important;
            z-index: 1;
        }
        .hero p {
            margin: 1.2rem auto 0;
            font-size: 1.15rem;
            color: #cbd5e1 !important;
            max-width: 700px;
            line-height: 1.6;
            font-weight: 400;
            z-index: 1;
        }

        /* Sleek White Panels */
        .panel {
            background: var(--panel-bg);
            border-radius: 20px;
            padding: 2rem 2.2rem;
            border: 1px solid var(--border-light);
            box-shadow: var(--shadow-subtle);
            transition: box-shadow 0.3s ease, transform 0.3s ease, border-color 0.3s ease;
            position: relative;
        }
        .panel:hover {
            box-shadow: var(--shadow-floating);
            transform: translateY(-1px);
            border-color: var(--border);
        }

        .section-kicker {
            display: inline-block;
            background: #f1f5f9;
            color: #475569;
            padding: 0.35rem 0.9rem;
            border-radius: 100px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 1rem;
        }
        .section-head {
            color: var(--text-main);
            font-size: 1.8rem;
            font-weight: 800;
            margin: 0 0 1.8rem 0;
            letter-spacing: -0.015em;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--border-light);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }
        [data-testid="stSidebar"] label {
            color: var(--text-main) !important;
            font-weight: 500;
            font-size: 0.95rem;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-main) !important;
            font-weight: 500;
        }
        .sidebar-note {
            background: #f8fafc;
            border: 1px solid var(--border-light);
            border-radius: 14px;
            padding: 1.2rem;
            color: #475569;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-top: 1.5rem;
        }

        /* Metric Cards */
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--border-light);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        [data-testid="stMetric"]::before {
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 5px; height: 100%;
            background: var(--text-muted);
            border-radius: 5px 0 0 5px;
            transition: background 0.3s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-floating);
            border-color: var(--border);
            background: #f8fbff;
        }
        [data-testid="stMetric"]:hover::before {
            background: var(--accent-blue);
        }
        
        /* First metric (Prediction) gets orange accent */
        [data-testid="column"]:first-child [data-testid="stMetric"]::before {
            background: var(--predict);
        }

        [data-testid="stMetricValue"] {
            font-size: 1.7rem !important;
            font-weight: 800 !important;
            color: var(--text-main) !important;
            letter-spacing: -0.015em;
            line-height: 1.2;
        }
        [data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.5rem;
        }

        /* Primary Button Style */
        .stButton > button {
            border-radius: 12px !important;
            transition: all 0.2s ease !important;
            font-weight: 600 !important;
            width: 100%;
        }
        
        .stButton > button[kind="primary"] {
            background: var(--accent) !important;
            color: white !important;
            border: none !important;
            padding: 1.2rem !important;
            font-size: 1.05rem !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.15) !important;
        }
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 12px rgba(15, 23, 42, 0.25) !important;
            background: var(--accent-hover) !important;
        }
        .stButton > button[kind="secondary"] {
            background: #f1f5f9 !important;
            color: var(--text-main) !important;
            border: 1px solid var(--border-light) !important;
        }

        /* Native image/chart shells */
        [data-testid="stImage"] {
            border-radius: 16px;
            border: 1px solid var(--border-light);
            background: #ffffff;
            padding: 0.6rem;
            margin: 0.65rem 0 0.95rem;
            box-shadow: var(--shadow-subtle);
        }

        /* Altair Chart Container */
        [data-testid="stVegaLiteChart"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
            border: 1px solid #dbeafe;
            border-radius: 16px;
            padding: 1rem;
            margin: 1rem 0 0.7rem;
            box-shadow: var(--shadow-subtle);
        }
        
        .vega-embed {
            background: #ffffff !important;
            color: #0f172a !important;
        }

        .vega-embed .vega-actions,
        .vega-embed .mark-text,
        .vega-embed text {
            color: #334155 !important;
            fill: #334155 !important;
        }

        /* File Uploader */
        [data-testid="stFileUploaderDropzone"] {
            background: #ffffff;
            border: 2px dashed var(--border);
            border-radius: 16px;
            transition: all 0.2s ease;
            padding: 2rem !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent-blue);
            background: #f0f9ff;
            transform: translateY(-1px);
        }
        
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] p {
            color: #334155 !important;
            font-weight: 500;
            opacity: 1 !important;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: #334155 !important;
        }
        
        /* Status Messages */
        .result-note {
            background: linear-gradient(135deg, #fef3c7 0%, #fef08a 100%);
            border: 1px solid #fbbf24;
            border-radius: 14px;
            padding: 1.2rem 1.5rem;
            color: #92400e;
            font-size: 1rem;
            line-height: 1.6;
            margin: 2rem 0;
        }
        .result-note strong {
            font-weight: 700;
            color: #b45309;
        }
        
        .placeholder-note {
            background: #f8fafc;
            border: 2px dashed var(--border);
            border-radius: 20px;
            padding: 2.2rem 1.6rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 1.1rem;
            font-weight: 500;
            margin-top: 1.5rem;
        }

        /* Expanders & Table */
        [data-testid="stExpander"] {
            border-radius: 16px !important;
            border: 1px solid var(--border-light) !important;
            background: #ffffff !important;
            margin-top: 1.5rem;
            overflow: hidden;
        }
        [data-testid="stExpander"] > div:first-child {
            background: #f8fafc !important;
            border-bottom: 1px solid var(--border-light) !important;
            padding: 1.2rem !important;
        }
        [data-testid="stExpander"] > div:first-child button {
            color: var(--text-main) !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stDataFrame"] {
            border-radius: 16px;
            border: 1px solid var(--border-light);
            overflow: auto;
            box-shadow: var(--shadow-subtle);
        }

        /* Typography & General */
        h2, h3, h4 { 
            color: var(--text-main) !important; 
            font-weight: 800 !important; 
            letter-spacing: -0.01em;
            margin-top: 0 !important;
        }
        h2 { font-size: 1.8rem !important; }
        h3 { font-size: 1.4rem !important; }
        h4 { font-size: 1.15rem !important; }

        .stApp p,
        .stApp span,
        .stApp a,
        .stApp label {
            color: #334155;
        }

        .hero p {
            color: #cbd5e1 !important;
        }

        .hero h1,
        .hero .hero-eyebrow {
            color: #ffffff !important;
        }
        
        [data-testid="stCaptionContainer"] {
            color: #475569 !important;
        }
        
        [data-testid="stMarkdownContainer"] p {
            color: inherit;
        }
        
        .mini-note { 
            color: var(--text-light) !important; 
            font-size: 0.9rem; 
            line-height: 1.5; 
            margin-top: 0.8rem; 
        }
        
        hr { 
            border-color: var(--border-light); 
            margin: 1.5rem 0; 
        }
        
        [data-testid="stAlert"] {
            border-radius: 14px !important;
            border: 1px solid var(--border-light) !important;
            background: #f8fafc !important;
        }
        [data-testid="stAlert"] p {
            color: var(--text-main) !important;
        }

        /* Input Fields */
        [data-testid="stWidgetLabel"] {
            color: var(--text-main) !important;
            font-weight: 600;
        }
        input, select, textarea {
            border-radius: 10px !important;
            border-color: var(--border-light) !important;
            font-family: 'Inter', sans-serif !important;
            color: #334155 !important;
        }
        input::placeholder, select::placeholder, textarea::placeholder {
            color: #94a3b8 !important;
        }
        input:focus, select:focus, textarea:focus {
            border-color: var(--accent-blue) !important;
            box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1) !important;
        }

        /* Responsive Design */
        @media (max-width: 900px) {
            .main .block-container {
                padding-left: 1.5rem;
                padding-right: 1.5rem;
            }
            .hero {
                padding: 2rem 1.5rem;
            }
            .panel {
                padding: 1.5rem;
            }
            .section-head {
                font-size: 1.4rem;
            }
        }
        
        /* Spacing utilities - reduce gaps */
        [data-testid="stVerticalBlock"] {
            gap: 0.75rem !important;
        }
        
        [data-testid="stHorizontalBlock"] {
            gap: 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_reference_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = ["aircraft_name", "aircraft_type", "rcs"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required columns in reference CSV: {missing_text}")

    clean = df[needed].dropna().copy()
    clean["rcs"] = clean["rcs"].astype(float)
    clean = clean.sort_values("rcs", ascending=False).reset_index(drop=True)
    return clean


@st.cache_resource(show_spinner=True)
def load_inference_bundle(checkpoint_path: str, device_name: str):
    device = resolve_device(device_name)
    model, transform, target_mode = load_checkpoint(checkpoint_path=checkpoint_path, device=device)
    return model, transform, target_mode, device


def run_single_inference(uploaded_file, bundle) -> tuple[float, dict]:
    model, transform, target_mode, device = bundle
    suffix = Path(uploaded_file.name).suffix or ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    try:
        predict_kwargs = {
            "model": model,
            "transform": transform,
            "image_paths": [tmp_path],
            "device": device,
            "target_mode": target_mode,
            "batch_size": 1,
            "num_workers": 0,
            "cache_mode": "off",
            "cache_dir": str(Path(".cache") / "image2rcs_ui"),
            "profile": False,
        }
        if _supports_kwarg(predict_images, "memory_cache_items"):
            predict_kwargs["memory_cache_items"] = 64

        predictions, profile_stats = predict_images(**predict_kwargs)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not predictions:
        raise RuntimeError("Inference returned no predictions.")
    _, predicted_rcs = predictions[0]
    return float(predicted_rcs), profile_stats


def build_comparison_dataset(reference_df: pd.DataFrame, predicted_rcs: float, max_items: int) -> pd.DataFrame:
    ref = reference_df.copy()
    predicted_safe = safe_rcs(predicted_rcs)
    pred_log = math.log10(predicted_safe)

    ref["rcs_safe"] = ref["rcs"].clip(lower=RCS_EPSILON).astype(float)
    ref["distance_abs"] = (ref["rcs"] - predicted_rcs).abs().astype(float)
    # Use log-distance so nearest neighbors are accurate for multiplicative RCS scales.
    ref["distance_log"] = (np.log10(ref["rcs_safe"]) - pred_log).abs().astype(float)
    ref["ratio_to_prediction"] = (ref["rcs_safe"] / predicted_safe).astype(float)
    ref["delta_db"] = (10.0 * np.log10(ref["ratio_to_prediction"])).astype(float)

    nearest = ref.nsmallest(max_items, "distance_log")
    anchors = ref.sort_values("rcs", ascending=False).head(2)
    anchors = pd.concat([anchors, ref.sort_values("rcs", ascending=True).head(2)], ignore_index=True)

    display = pd.concat([nearest, anchors], ignore_index=True).drop_duplicates(subset=["aircraft_name"])
    display = display.sort_values("rcs", ascending=False).reset_index(drop=True)
    display["source"] = "Reference"

    predicted_row = pd.DataFrame(
        [
            {
                "aircraft_name": "Your Uploaded Image",
                "aircraft_type": "prediction",
                "rcs": predicted_rcs,
                "rcs_safe": predicted_safe,
                "distance_abs": 0.0,
                "distance_log": 0.0,
                "ratio_to_prediction": 1.0,
                "delta_db": 0.0,
                "source": "Prediction",
            }
        ]
    )
    chart_df = pd.concat(
        [
            display[
                [
                    "aircraft_name",
                    "aircraft_type",
                    "rcs",
                    "rcs_safe",
                    "distance_abs",
                    "distance_log",
                    "ratio_to_prediction",
                    "delta_db",
                    "source",
                ]
            ],
            predicted_row,
        ],
        ignore_index=True,
    )
    return chart_df


def make_comparison_chart(chart_df: pd.DataFrame) -> alt.Chart:
    plot_df = chart_df.copy()
    plot_df["rcs_plot"] = plot_df["rcs_safe"].clip(lower=RCS_EPSILON)
    plot_df["dbsm"] = plot_df["rcs_plot"].map(to_dbsm)
    low = max(float(plot_df["rcs_plot"].min()) * 0.6, 1e-5)
    high = max(float(plot_df["rcs_plot"].max()) * 1.7, low * 10.0)
    y_order = plot_df.sort_values("rcs_plot", ascending=False)["aircraft_name"].tolist()

    prediction_mask = plot_df["source"] == "Prediction"
    if prediction_mask.any():
        prediction_value = float(plot_df.loc[prediction_mask, "rcs_plot"].iloc[0])
    else:
        prediction_value = float(plot_df["rcs_plot"].median())
    lower_band = max(prediction_value / 2.0, low)
    upper_band = min(prediction_value * 2.0, high)

    log_low = math.log10(low)
    log_high = math.log10(high)
    log_span = max(log_high - log_low, 1e-6)
    plot_df["marker_scale"] = ((np.log10(plot_df["rcs_plot"]) - log_low) / log_span).clip(0.0, 1.0)
    plot_df["marker_size"] = 95.0 + (530.0 * np.power(plot_df["marker_scale"], 0.82))
    plot_df.loc[plot_df["source"] == "Prediction", "marker_size"] *= 1.10
    plot_df["marker_size"] = plot_df["marker_size"].clip(lower=90.0, upper=760.0)

    base = alt.Chart(plot_df).encode(
        x=alt.X(
            "rcs_plot:Q",
            title="Radar Cross-Section (m^2, log scale)",
            scale=alt.Scale(type="log", domain=[low, high]),
            axis=alt.Axis(
                labelFontSize=12,
                titleFontSize=13,
                format=".2e",
                gridColor="#e2e8f0",
                titleColor="#334155",
                labelColor="#475569",
                domainColor="#cbd5e1",
                tickColor="#cbd5e1",
                titlePadding=15,
            ),
        ),
        y=alt.Y(
            "aircraft_name:N",
            title=None,
            sort=y_order,
            axis=alt.Axis(labelFontSize=12, labelColor="#334155", domainColor="#cbd5e1", tickColor="#cbd5e1", labelPadding=12),
        ),
        tooltip=[
            alt.Tooltip("aircraft_name:N", title="Aircraft"),
            alt.Tooltip("aircraft_type:N", title="Type"),
            alt.Tooltip("rcs:Q", title="RCS (m^2)", format=".6f"),
            alt.Tooltip("dbsm:Q", title="RCS (dBsm)", format=".2f"),
            alt.Tooltip("distance_abs:Q", title="|Delta| m^2", format=".6f"),
            alt.Tooltip("distance_log:Q", title="|Delta| log10", format=".3f"),
            alt.Tooltip("delta_db:Q", title="Delta (dB)", format=".2f"),
            alt.Tooltip("source:N", title="Source"),
        ],
    )

    references = base.transform_filter(alt.datum.source == "Reference").mark_circle(
        stroke="#0f172a",
        strokeWidth=1.0,
        opacity=0.82,
    ).encode(
        size=alt.Size("marker_size:Q", scale=None, legend=None),
        color=alt.Color(
            "aircraft_type:N",
            legend=alt.Legend(
                title="Aircraft Type",
                labelFontSize=11,
                titleFontSize=12,
                symbolSize=160,
                titleColor="#334155",
                labelColor="#334155",
            ),
            scale=alt.Scale(
                range=[
                    "#0d9488",
                    "#2563eb",
                    "#dc2626",
                    "#7c3aed",
                    "#ca8a04",
                    "#0891b2",
                    "#be123c",
                    "#4338ca",
                ]
            ),
        )
    )

    prediction = base.transform_filter(alt.datum.source == "Prediction").mark_point(
        shape="diamond",
        filled=True,
        stroke="#9a3412",
        strokeWidth=1.5,
        color="#ea580c",
        opacity=1.0,
    ).encode(size=alt.Size("marker_size:Q", scale=None, legend=None))

    band_df = pd.DataFrame({"x0": [lower_band], "x1": [upper_band]})
    confidence_band = alt.Chart(band_df).mark_rect(color="#fed7aa", opacity=0.26).encode(x="x0:Q", x2="x1:Q")

    rule_df = pd.DataFrame({"rcs_plot": [prediction_value], "label": [f"Prediction: {prediction_value:.3e} m^2"]})
    prediction_rule = alt.Chart(rule_df).mark_rule(color="#ea580c", strokeDash=[6, 4], strokeWidth=2.0).encode(x="rcs_plot:Q")
    rule_label = (
        alt.Chart(rule_df)
        .mark_text(
            align="left",
            dx=8,
            dy=-12,
            color="#ea580c",
            fontSize=12,
            fontWeight=700,
        )
        .encode(x="rcs_plot:Q", y=alt.value(10), text="label:N")
    )

    chart = (
        (confidence_band + prediction_rule + references + prediction + rule_label)
        .properties(
            height=min(max(480, 38 * len(plot_df)), 880),
            title=alt.TitleParams(
                text="RCS Comparison Mapping",
                subtitle=["Nearest aircraft are chosen using log-distance for better multiplicative RCS accuracy."],
                fontSize=18,
                fontWeight=800,
                subtitleFontSize=13,
                subtitleColor="#475569",
                offset=16,
                color="#0f172a"
            ),
            width="container",
        )
        .configure(background="#ffffff")
        .configure_view(
            strokeOpacity=0,
            fill="#ffffff",
            continuousWidth=750,
            continuousHeight=480,
            opacity=1.0
        )
        .configure_axis(
            labelFont="Inter, sans-serif",
            titleFont="Inter, sans-serif",
            labelFontSize=12,
            titleFontSize=13,
            labelColor="#334155",
            titleColor="#0f172a",
            gridOpacity=0.12,
            gridColor="#e5e7eb"
        )
        .configure_title(
            font="Inter, sans-serif",
            subtitleFont="Inter, sans-serif",
            anchor="start",
            color="#0f172a",
            subtitleColor="#475569"
        )
        .configure_legend(
            labelFont="Inter, sans-serif",
            titleFont="Inter, sans-serif",
            padding=15,
            labelFontSize=11,
            titleFontSize=12,
            labelColor="#334155",
            titleColor="#0f172a"
        )
        .configure_text(
            font="Inter, sans-serif",
            color="#0f172a"
        )
        .configure_mark(
            opacity=0.9
        )
        .interactive()
    )
    return chart


def main() -> None:
    st.set_page_config(page_title="Image2RCS Explorer", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Image2RCS</div>
            <h1>Radar Cross-Section (RCS) Explorer</h1>
            <p>Upload an aircraft image, run inference, then place it on a comparative RCS map with nearby reference aircraft and percentile context.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.caption("Adjust model inputs and comparison density.")
        checkpoint_path = st.text_input("Checkpoint path", value=str(DEFAULT_CHECKPOINT))
        csv_path = st.text_input("Reference CSV path", value=str(DEFAULT_CSV))
        compare_count = st.slider("Comparison entries", min_value=8, max_value=40, value=18, step=2)
        show_profile = st.checkbox("Show inference timing", value=False)
        device_options = ["auto", "cpu"]
        if torch.cuda.is_available():
            device_options.insert(1, "cuda")
        if mps_is_available():
            device_options.insert(1, "mps")
        device_name = st.selectbox("Device", options=device_options, index=0)
        if not torch.cuda.is_available() and not mps_is_available():
            st.caption("No GPU backend detected. Running on CPU.")
        st.markdown(
            """
            <div class="sidebar-note">
                Comparison entries controls how many nearest reference aircraft are shown in the chart.
                Nearness is computed in log-space to better reflect multiplicative RCS scales.
            </div>
            """,
            unsafe_allow_html=True,
        )

    left_col, right_col = st.columns([1.25, 2.0], gap="large")

    with left_col:
        st.markdown('<div class="section-kicker">Step 1</div><h3 class="section-head">Input Image</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload aircraft image",
            type=["jpg", "jpeg", "png", "webp", "bmp", "heic", "heif", "tif", "tiff"],
        )
        if uploaded_file is not None:
            try:
                preview = load_uploaded_preview(uploaded_file)
                render_responsive_image(preview, caption="Uploaded image preview")
                st.markdown(
                    '<p class="mini-note">Tip: front, side, or top views usually produce more stable comparisons.</p>',
                    unsafe_allow_html=True,
                )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                st.error(f"Could not preview this image format: {exc}")
        else:
            st.markdown(
                '<p class="mini-note">No file selected yet. Upload one image to run single-image inference.</p>',
                unsafe_allow_html=True,
            )
        run_inference = render_full_width_button("Predict and Compare", type="primary")

    with right_col:
        st.markdown('<div class="section-kicker">Step 2</div><h3 class="section-head">RCS Comparison</h3>', unsafe_allow_html=True)

        if run_inference:
            if uploaded_file is None:
                st.warning("Please upload an aircraft image first to proceed with prediction.")
            else:
                try:
                    reference_df = load_reference_table(csv_path)
                    bundle = load_inference_bundle(checkpoint_path, device_name)
                    st.caption(f"Runtime device: {describe_device(bundle[3])}")
                    predicted_rcs, profile_stats = run_single_inference(uploaded_file, bundle)

                    closest_idx = (reference_df["rcs"] - predicted_rcs).abs().idxmin()
                    closest = reference_df.loc[closest_idx]
                    closest_rcs = float(closest["rcs"])
                    delta_rcs = predicted_rcs - closest_rcs
                    ratio_to_closest = safe_rcs(predicted_rcs) / safe_rcs(closest_rcs)
                    delta_db = 10.0 * math.log10(ratio_to_closest)
                    percentile = compute_percentile(reference_df["rcs"], predicted_rcs)

                    m1_col, m2_col, m3_col, m4_col = st.columns(4, gap="small")
                    with m1_col:
                        st.metric("Predicted RCS", format_rcs(predicted_rcs), f"{to_dbsm(predicted_rcs):+.2f} dBsm")
                    with m2_col:
                        st.metric("Closest Match", str(closest["aircraft_name"]))
                    with m3_col:
                        st.metric("Difference", format_rcs(delta_rcs), f"{delta_db:+.2f} dB")
                    with m4_col:
                        st.metric("Percentile Rank", f"{percentile:.1f}%")

                    direction_text = "higher" if delta_rcs >= 0 else "lower"
                    closest_name = escape(str(closest["aircraft_name"]))
                    st.markdown(
                        (
                            f'<div class="result-note">✨ <div>Prediction is <strong>{format_rcs(abs(delta_rcs))} {direction_text}</strong> '
                            f"than closest reference <strong>{closest_name}</strong> ({format_rcs(closest_rcs)}). "
                            f"Relative offset: <strong>{delta_db:+.2f} dB</strong>.</div></div>"
                        ),
                        unsafe_allow_html=True,
                    )

                    chart_df = build_comparison_dataset(reference_df, predicted_rcs, compare_count)
                    chart = make_comparison_chart(chart_df)
                    render_responsive_altair_chart(chart)

                    with st.expander("Show comparison table", expanded=False):
                        table = chart_df.sort_values("rcs", ascending=False).reset_index(drop=True).copy()
                        table["rcs_m2"] = table["rcs"].map(format_rcs)
                        table["rcs_dbsm"] = table["rcs_safe"].map(lambda v: f"{to_dbsm(v):.2f}")
                        table["delta_db"] = table["delta_db"].map(lambda v: f"{float(v):+.2f}")
                        table["log_distance"] = table["distance_log"].map(lambda v: f"{float(v):.3f}")
                        table = table[["aircraft_name", "aircraft_type", "source", "rcs_m2", "rcs_dbsm", "delta_db", "log_distance"]]
                        render_responsive_dataframe(table, hide_index=True)

                    if show_profile:
                        with st.expander("Inference timing", expanded=False):
                            st.json(profile_stats)
                except Exception as exc:
                    st.error(f"❌ Inference failed: {exc}")

        else:
            st.markdown(
                '<div class="placeholder-note">✈️<br/><br/>Upload an image and click <strong>Predict and Compare</strong> '
                "to place it against known aircraft references on the RCS scale.</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
