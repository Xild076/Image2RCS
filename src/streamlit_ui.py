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
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from device_utils import describe_device, mps_is_available, resolve_device
from inference import load_checkpoint, predict_images
from model_interp import generate_interpretation


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
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&display=swap');

        :root {
            --bg-color: #FAFAFA;
            --panel-bg: #FFFFFF;
            --text-main: #2C363F;
            --text-muted: #6A7B82;
            --text-light: #A0AEB4;
            --border: #EAEAE4;
            --border-light: #F4F4EF;
            --accent: #4A6C6F;
            --accent-hover: #365255;
            --predict: #D4795B;
            --predict-soft: #FBF1ED;
            --shadow-soft: 0 4px 16px rgba(44, 54, 63, 0.04);
            --shadow-float: 0 16px 40px rgba(44, 54, 63, 0.08);
        }

        * { box-sizing: border-box; }

        /* Base setup */
        html, body {
            background-color: var(--bg-color);
            color: var(--text-main);
        }

        .stApp {
            background-color: var(--bg-color);
            font-family: 'Outfit', sans-serif;
            color: var(--text-main);
        }

        .main .block-container {
            max-width: 1400px;
            padding: 3rem 4rem;
        }
        
        [data-testid="stAppViewContainer"] {
            background-color: transparent;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-main) !important;
            font-family: 'Playfair Display', serif !important;
            letter-spacing: -0.01em;
            font-weight: 600 !important;
        }

        p, span, div {
            font-family: 'Outfit', sans-serif;
            color: var(--text-main);
            line-height: 1.6;
            letter-spacing: 0.01em;
        }

        /* Hero Container */
        .hero {
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 4rem;
            color: var(--text-main);
            box-shadow: var(--shadow-float);
            margin-bottom: 3rem;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        .hero-eyebrow {
            display: inline-block;
            margin-bottom: 1.2rem;
            padding: 0.4rem 1.2rem;
            background: rgba(74, 108, 111, 0.1);
            border-radius: 40px;
            font-size: 0.85rem;
            font-weight: 500;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--accent);
            z-index: 1;
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(2.5rem, 4vw, 3.5rem);
            font-weight: 700 !important;
            line-height: 1.2;
            color: var(--text-main) !important;
            z-index: 1;
        }
        .hero p {
            margin: 1.5rem auto 0;
            font-size: 1.15rem;
            color: var(--text-muted) !important;
            max-width: 650px;
            z-index: 1;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: #F4F4EF;
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] hr {
            border-color: var(--border);
        }
        [data-testid="stSidebar"] label {
            color: var(--text-main) !important;
            font-weight: 500;
            font-size: 0.9rem;
        }
        
        /* Metric Cards */
        [data-testid="stMetric"] {
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            transition: all 0.3s ease;
            position: relative;
            box-shadow: var(--shadow-soft);
        }
        [data-testid="stMetric"]::before {
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 4px; height: 100%;
            background: var(--accent);
            border-radius: 8px 0 0 8px;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-float);
            border-color: var(--border-light);
        }
        
        /* First metric (Prediction) gets subtle accent */
        [data-testid="column"]:first-child [data-testid="stMetric"]::before {
            background: var(--predict);
        }

        [data-testid="stMetricValue"] {
            font-size: 1.7rem !important;
            font-weight: 500 !important;
            color: var(--text-main) !important;
            font-family: 'Playfair Display', serif !important;
        }
        [data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        /* Buttons */
        .stButton > button {
            background: var(--panel-bg) !important;
            color: var(--text-main) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 500 !important;
            font-size: 1.05rem !important;
            padding: 0.8rem 1.5rem !important;
            transition: all 0.2s ease !important;
            box-shadow: var(--shadow-soft);
        }
        .stButton > button[kind="primary"] {
            background: var(--text-main) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--text-main) !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: var(--shadow-float) !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: #4A5560 !important;
            color: #FFFFFF !important;
            border-color: #4A5560 !important;
        }

        /* Altair Chart Container */
        [data-testid="stVegaLiteChart"] {
            background: var(--panel-bg) !important;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1.5rem 0 1rem;
            box-shadow: var(--shadow-soft);
        }

        /* File Uploader */
        [data-testid="stFileUploaderDropzone"] {
            background: var(--panel-bg);
            border: 1px dashed var(--text-light) !important;
            border-radius: 8px;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-soft);
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border: 1px solid var(--accent) !important;
            background: #F9FAFA;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: var(--text-main) !important;
        }
        
        /* Status / Result Messages */
        .result-note {
            background: var(--predict-soft);
            border: 1px solid rgba(212, 121, 91, 0.2);
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            color: var(--text-main);
            font-size: 1.05rem;
            line-height: 1.6;
            margin: 2rem 0;
            box-shadow: var(--shadow-soft);
        }
        .result-note strong {
            color: var(--predict);
            font-weight: 500;
        }
        
        .placeholder-note {
            background: var(--panel-bg);
            border: 1px dashed var(--border);
            border-radius: 8px;
            padding: 3rem 2rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 1.1rem;
            margin-top: 1.5rem;
            box-shadow: var(--shadow-soft);
        }

        /* Expanders & Table */
        [data-testid="stExpander"] {
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
            background: var(--panel-bg) !important;
            margin-top: 1.5rem;
            box-shadow: var(--shadow-soft);
        }
        [data-testid="stExpander"] > div:first-child {
            background: var(--bg-color) !important;
            border-bottom: 1px solid var(--border) !important;
            padding: 1rem 1.2rem !important;
        }
        [data-testid="stExpander"] > div:first-child p {
            color: var(--text-main) !important;
            font-weight: 500 !important;
        }
        
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
        }

        /* Input Fields */
        input, select, textarea {
            background-color: var(--panel-bg) !important;
            border-radius: 6px !important;
            border: 1px solid var(--border) !important;
            font-family: 'Outfit', sans-serif !important;
            color: var(--text-main) !important;
        }
        input:focus, select:focus, textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px rgba(74, 108, 111, 0.1) !important;
        }

        /* Tweak images */
        [data-testid="stImage"] {
            background: var(--panel-bg);
            padding: 0.5rem;
            border-radius: 8px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-soft);
        }
        [data-testid="stImage"] img {
            border-radius: 4px;
        }

        .section-kicker {
            display: inline-block;
            background: var(--border);
            color: var(--text-muted);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
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
                gridColor="#EAEAE4",
                titleColor="#2C363F",
                labelColor="#6A7B82",
                domainColor="#EAEAE4",
                tickColor="#EAEAE4",
                titlePadding=15,
                gridDash=[4, 4],
            ),
        ),
        y=alt.Y(
            "aircraft_name:N",
            title=None,
            sort=y_order,
            axis=alt.Axis(labelFontSize=12, labelColor="#6A7B82", domainColor="#EAEAE4", tickColor="#EAEAE4", labelPadding=12),
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
        stroke="#2C363F",
        strokeWidth=1.0,
        opacity=0.85,
    ).encode(
        size=alt.Size("marker_size:Q", scale=None, legend=None),
        color=alt.Color(
            "aircraft_type:N",
            legend=alt.Legend(
                title="Aircraft Type",
                labelFontSize=11,
                titleFontSize=12,
                symbolSize=160,
                titleColor="#2C363F",
                labelColor="#6A7B82",
            ),
            scale=alt.Scale(
                range=[
                    "#4A6C6F",
                    "#618A9D",
                    "#A0AEB4",
                    "#D4795B",
                    "#798D98",
                    "#B49A84",
                    "#95A8A4",
                    "#E0A96D",
                ]
            ),
        )
    )

    prediction = base.transform_filter(alt.datum.source == "Prediction").mark_point(
        shape="diamond",
        filled=True,
        stroke="#FFFFFF",
        strokeWidth=1.5,
        color="#D4795B",
        opacity=1.0,
    ).encode(size=alt.Size("marker_size:Q", scale=None, legend=None))

    band_df = pd.DataFrame({"x0": [lower_band], "x1": [upper_band]})
    confidence_band = alt.Chart(band_df).mark_rect(color="#D4795B", opacity=0.08).encode(x="x0:Q", x2="x1:Q")

    rule_df = pd.DataFrame({"rcs_plot": [prediction_value], "label": [f"Estimated RCS: {prediction_value:.3e} m^2"]})
    prediction_rule = alt.Chart(rule_df).mark_rule(color="#D4795B", strokeDash=[4, 4], strokeWidth=2.0).encode(x="rcs_plot:Q")
    rule_label = (
        alt.Chart(rule_df)
        .mark_text(
            align="left",
            dx=8,
            dy=-12,
            color="#D4795B",
            fontSize=13,
            font="Outfit",
            fontWeight=600,
        )
        .encode(x="rcs_plot:Q", y=alt.value(10), text="label:N")
    )

    chart = (
        (confidence_band + prediction_rule + references + prediction + rule_label)
        .properties(
            height=min(max(480, 38 * len(plot_df)), 880),
            title=alt.TitleParams(
                text="Radar Cross-Section Comparison",
                subtitle=["Nearest matches calculated via log-distance for accuracy across scales."],
                fontSize=18,
                fontWeight=600,
                subtitleFontSize=13,
                subtitleColor="#6A7B82",
                offset=16,
                color="#2C363F",
            ),
            width="container",
        )
        .configure(background="transparent")
        .configure_view(
            strokeOpacity=0,
            fill="transparent",
            continuousWidth=750,
            continuousHeight=480,
            opacity=1.0
        )
        .configure_axis(
            labelFont="Outfit",
            titleFont="Playfair Display",
            labelFontSize=12,
            titleFontSize=13,
            labelColor="#6A7B82",
            titleColor="#2C363F",
            gridOpacity=0.6,
            gridColor="#EAEAE4",
            gridDash=[4, 4],
        )
        .configure_title(
            font="Playfair Display",
            subtitleFont="Outfit",
            anchor="start",
            color="#2C363F",
            subtitleColor="#6A7B82"
        )
        .configure_legend(
            labelFont="Outfit",
            titleFont="Outfit",
            padding=15,
            labelFontSize=11,
            titleFontSize=12,
            labelColor="#6A7B82",
            titleColor="#2C363F"
        )
        .configure_text(
            font="Outfit",
            color="#2C363F"
        )
        .configure_mark(
            opacity=0.9
        )
        .interactive()
    )
    return chart


def main() -> None:
    st.set_page_config(page_title="RCS Analysis", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Aviation Analysis</div>
            <h1>Radar Cross-Section Estimation</h1>
            <p>Upload a photograph to process an AI-driven estimate of the aircraft's Radar Cross-Section. The system will map your image against a curated database of known aviation signatures.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Configuration")
        st.caption("Adjust model and comparison parameters.")
        checkpoint_path = st.text_input("Model Checkpoint", value=str(DEFAULT_CHECKPOINT))
        csv_path = st.text_input("Reference Database", value=str(DEFAULT_CSV))
        compare_count = st.slider("Comparison Nodes", min_value=8, max_value=40, value=18, step=2)
        show_profile = st.checkbox("Show processing metrics", value=False)
        device_options = ["auto", "cpu"]
        if torch.cuda.is_available():
            device_options.insert(1, "cuda")
        if mps_is_available():
            device_options.insert(1, "mps")
        device_name = st.selectbox("Compute Engine", options=device_options, index=0)
        
        st.markdown("---")
        st.markdown("## Interpretability")
        st.caption("Parameters for feature attribution mapping.")
        patch_size = st.slider("Patch Size (px)", 5, 30, 15, help="Size of the occlusion window in pixels")
        stride = st.slider("Stride (px)", 2, 20, 8, help="Step size for moving the occlusion window")
        tile_threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.65, help="Heatmap threshold for visual markers")

        st.markdown(
            """
            <div class="sidebar-note">
                <strong>Note:</strong> Comparison proximity is calculated using logarithmic distance to better reflect the multiplicative scale of RCS values.
            </div>
            """,
            unsafe_allow_html=True,
        )

    left_col, right_col = st.columns([1.25, 2.0], gap="large")

    with left_col:
        st.markdown('<div class="section-kicker">Step 01</div><h3 class="section-head">Image Selection</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Select an aircraft image",
            type=["jpg", "jpeg", "png", "webp", "bmp", "heic", "heif", "tif", "tiff"],
        )
        if uploaded_file is not None:
            try:
                preview = load_uploaded_preview(uploaded_file)
                render_responsive_image(preview, caption="Image successfully loaded")
                st.markdown(
                    '<p class="mini-note">Tip: Orthogonal profiles (top, side, or front views) generally produce more accurate estimations.</p>',
                    unsafe_allow_html=True,
                )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                st.error(f"Error processing image format: {exc}")
        else:
            st.markdown(
                '<p class="mini-note">Please upload an image to begin.</p>',
                unsafe_allow_html=True,
            )
            
        analyze_button = render_full_width_button("Analyze Image & Estimate RCS", type="primary")

    with right_col:
        st.markdown('<div class="section-kicker">Step 02</div><h3 class="section-head">Analysis Results</h3>', unsafe_allow_html=True)

        if analyze_button:
            if uploaded_file is None:
                st.warning("Please upload an image before proceeding with the analysis.")
            else:
                try:
                    with st.spinner("Processing image and generating estimations..."):
                        # Phase 1: Inference & Predictions
                        reference_df = load_reference_table(csv_path)
                        bundle = load_inference_bundle(checkpoint_path, device_name)
                        model, transform, target_mode, device = bundle
                        st.caption(f"Compute Engine Active: {describe_device(device)}")
                        
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
                            st.metric("Estimated RCS", format_rcs(predicted_rcs), f"{to_dbsm(predicted_rcs):+.2f} dBsm")
                        with m2_col:
                            st.metric("Nearest Match", str(closest["aircraft_name"]))
                        with m3_col:
                            st.metric("Variance", format_rcs(delta_rcs), f"{delta_db:+.2f} dB")
                        with m4_col:
                            st.metric("Percentile", f"{percentile:.1f}%")

                        direction_text = "higher" if delta_rcs >= 0 else "lower"
                        closest_name = escape(str(closest["aircraft_name"]))
                        st.markdown(
                            (
                                f'<div class="result-note">✨ Estimated RCS is <strong>{format_rcs(abs(delta_rcs))} {direction_text}</strong> '
                                f"than the closest reference, the <strong>{closest_name}</strong> ({format_rcs(closest_rcs)}). "
                                f"Relative variance: <strong>{delta_db:+.2f} dB</strong>.</div>"
                            ),
                            unsafe_allow_html=True,
                        )
                        
                        # Phase 2: Interpretability Plot
                        st.markdown("#### Visual Feature Attribution")
                        suffix = Path(uploaded_file.name).suffix or ".png"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = Path(tmp.name)
                        
                        fig, _ = generate_interpretation(
                            image_path=tmp_path,
                            model=model,
                            transform=transform,
                            target_mode=target_mode,
                            device=device,
                            patch_size=patch_size,
                            stride=stride,
                            tile_threshold=tile_threshold,
                        )
                        st.pyplot(fig, transparent=True)
                        tmp_path.unlink(missing_ok=True)
                        st.caption("Areas highlighted identify the features most influential in determining the final RCS estimation.")

                        # Phase 3: Charts & Data
                        st.markdown("#### Database Comparison")
                        chart_df = build_comparison_dataset(reference_df, predicted_rcs, compare_count)
                        chart = make_comparison_chart(chart_df)
                        render_responsive_altair_chart(chart)

                        with st.expander("View Detailed Comparison Data", expanded=False):
                            table = chart_df.sort_values("rcs", ascending=False).reset_index(drop=True).copy()
                            table["rcs_m2"] = table["rcs"].map(format_rcs)
                            table["rcs_dbsm"] = table["rcs_safe"].map(lambda v: f"{to_dbsm(v):.2f}")
                            table["delta_db"] = table["delta_db"].map(lambda v: f"{float(v):+.2f}")
                            table["log_distance"] = table["distance_log"].map(lambda v: f"{float(v):.3f}")
                            table = table[["aircraft_name", "aircraft_type", "source", "rcs_m2", "rcs_dbsm", "delta_db", "log_distance"]]
                            render_responsive_dataframe(table, hide_index=True)

                        if show_profile:
                            with st.expander("View Processing Metrics", expanded=False):
                                st.json(profile_stats)
                except Exception as exc:
                    st.error(f"An error occurred during analysis: {exc}")

        else:
            st.markdown(
                '<div class="placeholder-note">Awaiting image upload.<br/>Select an image and click <strong>Analyze Image & Estimate RCS</strong> to continue.</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
