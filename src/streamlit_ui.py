from __future__ import annotations

import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from inference import load_checkpoint, predict_images


DEFAULT_CHECKPOINT = Path("checkpoints") / "best_model.pt"
DEFAULT_CSV = Path("data") / "aircraft_rcs.csv"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .hero {
            background: linear-gradient(135deg, #0b1a2a 0%, #1a365d 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2.5rem 3rem;
            color: #ffffff;
            box-shadow: 0 20px 40px -10px rgba(11, 26, 42, 0.3);
            margin-bottom: 2rem;
            text-align: center;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.6rem;
            line-height: 1.2;
            letter-spacing: -0.02em;
            font-weight: 800;
            color: #ffffff !important;
        }
        .hero p {
            margin: 1rem auto 0;
            color: #bcccdc;
            font-size: 1.15rem;
            line-height: 1.6;
            max-width: 800px;
            font-weight: 400;
        }
        .panel {
            border-radius: 16px;
            padding: 1.8rem;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
            margin-bottom: 1.5rem;
        }
        .mini-note {
            color: #627d98;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-top: 0.8rem;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border-radius: 12px;
            padding: 1.2rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            border-color: #cbd5e1;
        }
        [data-testid="stMetricValue"] {
            color: #102a43 !important;
            font-weight: 700;
            font-size: 1.8rem;
        }
        [data-testid="stMetricLabel"] {
            color: #486581 !important;
            font-weight: 600;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        button[kind="primary"] {
            background: linear-gradient(135deg, #1a365d 0%, #244c82 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.8rem 1.5rem !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            box-shadow: 0 4px 10px rgba(26, 54, 93, 0.2) !important;
            transition: all 0.2s ease !important;
            width: 100%;
            margin-top: 1rem;
        }
        button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 15px rgba(26, 54, 93, 0.3) !important;
            background: linear-gradient(135deg, #244c82 0%, #2b5c9e 100%) !important;
        }
        .stExpander {
            background: #f8fafc !important;
            border-radius: 12px !important;
            border: 1px solid #e2e8f0 !important;
        }
        h2, h3 {
            color: #102a43 !important;
            font-weight: 700 !important;
        }
        hr {
            margin-top: 1em;
            margin-bottom: 1em;
            border-color: #e2e8f0;
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
    device = torch.device(device_name)
    model, transform, target_mode = load_checkpoint(checkpoint_path=checkpoint_path, device=device)
    return model, transform, target_mode, device


def run_single_inference(uploaded_file, bundle) -> tuple[float, dict]:
    model, transform, target_mode, device = bundle
    suffix = Path(uploaded_file.name).suffix or ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    try:
        predictions, profile_stats = predict_images(
            model=model,
            transform=transform,
            image_paths=[tmp_path],
            device=device,
            target_mode=target_mode,
            batch_size=1,
            num_workers=0,
            cache_mode="off",
            cache_dir=str(Path(".cache") / "image2rcs_ui"),
            profile=False,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if not predictions:
        raise RuntimeError("Inference returned no predictions.")
    _, predicted_rcs = predictions[0]
    return float(predicted_rcs), profile_stats


def build_comparison_dataset(reference_df: pd.DataFrame, predicted_rcs: float, max_items: int) -> pd.DataFrame:
    ref = reference_df.copy()
    ref["distance"] = (ref["rcs"].apply(lambda val: abs(val - predicted_rcs))).astype(float)

    nearest = ref.nsmallest(max_items, "distance")
    anchors = ref.sort_values("rcs", ascending=False).head(2)
    anchors = pd.concat([anchors, ref.sort_values("rcs", ascending=True).head(2)], ignore_index=True)

    display = pd.concat([nearest, anchors], ignore_index=True).drop_duplicates(subset=["aircraft_name"])
    display = display.sort_values("rcs", ascending=False).reset_index(drop=True)

    predicted_row = pd.DataFrame(
        [
            {
                "aircraft_name": "Your Uploaded Image",
                "aircraft_type": "prediction",
                "rcs": predicted_rcs,
            }
        ]
    )
    chart_df = pd.concat([display[["aircraft_name", "aircraft_type", "rcs"]], predicted_row], ignore_index=True)
    return chart_df


def make_comparison_chart(chart_df: pd.DataFrame) -> alt.Chart:
    low = max(float(chart_df["rcs"].min()) * 0.6, 1e-5)
    high = float(chart_df["rcs"].max()) * 1.7
    y_order = chart_df.sort_values("rcs", ascending=False)["aircraft_name"].tolist()

    chart = (
        alt.Chart(chart_df)
        .mark_circle(opacity=0.85, stroke="#16334f", strokeWidth=1.2)
        .encode(
            x=alt.X(
                "rcs:Q",
                title="Radar Cross-Section (m², log scale)",
                scale=alt.Scale(type="log", domain=[low, high]),
                axis=alt.Axis(labelFontSize=11, titleFontSize=12, format=".1e"),
            ),
            y=alt.Y("aircraft_name:N", title=None, sort=y_order, axis=alt.Axis(labelFontSize=10)),
            size=alt.Size("rcs:Q", title="RCS", scale=alt.Scale(range=[100, 1800])),
            color=alt.condition(
                alt.datum.aircraft_type == "prediction",
                alt.value("#ff4f3f"),
                alt.Color(
                    "aircraft_type:N",
                    legend=alt.Legend(title="Aircraft Type", labelFontSize=10, titleFontSize=11),
                    scale=alt.Scale(
                        range=[
                            "#4c78a8",
                            "#f58518",
                            "#54a24b",
                            "#e45756",
                            "#72b7b2",
                            "#b279a2",
                            "#ff9da6",
                            "#9d755d",
                        ]
                    ),
                ),
            ),
            tooltip=[
                alt.Tooltip("aircraft_name:N", title="Aircraft"),
                alt.Tooltip("aircraft_type:N", title="Type"),
                alt.Tooltip("rcs:Q", title="RCS (m²)", format=".6f"),
            ],
        )
        .properties(
            height=max(420, 32 * len(chart_df)), 
            title=alt.TitleParams(
                text="RCS Comparison Chart",
                subtitle=["Your prediction (red) compared to known aircraft types"],
                fontSize=15,
                fontWeight="bold",
                subtitleFontSize=12,
                subtitleColor="#486581"
            ),
            width="container"
        )
        .interactive()
    )
    return chart


def main() -> None:
    st.set_page_config(page_title="Image2RCS Explorer", page_icon="✈️", layout="wide")
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Radar Cross-Section (RCS) Explorer</h1>
            <p>Upload an aircraft image, predict its RCS, and compare it on a visual scale against known aircraft.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Settings", divider="blue")
        checkpoint_path = st.text_input("Checkpoint path", value=str(DEFAULT_CHECKPOINT))
        csv_path = st.text_input("Reference CSV path", value=str(DEFAULT_CSV))
        compare_count = st.slider("Comparison entries", min_value=8, max_value=40, value=18, step=2)
        show_profile = st.checkbox("📊 Show inference timing", value=False)
        if torch.cuda.is_available():
            device_name = st.selectbox("Device", options=["cuda", "cpu"], index=0)
        else:
            device_name = "cpu"
            st.caption("⚠️ CUDA not detected. Running on CPU.")

    left_col, right_col = st.columns([1.0, 1.6], gap="large")

    with left_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Input Image")
        uploaded_file = st.file_uploader("Upload aircraft image", type=["jpg", "jpeg", "png", "webp", "bmp"])
        if uploaded_file is not None:
            preview = Image.open(uploaded_file).convert("RGB")
            st.image(preview, caption="Uploaded image", width='stretch')
            st.markdown('<p class="mini-note">✈️ Tip: front, side, or top views work best for comparison.</p>', unsafe_allow_html=True)
        run_inference = st.button("🔍 Predict and Compare", type="primary", width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("RCS Comparison")

        if run_inference:
            if uploaded_file is None:
                st.warning("⚠️ Please upload an aircraft image first to proceed with prediction.")
            else:
                try:
                    reference_df = load_reference_table(csv_path)
                    bundle = load_inference_bundle(checkpoint_path, device_name)
                    predicted_rcs, profile_stats = run_single_inference(uploaded_file, bundle)

                    closest_idx = (reference_df["rcs"] - predicted_rcs).abs().idxmin()
                    closest = reference_df.loc[closest_idx]
                    percentile = float((reference_df["rcs"] <= predicted_rcs).mean() * 100.0)

                    with st.container(border=True):
                        m1_col, m2_col, m3_col = st.columns(3, gap="small")
                        with m1_col:
                            st.metric("🎯 Predicted RCS", f"{predicted_rcs:.6f} m²")
                        with m2_col:
                            st.metric("🔗 Closest Match", str(closest["aircraft_name"]))
                        with m3_col:
                            st.metric("📊 Percentile", f"{percentile:.1f}%")

                    chart_df = build_comparison_dataset(reference_df, predicted_rcs, compare_count)
                    chart = make_comparison_chart(chart_df)
                    st.altair_chart(chart, width='stretch')

                    with st.expander("📋 Show comparison table", expanded=False):
                        table = chart_df.sort_values("rcs", ascending=False).reset_index(drop=True)
                        st.dataframe(table, width='stretch', hide_index=True, use_container_height=True)

                    if show_profile:
                        with st.expander("⏱️ Inference timing", expanded=False):
                            st.json(profile_stats)
                except Exception as exc:
                    st.error(f"❌ Inference failed: {exc}")

        else:
            st.info("ℹ️ Upload an image and click 'Predict and Compare' to view your point on the RCS comparison chart.")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()