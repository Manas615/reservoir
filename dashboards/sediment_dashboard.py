from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Reservoir Sediment Hotspots", layout="wide")
st.title("Reservoir Sediment Hotspot Prediction")

heatmap_path = st.sidebar.text_input("Hotspot heatmap .npy", "data/outputs/future_sediment_hotspot_heatmap.npy")
zone_path = st.sidebar.text_input("Zone risk CSV", "data/outputs/zone_sediment_risk.csv")
attention_path = st.sidebar.text_input("Attention CSV", "data/outputs/modality_attention.csv")
timeline_path = st.sidebar.text_input("Risk timeline CSV", "data/outputs/risk_timeline.csv")
use_demo_data = st.sidebar.toggle("Show demo data when outputs are missing", value=True)


@st.cache_data
def demo_heatmap() -> np.ndarray:
    y, x = np.mgrid[-1:1:128j, -1:1:128j]
    channel = np.exp(-((x + 0.25) ** 2 / 0.18 + (y - 0.1) ** 2 / 0.45))
    delta = 0.65 * np.exp(-((x - 0.35) ** 2 / 0.08 + (y + 0.35) ** 2 / 0.16))
    ridge = 0.35 * np.exp(-((x + y * 0.8) ** 2 / 0.04))
    heatmap = channel + delta + ridge
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap


@st.cache_data
def demo_zone_risk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone": [f"Zone {chr(65 + i)}" for i in range(8)],
            "risk": [0.18, 0.32, 0.47, 0.71, 0.56, 0.28, 0.63, 0.39],
        }
    )


@st.cache_data
def demo_attention() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "modality": ["satellite", "flow", "terrain", "landuse", "rain"],
            "attention_weight": [0.34, 0.25, 0.17, 0.10, 0.14],
        }
    )


@st.cache_data
def demo_timeline() -> pd.DataFrame:
    start = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [start + timedelta(days=i) for i in range(14)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "mean_risk": [0.24, 0.28, 0.31, 0.38, 0.46, 0.53, 0.61, 0.57, 0.49, 0.44, 0.39, 0.35, 0.32, 0.30],
            "high_risk_area": [0.08, 0.09, 0.12, 0.16, 0.22, 0.29, 0.34, 0.31, 0.26, 0.21, 0.18, 0.15, 0.13, 0.11],
        }
    )


def file_exists(path: str) -> bool:
    return Path(path).expanduser().exists()


def missing_message(path: str) -> str:
    return (
        f"Waiting for `{path}`. Run sediment inference to replace the demo "
        "with model output."
    )

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Reservoir Hotspot Heatmap")
    if file_exists(heatmap_path):
        heatmap = np.load(heatmap_path)
        st.image(heatmap, clamp=True, caption="Future sediment accumulation probability")
    elif use_demo_data:
        st.image(demo_heatmap(), clamp=True, caption="Demo sediment accumulation probability")
        st.caption(missing_message(heatmap_path))
    else:
        st.info(missing_message(heatmap_path))

with right:
    st.subheader("Zone-Wise Predictions")
    if file_exists(zone_path):
        zone_df = pd.read_csv(zone_path)
    elif use_demo_data:
        zone_df = demo_zone_risk()
        st.caption(missing_message(zone_path))
    else:
        zone_df = None
        st.info(missing_message(zone_path))

    if zone_df is not None:
        zone_df = zone_df.copy()
        zone_df["risk_percent"] = zone_df["risk"] * 100
        st.dataframe(zone_df, use_container_width=True)
        st.bar_chart(zone_df.set_index("zone")["risk_percent"])

st.subheader("Sediment Risk Timeline")
if file_exists(timeline_path):
    timeline = pd.read_csv(timeline_path)
elif use_demo_data:
    timeline = demo_timeline()
    st.caption(missing_message(timeline_path))
else:
    timeline = None
    st.info(missing_message(timeline_path))

if timeline is not None:
    timeline = timeline.copy()
    timeline["timestamp"] = pd.to_datetime(timeline["timestamp"])
    st.line_chart(timeline.set_index("timestamp"))

st.subheader("Model Explanation")
if file_exists(attention_path):
    attention = pd.read_csv(attention_path)
elif use_demo_data:
    attention = demo_attention()
    st.caption(missing_message(attention_path))
else:
    attention = None
    st.info(missing_message(attention_path))

if attention is not None:
    st.bar_chart(attention.set_index("modality")["attention_weight"])
    top = attention.sort_values("attention_weight", ascending=False).head(3)["modality"].tolist()
    st.write("High sediment risk due to " + " + ".join(top) + ".")
