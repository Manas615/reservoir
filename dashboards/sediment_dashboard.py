import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Reservoir Sediment Hotspots", layout="wide")
st.title("Reservoir Sediment Hotspot Prediction")

heatmap_path = st.sidebar.text_input("Hotspot heatmap .npy", "data/outputs/future_sediment_hotspot_heatmap.npy")
zone_path = st.sidebar.text_input("Zone risk CSV", "data/outputs/zone_sediment_risk.csv")
attention_path = st.sidebar.text_input("Attention CSV", "data/outputs/modality_attention.csv")
timeline_path = st.sidebar.text_input("Risk timeline CSV", "data/outputs/risk_timeline.csv")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Reservoir Hotspot Heatmap")
    try:
        heatmap = np.load(heatmap_path)
        st.image(heatmap, clamp=True, caption="Future sediment accumulation probability")
    except Exception as exc:
        st.info(f"Heatmap not available yet: {exc}")

with right:
    st.subheader("Zone-Wise Predictions")
    try:
        zone_df = pd.read_csv(zone_path)
        zone_df["risk_percent"] = zone_df["risk"] * 100
        st.dataframe(zone_df, use_container_width=True)
        st.bar_chart(zone_df.set_index("zone")["risk_percent"])
    except Exception as exc:
        st.info(f"Zone predictions not available yet: {exc}")

st.subheader("Sediment Risk Timeline")
try:
    timeline = pd.read_csv(timeline_path)
    timeline["timestamp"] = pd.to_datetime(timeline["timestamp"])
    st.line_chart(timeline.set_index("timestamp"))
except Exception as exc:
    st.info(f"Timeline not available yet: {exc}")

st.subheader("Model Explanation")
try:
    attention = pd.read_csv(attention_path)
    st.bar_chart(attention.set_index("modality")["attention_weight"])
    top = attention.sort_values("attention_weight", ascending=False).head(3)["modality"].tolist()
    st.write("High sediment risk due to " + " + ".join(top) + ".")
except Exception as exc:
    st.info(f"Explanation data not available yet: {exc}")
