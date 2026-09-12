import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure repository root is on sys.path so modules like river_weather_predictions and sediment_hotspot import seamlessly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="CSDN • Reservoir Sediment Prediction", layout="wide")
st.title("Counterfactual Sediment Dynamics & Hotspot Prediction")
st.caption("Multimodal AI • CSDN Interventions • WR-GNN • Physics Loss • Causal Attention • MC Dropout")

OUT = REPO_ROOT / "data/outputs"
DIFF_DIR = OUT / "counterfactual_difference_maps"
SENS_DIR = OUT / "sensitivity_maps"


def exists(p: Path | str) -> bool:
    return Path(p).expanduser().exists()


def load_array(p: Path | str, description: str) -> np.ndarray | None:
    path = Path(p)
    if not path.exists():
        st.warning(
            f"Missing required artifact: `{path}` ({description}).\n\n"
            "Please run the pipeline step to generate it:\n"
            "`PYTHONPATH=. python sediment_hotspot/inference/predict_hotspots.py --model data/models/sediment_fusion_model.pt --sample data/processed/sample_0.npz`\n"
            "and\n"
            "`PYTHONPATH=. python -c 'from sediment_hotspot.counterfactual.scenario_engine import run_all_scenarios; run_all_scenarios(\"data/models/sediment_fusion_model.pt\", \"data/processed/sample_0.npz\")'`"
        )
        return None
    return np.load(path)


def plot_heatmap(
    data: np.ndarray,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
    contours: list[float] | None = None,
    contour_colors: list[str] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=120)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    if contours:
        colors = contour_colors or ["#ffffff", "#ffd700"]
        for level, color in zip(contours, colors):
            if data.min() < level < data.max():
                cs = ax.contour(data, levels=[level], colors=[color], linewidths=1.2, linestyles="--")
                ax.clabel(cs, inline=True, fontsize=7, fmt=f"{level:.1f}")

    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
    ax.set_xlabel("Grid X (30m)", fontsize=8)
    ax.set_ylabel("Grid Y (30m)", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


# Main Tabs
tabs = st.tabs([
    "1. Overview",
    "2. Rainfall Forecast",
    "3. Hotspot & DSDI",
    "4. Zone Risk & Connectivity",
    "5. Epistemic Uncertainty",
    "6. Explainability & Causal Attention",
    "7. Counterfactual Scenarios (CSDN)",
])

# Tab 1: Overview
with tabs[0]:
    st.subheader("System Overview & Research Innovations")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Framework", "CSDN + WR-GNN", help="Counterfactual Sediment Dynamics Network")
    c2.metric("Target Resolution", "30m Spatial Grid")
    c3.metric("Modalities", "7 Encoders", help="Satellite, Terrain, Land-use, Flow, Rain, Graph, DSDI")
    c4.metric("Inference Paradigm", "Intervention-Aware")
    st.markdown("""
    ### Core Scientific Architecture
    - **Counterfactual Sediment Dynamics Network (CSDN)**: Models how sediment deposition changes under controlled meteorological & hydrological interventions (Rainfall $\\pm 10\\%, 20\\%, 30\\%$, Flow $\\pm 10\\%, 20\\%$, Inflow $\\pm 20\\%$).
    - **Dynamic Sediment Deposition Index (DSDI)**: Gated environmental prior fusing 11 normalized forcing and erosion variables.
    - **Watershed-to-Reservoir GNN (WR-GNN)**: Directed topological message passing encoding river networks and sediment transport paths.
    - **Physics-Informed Sediment Loss (PISL)**: Differentiable erosion supply and deposition drop constraints.
    - **Uncertainty-Aware Hotspot Prediction (UAHP)**: Monte Carlo Dropout confidence and sensitivity triage.
    """)

# Tab 2: Rainfall Forecast
with tabs[1]:
    st.subheader("Preserved Upstream Rainfall Forecasting Subsystem")
    st.caption("Live meteorological observations via Open-Meteo API • Gradient-Boosted Precipitation Regressor (XGBoost)")

    rain_p = REPO_ROOT / "rainfall_predictions.csv"

    act_c1, act_c2 = st.columns([3, 1])
    with act_c2:
        if st.button("🔄 Fetch Live Weather & Predict", use_container_width=True):
            with st.spinner("Fetching live Open-Meteo forecasts and predicting next-hour rainfall..."):
                try:
                    if str(REPO_ROOT) not in sys.path:
                        sys.path.insert(0, str(REPO_ROOT))
                    import river_weather_predictions
                    river_weather_predictions.run_live_predictions()
                    st.success("Successfully generated live predictions!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error executing prediction pipeline: {e}")

    if exists(rain_p):
        rf_df = pd.read_csv(rain_p)

        # Filters
        rivers = ["All River Basins"] + sorted(rf_df["river"].unique().tolist())
        with act_c1:
            selected_river = st.selectbox("Filter by River Basin", rivers)

        filtered_df = rf_df if selected_river == "All River Basins" else rf_df[rf_df["river"] == selected_river]

        # Metric Cards
        max_row = rf_df.loc[rf_df["predicted_rainfall_next_hour"].idxmax()]
        active_alerts = int((rf_df["alert"].fillna("") != "").sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Monitored Zones", f"{len(rf_df)} zones (3 Basins)")
        m2.metric("Peak Next-Hour Rain", f"{max_row['predicted_rainfall_next_hour']:.2f} mm", delta=max_row["zone"])
        m3.metric("Mean Regional Rain", f"{rf_df['predicted_rainfall_next_hour'].mean():.2f} mm/hr")
        m4.metric("Active Extreme Alerts", f"{active_alerts}", delta="Normal" if active_alerts == 0 else "Caution", delta_color="inverse" if active_alerts > 0 else "normal")

        # Bar chart of predicted rainfall
        st.markdown("#### Next-Hour Rainfall Forecast by Zone (mm)")
        chart_data = filtered_df.set_index("zone")[["current_rainfall", "predicted_rainfall_next_hour"]]
        chart_data.columns = ["Current Rain (mm)", "Predicted Next-Hour (mm)"]
        st.bar_chart(chart_data, height=320)

        # Zone Details Table
        st.markdown("#### Real-Time Zone Observation & Forecast Table")
        st.dataframe(
            filtered_df[[
                "timestamp", "river", "zone", "current_temp", "current_humidity",
                "current_rainfall", "predicted_rainfall_next_hour", "rainfall_class", "alert"
            ]],
            use_container_width=True,
        )
    else:
        st.warning("No rainfall predictions found. Click below or run `python river_weather_predictions.py`.")
        if st.button("▶️ Generate Real-Time Rainfall Predictions Now"):
            with st.spinner("Fetching live weather and running XGBoost model..."):
                try:
                    if str(REPO_ROOT) not in sys.path:
                        sys.path.insert(0, str(REPO_ROOT))
                    import river_weather_predictions
                    river_weather_predictions.run_live_predictions()
                    st.success("Predictions generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# Tab 3: Hotspots & DSDI
with tabs[2]:
    st.subheader("Spatial Hotspot Probability & DSDI Prior")
    risk = load_array(OUT / "future_sediment_hotspot_heatmap.npy", "Hotspot Heatmap")
    dsdi = load_array(OUT / "dsdi_heatmap.npy", "DSDI Prior Map")

    if risk is not None and dsdi is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = plot_heatmap(
                risk,
                "Predicted Sediment Deposition Hotspots f(X)",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                cbar_label="Hotspot Risk Probability [0 - 1]",
                contours=[0.5, 0.7],
                contour_colors=["#ffffff", "#ffd700"],
            )
            st.pyplot(fig1)
            plt.close(fig1)

        with col2:
            fig2 = plot_heatmap(
                dsdi,
                "Dynamic Sediment Deposition Index (DSDI Prior)",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                cbar_label="DSDI Value [0 - 1]",
                contours=[0.5],
                contour_colors=["#ffffff"],
            )
            st.pyplot(fig2)
            plt.close(fig2)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Hotspot Risk", f"{risk.mean():.4f}")
        m2.metric("Peak Hotspot Risk", f"{risk.max():.4f}")
        m3.metric("High-Risk Area (≥ 0.5)", f"{(risk >= 0.5).mean() * 100:.1f}%")
        m4.metric("Mean DSDI Prior", f"{dsdi.mean():.4f}")

# Tab 4: Zone Risk
with tabs[3]:
    st.subheader("Zone-Wise Sedimentation Risk")
    zone_p = OUT / "zone_sediment_risk.csv"
    if exists(zone_p):
        z_df = pd.read_csv(zone_p)
        st.bar_chart(z_df.set_index("zone")["risk"])
        st.dataframe(z_df, use_container_width=True)
    else:
        st.info("Run inference pipeline to generate zone risk profile.")

# Tab 5: Uncertainty
with tabs[4]:
    st.subheader("Epistemic Uncertainty & Priority Triage")
    unc = load_array(OUT / "uncertainty_heatmap.npy", "Uncertainty Heatmap")
    risk_base = load_array(OUT / "future_sediment_hotspot_heatmap.npy", "Risk Heatmap")

    if unc is not None and risk_base is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig_unc = plot_heatmap(
                unc,
                "MC Dropout Variance σ² (Epistemic Uncertainty)",
                cmap="magma",
                cbar_label="Predictive Variance σ²",
            )
            st.pyplot(fig_unc)
            plt.close(fig_unc)

        with col2:
            cutoff = float(np.quantile(unc, 0.75))
            high_risk = risk_base >= 0.5
            high_unc = unc >= cutoff

            triage_rgb = np.zeros((*risk_base.shape, 3))
            triage_rgb[~high_risk] = [0.12, 0.42, 0.78]            # Low Risk (Blue)
            triage_rgb[high_risk & ~high_unc] = [0.88, 0.18, 0.12] # High Risk / High Conf (Red)
            triage_rgb[high_risk & high_unc] = [1.0, 0.70, 0.08]  # High Risk / Low Conf (Amber)

            fig_tri, ax_tri = plt.subplots(figsize=(4.8, 4.2), dpi=120)
            ax_tri.imshow(triage_rgb, origin="upper", aspect="equal")
            ax_tri.set_title("Decision Triage Map", fontsize=9, fontweight="bold", pad=8)
            ax_tri.set_xlabel("Grid X (30m)", fontsize=8)
            ax_tri.set_ylabel("Grid Y (30m)", fontsize=8)
            ax_tri.tick_params(labelsize=7)

            patches = [
                mpatches.Patch(color=[0.88, 0.18, 0.12], label="High Risk / High Confidence"),
                mpatches.Patch(color=[1.0, 0.70, 0.08], label="High Risk / Low Conf (Priority Monitoring)"),
                mpatches.Patch(color=[0.12, 0.42, 0.78], label="Low Risk (Stable)"),
            ]
            ax_tri.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.9)
            plt.tight_layout()
            st.pyplot(fig_tri)
            plt.close(fig_tri)

        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Mean Epistemic Variance", f"{unc.mean():.6f}")
        u2.metric("Max Epistemic Variance", f"{unc.max():.6f}")
        u3.metric("High-Conf Hotspot Area", f"{(high_risk & ~high_unc).mean() * 100:.1f}%")
        u4.metric("Priority Monitoring Area", f"{(high_risk & high_unc).mean() * 100:.1f}%")

# Tab 6: Explainability
with tabs[5]:
    st.subheader("Causal Attention & Factor Attributions")
    attn_p = OUT / "modality_attention.csv"
    if exists(attn_p):
        attn_df = pd.read_csv(attn_p)
        st.bar_chart(attn_df.set_index("modality")[["learned_attention", "causal_attention", "fusion_weight"]])
        st.dataframe(attn_df, use_container_width=True)
    else:
        st.info("Run inference pipeline to generate modality attention weights.")

# Tab 7: Counterfactual Scenarios (CSDN)
with tabs[6]:
    st.subheader("Counterfactual Sediment Dynamics Network (CSDN)")
    st.markdown("Evaluate spatial sediment risk response under controlled hydrological and meteorological interventions:")

    SCENARIO_OPTIONS = [
        ("Rainfall +10%", "rainfall_plus10", 0.10, "Rainfall"),
        ("Rainfall +20%", "rainfall_plus20", 0.20, "Rainfall"),
        ("Rainfall +30%", "rainfall_plus30", 0.30, "Rainfall"),
        ("River Flow +10%", "flow_plus10", 0.10, "River Flow"),
        ("River Flow -10%", "flow_minus10", -0.10, "River Flow"),
        ("River Flow -20%", "flow_minus20", -0.20, "River Flow"),
        ("Reservoir Inflow +20%", "inflow_plus20", 0.20, "Reservoir Inflow"),
        ("Reservoir Inflow -20%", "inflow_minus20", -0.20, "Reservoir Inflow"),
    ]

    selected_label = st.selectbox(
        "Select Intervention Scenario",
        [opt[0] for opt in SCENARIO_OPTIONS],
        index=1,
    )
    sel_opt = next(opt for opt in SCENARIO_OPTIONS if opt[0] == selected_label)
    sc_name, sc_mag, sc_type = sel_opt[1], sel_opt[2], sel_opt[3]

    base_p = OUT / "counterfactual_baseline.npy"
    if not exists(base_p):
        base_p = OUT / "future_sediment_hotspot_heatmap.npy"

    cf_p = OUT / f"counterfactual_{sc_name}.npy"
    diff_p = DIFF_DIR / f"diff_{sc_name}.npy"
    sens_p = SENS_DIR / f"sens_{sc_name}.npy"

    base_map = load_array(base_p, "Baseline Heatmap")
    cf_map = load_array(cf_p, f"Counterfactual Map ({selected_label})")
    diff_map = load_array(diff_p, f"Difference Map ({selected_label})")
    sens_map = load_array(sens_p, f"Sensitivity Map ({selected_label})")

    if all(m is not None for m in [base_map, cf_map, diff_map, sens_map]):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            fig_b = plot_heatmap(
                base_map,
                "1. Baseline Heatmap H = f(X)",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                cbar_label="Risk Score [0 - 1]",
                contours=[0.5, 0.7],
                contour_colors=["#ffffff", "#ffd700"],
            )
            st.pyplot(fig_b)
            plt.close(fig_b)

        with c2:
            fig_cf = plot_heatmap(
                cf_map,
                f"2. Counterfactual H_cf ({selected_label})",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                cbar_label="Risk Score [0 - 1]",
                contours=[0.5, 0.7],
                contour_colors=["#ffffff", "#ffd700"],
            )
            st.pyplot(fig_cf)
            plt.close(fig_cf)

        with c3:
            diff_abs_max = max(0.02, float(max(abs(diff_map.min()), abs(diff_map.max()))))
            fig_d = plot_heatmap(
                diff_map,
                "3. Risk Change ΔH = H_cf − H",
                cmap="RdBu_r",
                vmin=-diff_abs_max,
                vmax=diff_abs_max,
                cbar_label="Δ Risk (Red: +Risk, Blue: -Risk)",
            )
            st.pyplot(fig_d)
            plt.close(fig_d)

        with c4:
            fig_s = plot_heatmap(
                sens_map,
                "4. Spatial Sensitivity |ΔH| / |Δx|",
                cmap="magma",
                vmin=0.0,
                vmax=float(sens_map.max()) if sens_map.max() > 0 else 1.0,
                cbar_label="Sensitivity |ΔH| / |Δx|",
            )
            st.pyplot(fig_s)
            plt.close(fig_s)

        st.markdown("#### Scenario Quantitative Diagnostics")
        sum1, sum2, sum3, sum4 = st.columns(4)
        with sum1:
            st.markdown("**Baseline State**")
            st.write(f"• Mean Risk: `{base_map.mean():.4f}`")
            st.write(f"• Peak Risk: `{base_map.max():.4f}`")
            st.write(f"• Area ≥ 0.5: `{(base_map >= 0.5).mean() * 100:.2f}%`")
        with sum2:
            st.markdown(f"**Counterfactual State ({selected_label})**")
            st.write(f"• Mean Risk: `{cf_map.mean():.4f}`")
            st.write(f"• Peak Risk: `{cf_map.max():.4f}`")
            st.write(f"• Area ≥ 0.5: `{(cf_map >= 0.5).mean() * 100:.2f}%`")
        with sum3:
            st.markdown("**Risk Change (ΔH)**")
            st.write(f"• Mean ΔH: `{diff_map.mean():+.4f}`")
            st.write(f"• Max Increase: `{diff_map.max():+.4f}`")
            st.write(f"• Max Decrease: `{diff_map.min():+.4f}`")
        with sum4:
            st.markdown("**Spatial Sensitivity**")
            st.write(f"• Mean Sensitivity: `{sens_map.mean():.4f}`")
            st.write(f"• Max Sensitivity: `{sens_map.max():.4f}`")
            st.write(f"• Intervention Mag |Δx|: `{abs(sc_mag):.2f}`")

        st.markdown("#### Zone Vulnerability & Sensitivity Ranking")
        num_zones = 8
        strip_h = base_map.shape[0] // num_zones
        zone_rows = []
        for i in range(num_zones):
            z_name = f"Zone {chr(65 + i)}"
            z_slice = slice(i * strip_h, (i + 1) * strip_h)
            b_risk = float(base_map[z_slice, :].mean())
            c_risk = float(cf_map[z_slice, :].mean())
            d_risk = float(diff_map[z_slice, :].mean())
            s_val = float(sens_map[z_slice, :].mean())
            zone_rows.append({
                "Zone": z_name,
                "Baseline Risk": round(b_risk, 4),
                "Counterfactual Risk": round(c_risk, 4),
                "Risk Change (ΔH)": round(d_risk, 4),
                "Mean Sensitivity": round(s_val, 4),
            })
        z_ranking_df = pd.DataFrame(zone_rows).sort_values("Mean Sensitivity", ascending=False).reset_index(drop=True)
        z_ranking_df["Rank"] = z_ranking_df.index + 1
        q75 = z_ranking_df["Mean Sensitivity"].quantile(0.75)
        q25 = z_ranking_df["Mean Sensitivity"].quantile(0.25)
        z_ranking_df["Vulnerability Category"] = z_ranking_df["Mean Sensitivity"].apply(
            lambda s: "Highly Sensitive" if s >= q75 else ("Moderately Sensitive" if s >= q25 else "Stable")
        )
        st.dataframe(
            z_ranking_df[["Rank", "Zone", "Baseline Risk", "Counterfactual Risk", "Risk Change (ΔH)", "Mean Sensitivity", "Vulnerability Category"]],
            use_container_width=True,
        )

    cf_res_p = OUT / "counterfactual_results.csv"
    if exists(cf_res_p):
        st.markdown("#### Multi-Scenario Systematic Benchmark")
        st.dataframe(pd.read_csv(cf_res_p), use_container_width=True)
