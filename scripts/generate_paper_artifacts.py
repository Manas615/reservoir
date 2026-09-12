from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sediment_hotspot.utils.io import ensure_dir

def generate_paper_artifacts(results_dir: str = "results", outputs_dir: str = "data/outputs", paper_dir: str = "results/paper"):
    res_d = Path(results_dir)
    out_d = Path(outputs_dir)
    p_d = ensure_dir(paper_dir)

    print("Generating Paper Tables & Figures...")

    # Load result dataframes
    b_df = pd.read_csv(res_d / "baseline_results.csv") if (res_d / "baseline_results.csv").exists() else pd.DataFrame()
    a_df = pd.read_csv(res_d / "ablation_results.csv") if (res_d / "ablation_results.csv").exists() else pd.DataFrame()
    cf_df = pd.read_csv(out_d / "counterfactual_results.csv") if (out_d / "counterfactual_results.csv").exists() else pd.DataFrame()
    z_df = pd.read_csv(out_d / "zone_sensitivity.csv") if (out_d / "zone_sensitivity.csv").exists() else pd.DataFrame()

    # 1. Table 1: Dataset Statistics
    t1 = pd.DataFrame({
        "Modality": ["Satellite (Sentinel-2 / NDTI)", "DEM Morphometry (SRTM)", "Land-Use (ESA WorldCover)", "Flow Sequences", "Rainfall Forcing", "Graph Connectivity", "DSDI Prior"],
        "Dimensions / Shape": ["[6, 128, 128]", "[4, 128, 128]", "[5, 128, 128]", "[10, 8]", "[4]", "[5, 8]", "[11]"],
        "Temporal Frequency": ["5–10 days", "Static", "Annual / Static", "Hourly / Daily", "Hourly Forecast", "Topological", "Dynamic Gated"],
        "Provenance": ["DERIVED_FROM_SATELLITE", "PHYSICS_DERIVED", "DERIVED_FROM_SATELLITE", "REAL_OBSERVATION", "REAL_OBSERVATION", "PHYSICS_DERIVED", "PHYSICS_DERIVED"],
    })
    t1.to_csv(p_d / "table1_dataset_statistics.csv", index=False)
    (p_d / "table1_dataset_statistics.tex").write_text(t1.to_latex(index=False), encoding="utf-8")

    # 2. Table 2: Baseline Comparison
    if not b_df.empty:
        b_df.to_csv(p_d / "table2_baseline_comparison.csv", index=False)
        (p_d / "table2_baseline_comparison.tex").write_text(b_df.to_latex(index=False), encoding="utf-8")

    # 3. Table 3: Ablation Study
    if not a_df.empty:
        a_df.to_csv(p_d / "table3_ablation_study.csv", index=False)
        (p_d / "table3_ablation_study.tex").write_text(a_df.to_latex(index=False), encoding="utf-8")

    # 4. Table 4: Counterfactual Scenario Results
    if not cf_df.empty:
        cf_df.to_csv(p_d / "table4_counterfactual_scenarios.csv", index=False)
        (p_d / "table4_counterfactual_scenarios.tex").write_text(cf_df.to_latex(index=False), encoding="utf-8")

    # 5. Table 5: Zone Sensitivity Ranking
    if not z_df.empty:
        z_df.to_csv(p_d / "table5_zone_sensitivity.csv", index=False)
        (p_d / "table5_zone_sensitivity.tex").write_text(z_df.to_latex(index=False), encoding="utf-8")

    # Generate Publication Figures using matplotlib
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 2: DSDI Heatmap
    if (out_d / "dsdi_heatmap.npy").exists():
        dsdi = np.load(out_d / "dsdi_heatmap.npy")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(dsdi, cmap="magma", vmin=0, vmax=1)
        ax.set_title("Dynamic Sediment Deposition Index (DSDI)", fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(p_d / "fig2_dsdi_heatmap.png", dpi=300)
        plt.close()

    # Figure 3: Predicted Hotspots
    if (out_d / "future_sediment_hotspot_heatmap.npy").exists():
        heat = np.load(out_d / "future_sediment_hotspot_heatmap.npy")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(heat, cmap="inferno", vmin=0, vmax=1)
        ax.set_title("Predicted Sediment Deposition Hotspots f(X)", fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(p_d / "fig3_predicted_hotspots.png", dpi=300)
        plt.close()

    # Figure 4: Counterfactual Difference Map
    diff_p = out_d / "counterfactual_difference_maps" / "diff_rainfall_plus20.npy"
    if diff_p.exists():
        diff = np.load(diff_p)
        fig, ax = plt.subplots(figsize=(6, 5))
        vmax = max(abs(diff.min()), abs(diff.max()), 0.05)
        im = ax.imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title("Counterfactual Risk Shift: Rainfall +20% (ΔH)", fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(p_d / "fig5_counterfactual_diff_rainfall20.png", dpi=300)
        plt.close()

    # Figure 6: Spatial Sensitivity Map
    sens_p = out_d / "sensitivity_maps" / "sens_rainfall_plus20.npy"
    if sens_p.exists():
        sens = np.load(sens_p)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(sens, cmap="viridis")
        ax.set_title("Spatial Sediment Sensitivity S_i (|ΔH| / |Δx|)", fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(p_d / "fig6_spatial_sensitivity.png", dpi=300)
        plt.close()

    # Figure 7: Uncertainty Map
    if (out_d / "uncertainty_heatmap.npy").exists():
        unc = np.load(out_d / "uncertainty_heatmap.npy")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(unc, cmap="plasma")
        ax.set_title("MC-Dropout Epistemic Uncertainty Map", fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(p_d / "fig7_uncertainty_map.png", dpi=300)
        plt.close()

    print(f"Successfully generated all publication tables and figures in {p_d}")

if __name__ == "__main__":
    generate_paper_artifacts()
