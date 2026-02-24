"""
visualisation.py — WinDrift++ Visualisations
=============================================
Produces charts for WinDrift++ experiment results:

  Fig 7 : Dataset overview (distributions per dataset)
  Fig 8 : ECDF comparison plots (Figures 8a-8d in paper)
  Fig 9 : Dataset distribution plots (C11-C17)
  Fig 5 : Performance comparison bar chart
  Custom: Drift flag heatmaps, per-test contribution, voting comparison
"""


# **** Shaping up provided code with parameters and correct indentation ****
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, sys
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from config import PLOTS_DIR, MAX_CYC_LEN


plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi":   100,
})


# ************************************************************
# Figure 7-style: Dataset overview C1-C10
# ************************************************************

def plot_dataset_means(datasets: dict, metadata: dict) -> None:
    """
    Plot monthly distribution means for C1-C5 (similar) and C6-C10 (dissimilar)
    replicating Figure 7 of the paper.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    month_labels = [f"M{i:02d}" for i in range(1, 25)]
    x_pos = list(range(1, 25))

    groups = [
        (["C1", "C2", "C3", "C4", "C5"], "Similar data patterns C1–C5", axes[0]),
        (["C6", "C7", "C8", "C9", "C10"], "Dissimilar data patterns C6–C10", axes[1]),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for cids, title, ax in groups:
        for cid, col in zip(cids, colors):
            df  = datasets[cid]
            means = df.mean()
            ax.plot(x_pos, means.values, marker="o", markersize=4,
                    linewidth=1.5, color=col, label=cid, alpha=0.85)

        ax.axvline(x=12.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(month_labels, rotation=45, fontsize=7)
        ax.set_xlabel("Time period (Months)")
        ax.set_ylabel("Distribution Mean")
        ax.set_title(title, fontweight="bold")
        ax.legend(ncol=5, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.text(6.5, ax.get_ylim()[0], "Y2019", ha="center", fontsize=8, color="gray")
        ax.text(18.5, ax.get_ylim()[0], "Y2020", ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig7_dataset_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")


# ************************************************************
# Figure 8-style: ECDF comparison plots (similar and dissimilar)
# ************************************************************

def plot_ecdf_comparisons(datasets: dict) -> None:
    """
    Reproduce Figure 8 of the paper:
    (a) C1-C5 Mode I: Jan2019 vs Feb2019
    (b) C1-C5 Mode II: Jan2019 vs Jan2020
    (c) C6-C10 Mode I: Jan2019 vs Feb2019
    (d) C6-C10 Mode II: Jan2019 vs Jan2020
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    cases = [
        ("C1",  0,   1,  "C1–C5 Mode I: Jan19 vs Feb19 (consecutive)",        axes[0, 0]),
        ("C1",  0,  12,  "C1–C5 Mode II: Jan19 vs Jan20 (corresponding)",      axes[0, 1]),
        ("C6",  0,   1,  "C6–C10 Mode I: Jan19 vs Feb19 (consecutive)",        axes[1, 0]),
        ("C6",  0,  12,  "C6–C10 Mode II: Jan19 vs Jan20 (corresponding)",     axes[1, 1]),
    ]

    for cid, col_h, col_n, title, ax in cases:
        df  = datasets[cid]
        x_h = df.iloc[:, col_h].values.astype(float)
        x_n = df.iloc[:, col_n].values.astype(float)
        merged = np.sort(np.concatenate([x_h, x_n]))
        ecdf_h = np.array([stats.percentileofscore(x_h, v, kind="weak") / 100 for v in merged])
        ecdf_n = np.array([stats.percentileofscore(x_n, v, kind="weak") / 100 for v in merged])
        abs_diff = np.abs(ecdf_h - ecdf_n)
        d_stat = float(np.max(abs_diff))
        d_crit = 1.36 * np.sqrt(1 / len(x_h) + 1 / len(x_n))
        d_idx  = int(np.argmax(abs_diff))

        ax.plot(merged, ecdf_h, label="ECDF of $W_H$", color="#2196F3", linewidth=2)
        ax.plot(merged, ecdf_n, label="ECDF of $W_N$", color="#4CAF50", linewidth=2,
                linestyle="--", marker="^", markersize=4, markevery=5)
        # Red line at max deviation
        ax.vlines(merged[d_idx], ecdf_h[d_idx], ecdf_n[d_idx],
                  colors="red", linewidth=2.5, label=f"Δ = {d_stat:.3f}")

        drift = d_stat >= d_crit
        status = "Δ > ε  (DRIFT)" if drift else "Δ < ε  (No Drift)"
        ax.set_title(f"{title}\n{status}", fontweight="bold",
                     color="red" if drift else "green")
        ax.set_xlabel("x"); ax.set_ylabel("$F^{HIST}(x) - F^{NEW}(x)$")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle("Figure 8: WinDrift++ ECDF comparisons on synthetic datasets",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig8_ecdf_comparisons.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")


# ************************************************************
# Figure 9-style: C11-C17 distribution shapes
# ************************************************************

def plot_c11_c17_distributions(datasets: dict) -> None:
    """
    Plot reference distribution (Y2019) vs changed distribution (Y2020)
    for each of C11-C17, replicating Figure 9.
    """
    configs = [
        ("C11", "Chi-sq(k=4)", "Skewness*"),
        ("C12", "Normal(0,10)", "Shape*"),
        ("C13", "Normal(3,1)",  "Location*"),
        ("C14", "Normal(0,0.5)","Light tail**"),
        ("C15", "Normal(0,2)",  "Heavy tail**"),
        ("C16", "Rayleigh(10)", "Outliers**"),
        ("C17", "Laplace(0,1)", "Ties**"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    # Ref distribution plot (Normal(0,1))
    ax0 = axes[0]
    x = np.linspace(-4, 4, 300)
    ax0.plot(x, stats.norm.pdf(x, 0, 1), color="#1565C0", linewidth=2)
    ax0.set_title("Y2019 Reference\nNormal(µ=0, σ=1)", fontweight="bold")
    ax0.set_xlabel("x"); ax0.set_ylabel("P(X=x)")
    ax0.grid(True, alpha=0.3)

    for idx, (cid, dist_label, change_type) in enumerate(configs):
        ax = axes[idx + 1]
        df = datasets[cid]
        y20_data = df.iloc[:, 12:].values.flatten()
        ax.hist(y20_data, bins=20, density=True, color="#FF6F00",
                alpha=0.6, edgecolor="white", linewidth=0.5)
        ax.set_title(f"Y2020 — {cid}\n{dist_label} ({change_type})", fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("P(X=x)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_visible(False)
    plt.suptitle("Figure 9: Synthetic datasets C11–C17 distribution shapes",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig9_c11_c17_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")


# ************************************************************
# Drift flag heatmap across all datasets
# ************************************************************

def plot_drift_heatmap(all_results: dict, metadata: dict) -> None:
    """
    Heatmap: dataset (rows) × comparison pair (cols), colour = majority_flag.
    Mode II (Corresponding) only, win_level=1 for monthly granularity.
    """
    rows = []
    for cid, df_res in all_results.items():
        sub = df_res[(df_res["mode"] == 2) & (df_res["win_level"] == 1)]
        for _, r in sub.iterrows():
            rows.append({
                "dataset": cid,
                "comparison": f"M{int(r.dbcount)-MAX_CYC_LEN:02d}→M{int(r.dbcount):02d}",
                "flag": int(r["majority_flag"])
            })
    if not rows:
        return

    pivot = pd.DataFrame(rows).pivot_table(
        index="dataset", columns="comparison", values="flag", aggfunc="first"
    )
    pivot = pivot.reindex(sorted(pivot.index,
                                  key=lambda x: int(x[1:]) if x[1:].isdigit() else 99))

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = sns.color_palette(["#E8F5E9", "#C62828"], as_cmap=False)
    sns.heatmap(
        pivot.astype(float), ax=ax,
        cmap=["#E8F5E9", "#C62828"],
        annot=True, fmt=".0f",
        linewidths=0.5, linecolor="gray",
        cbar_kws={"label": "1=Drift / 0=No Drift"},
        vmin=0, vmax=1
    )
    ax.set_title("WinDrift++ Drift Detection Results — Mode II (Corresponding, W=1)\n"
                 "Monthly year-on-year comparisons across all synthetic datasets",
                 fontweight="bold")
    ax.set_xlabel("Year-on-year comparison (Y2019 → Y2020)")
    ax.set_ylabel("Dataset")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "drift_heatmap_mode2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")


# ************************************************************
# Per-test contribution (Ablation study — Table X)
# ************************************************************

def plot_test_contributions(all_results: dict) -> None:
    """
    Stacked bar: per dataset, per test — how many Mode II comparisons flagged drift.
    Reproduces the spirit of Table X (ablation) and Figure 3b.
    """
    tests = ["ks_flag", "kuiper_flag", "cvm_flag", "ad_flag", "emd_flag"]
    test_labels = ["KS", "Kuiper", "CVM", "AD", "EMD"]
    colors = ["#1565C0", "#6A1B9A", "#2E7D32", "#E65100", "#37474F"]

    cids  = sorted(all_results.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 99)
    drift_counts = {t: [] for t in test_labels}
    majority_counts = []

    for cid in cids:
        sub = all_results[cid][all_results[cid]["mode"] == 2]
        for t, tl in zip(tests, test_labels):
            drift_counts[tl].append(int(sub[t].sum()))
        majority_counts.append(int(sub["majority_flag"].sum()))

    x = np.arange(len(cids))
    width = 0.14
    fig, ax = plt.subplots(figsize=(16, 6))

    for i, (tl, col) in enumerate(zip(test_labels, colors)):
        offset = (i - 2) * width
        ax.bar(x + offset, drift_counts[tl], width, label=tl, color=col, alpha=0.8)

    # Majority line
    ax.plot(x, majority_counts, "k^--", markersize=7, linewidth=1.5,
            label="WD++ Majority", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(cids, fontsize=9)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("# Drift flags (Mode II)")
    ax.set_title("Individual Test Contributions vs. WD++ Majority Vote\n"
                 "(Mode II — Corresponding comparisons)",
                 fontweight="bold")
    ax.legend(ncol=6, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "test_contributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")


# ************************************************************
# Performance comparison bar chart (Figure 5 style)
# ************************************************************

def plot_performance_comparison(summary_df: pd.DataFrame) -> None:
    """
    Horizontal bar chart comparing WD++ accuracy across all 17 datasets
    by experiment group. Styled after Figure 5 in the paper.
    """
    if "dataset_id" not in summary_df.columns:
        return

    sub = summary_df[summary_df["Acc"] != "-"].copy()
    if sub.empty:
        return

    sub["Acc_num"] = pd.to_numeric(sub["Acc"], errors="coerce")
    sub = sub.dropna(subset=["Acc_num"])

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#1565C0"] * len(sub)

    bars = ax.barh(sub["dataset_id"], sub["Acc_num"], color=colors, alpha=0.8, edgecolor="white")
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Accuracy")
    ax.set_title("WinDrift++ Detection Accuracy per Dataset\n"
                 "(Red = drift dataset, Green = no-drift dataset)",
                 fontweight="bold")
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, sub["Acc_num"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "performance_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")


# ************************************************************
# Voting methods comparison (Figure 6 style)
# ************************************************************

def plot_voting_comparison(voting_df: pd.DataFrame) -> None:
    """
    Grouped bar chart: accuracy per dataset per voting method.
    Styled after Figure 6 in the paper.
    """
    if voting_df is None or voting_df.empty:
        return

    methods = ["majority", "plurality", "borda", "elimination", "pairwise"]
    method_labels = ["Majority", "Plurality", "Borda Count", "Elimination", "Pairwise"]
    colors = ["#1565C0", "#6A1B9A", "#2E7D32", "#E65100", "#37474F"]

    cids = sorted(voting_df["dataset_id"].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 99)
    x = np.arange(len(cids))
    width = 0.15

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (m, ml, col) in enumerate(zip(methods, method_labels, colors)):
        sub = voting_df[voting_df["method"] == m]
        vals = [float(sub[sub["dataset_id"] == c]["accuracy"].values[0])
                if c in sub["dataset_id"].values else 0.0 for c in cids]
        ax.bar(x + (i - 2) * width, vals, width, label=ml, color=col, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(cids, fontsize=8)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")
    ax.set_title("WD++ Performance with Different Voting Methods\n",
                 fontweight="bold")
    ax.legend(ncol=5, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "voting_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")
