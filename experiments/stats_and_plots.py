"""Stage 6a (Basma): Statistical Analysis & Publication Plots.

Reads Stage 4 (results/stage4/*.csv) and Stage 5a (results/stage5a/*.csv) outputs,
prints key summary statistics, and saves 5 figures (with 95% confidence intervals)
into report/figures/.

Run from the project root:
    python experiments/stats_and_plots.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STAGE4_DIR = ROOT / "results" / "stage4"
STAGE5A_DIR = ROOT / "results" / "stage5a"
FIG_DIR = ROOT / "report" / "figures"

plt.rcParams.update({
    "figure.dpi": 180,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_fig1_small_grids(df_a: pd.DataFrame) -> None:
    """Bar chart of Mean Return (+/- 95% CI) across Suite A small-grid scenarios."""
    scenarios = list(df_a["scenario_id"].unique())
    agents = ["NoOp", "B0-Random", "B1-NearestFront", "B2-Greedy", "MCTS(N=150)", "Exact"]
    agents = [a for a in agents if a in set(df_a["agent"])]

    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(agents))

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, ag in enumerate(agents):
        sub = df_a[df_a["agent"] == ag].set_index("scenario_id").reindex(scenarios)
        offset = (i - (len(agents) - 1) / 2) * width
        ax.bar(
            x + offset,
            sub["return_mean"],
            width=width,
            yerr=sub["return_ci"],
            capsize=2.5,
            label=ag,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=8)
    ax.set_ylabel("Mean Episode Return (+/- 95% CI)")
    ax.set_title("Small-Grid Performance vs. Exact Solver (Suite A)")
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_small_grid_comparison.png")
    plt.close(fig)


def plot_fig2_budget_sweep(df_b: pd.DataFrame) -> None:
    """Optimality Gap (3x3) and Return (5x5) vs. MCTS Rollout Budget N."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    def extract_mcts(df_sub: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in df_sub.iterrows():
            m = re.match(r"MCTS\(N=(\d+),(\w+)\)", str(r["agent"]))
            if m:
                rows.append({
                    "N": int(m.group(1)),
                    "policy": m.group(2),
                    "return_mean": r["return_mean"],
                    "return_ci": r["return_ci"],
                    "gap_mean": r["optimality_gap_mean"],
                    "gap_ci": r["optimality_gap_ci"],
                })
        return pd.DataFrame(rows).sort_values("N")

    # Left panel: 3x3 Optimality Gap vs N
    sub_3 = df_b[df_b["scenario_id"] == "3x3_budget_sweep"]
    mcts_3 = extract_mcts(sub_3)
    for pol, style in (("random", "--o"), ("nearest", "-s")):
        d = mcts_3[mcts_3["policy"] == pol]
        if not d.empty:
            ax1.errorbar(d["N"], d["gap_mean"], yerr=d["gap_ci"], fmt=style, capsize=4, label=f"MCTS ({pol})")
    ax1.set_xscale("log")
    ax1.set_xlabel("MCTS Rollout Budget N (log scale)")
    ax1.set_ylabel("Optimality Gap  V*(s) - Q*(s, a) (+/- 95% CI)")
    ax1.set_title("3x3 Optimality Gap vs. Rollout Budget")
    ax1.legend()

    # Right panel: 5x5 Mean Return vs N
    sub_5 = df_b[df_b["scenario_id"] == "5x5_budget_sweep"]
    mcts_5 = extract_mcts(sub_5)
    for pol, style in (("random", "--o"), ("nearest", "-s")):
        d = mcts_5[mcts_5["policy"] == pol]
        if not d.empty:
            ax2.errorbar(d["N"], d["return_mean"], yerr=d["return_ci"], fmt=style, capsize=4, label=f"MCTS ({pol})")

    for base_name, ls in (("B1-NearestFront", ":"), ("B2-Greedy", "-.")):
        b = sub_5[sub_5["agent"] == base_name]
        if not b.empty:
            ax2.axhline(float(b["return_mean"].iloc[0]), linestyle=ls, color="gray", label=base_name)

    ax2.set_xscale("log")
    ax2.set_xlabel("MCTS Rollout Budget N (log scale)")
    ax2.set_ylabel("Mean Episode Return (+/- 95% CI)")
    ax2.set_title("5x5 Return vs. Rollout Budget")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_optimality_gap_vs_budget.png")
    plt.close(fig)


def plot_fig3_scalability(df_c: pd.DataFrame) -> None:
    """Grid-size scalability (k=1): Mean Return and Decision Latency (log ms)."""
    sub = df_c[df_c["scenario_id"].str.endswith("_k1")].copy()
    sub["grid_size"] = sub["scenario_id"].str.extract(r"^(\d+)x").astype(int)
    sub = sub.sort_values("grid_size")

    agents = ["NoOp", "B0-Random", "B1-NearestFront", "B2-Greedy", "MCTS(N=100)"]
    agents = [a for a in agents if a in set(sub["agent"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ag in agents:
        d = sub[sub["agent"] == ag]
        ax1.errorbar(d["grid_size"], d["return_mean"], yerr=d["return_ci"], marker="o", capsize=3, label=ag)
        # Replace 0 ms with tiny floor for log scale
        t_vals = np.maximum(d["time_per_decision_ms_mean"], 1e-3)
        ax2.plot(d["grid_size"], t_vals, marker="o", label=ag)

    sizes = sorted(sub["grid_size"].unique())
    for ax in (ax1, ax2):
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s}x{s}" for s in sizes])
        ax.set_xlabel("Grid Size (k = 1, East Wind)")

    ax1.set_ylabel("Mean Episode Return (+/- 95% CI)")
    ax1.set_title("Decision Quality Across Grid Sizes")
    ax1.legend(fontsize=8)

    ax2.set_yscale("log")
    ax2.set_ylabel("Time per Decision (ms, log scale)")
    ax2.set_title("Decision Latency Scaling")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_scalability_return_and_time.png")
    plt.close(fig)


def plot_fig4_robustness(df_5a: pd.DataFrame) -> None:
    """Model-mismatch robustness on 3x3 and 5x5 grids (Stage 5a)."""
    conditions = [
        "correct", "belief_wind_N", "belief_wind_W", "belief_wind_none",
        "belief_spread_x0.7", "belief_spread_x1.3", "shift_aware", "shift_unaware",
    ]
    labels = ["Correct", "Wind N", "Wind W", "No Wind", "Spread x0.7", "Spread x1.3", "Shift Aware", "Shift Unaware"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, group in zip(axes, ("3x3_mixed_E", "5x5_mixed_E")):
        sub = df_5a[df_5a["scenario_id"].str.startswith(group + "__")].copy()
        sub["cond"] = sub["scenario_id"].str.split("__").str[1]
        sub["agent_clean"] = sub["agent"].str.replace(r"\(.*\)", "", regex=True).str.strip()

        for ag in ("B1-NearestFront", "B2-Greedy", "MCTS", "Exact"):
            d = sub[sub["agent_clean"] == ag].set_index("cond").reindex(conditions)
            if d["return_mean"].notna().any():
                ax.errorbar(
                    np.arange(len(conditions)),
                    d["return_mean"],
                    yerr=d["return_ci"],
                    marker="o",
                    capsize=3,
                    label=ag,
                )

        ax.set_xticks(np.arange(len(conditions)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Mean Return (+/- 95% CI)")
        ax.set_title(f"Robustness under Model Mismatch ({group})")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_robustness_mismatch.png")
    plt.close(fig)


def plot_fig5_spatial_cost(df_c: pd.DataFrame) -> None:
    """Spatial cost: MCTS tree nodes created per decision across grid sizes."""
    sub = df_c[df_c["agent"].str.startswith("MCTS")].copy()
    sub["grid_size"] = sub["scenario_id"].str.extract(r"^(\d+)x").astype(int)
    sub["k"] = sub["scenario_id"].str.extract(r"_k(\d+)$").astype(int)
    sub = sub.sort_values("grid_size")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k_val in sorted(sub["k"].unique()):
        d = sub[sub["k"] == k_val]
        ax.errorbar(
            d["grid_size"], d["tree_nodes_mean"], yerr=d["tree_nodes_ci"],
            marker="o", capsize=4, label=f"MCTS(N=100), k={k_val}"
        )

    sizes = sorted(sub["grid_size"].unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Tree Nodes per Decision (+/- 95% CI)")
    ax.set_title("MCTS Spatial Cost (Tree Size vs. Grid Size)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_spatial_cost_tree_nodes.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df_a = pd.read_csv(STAGE4_DIR / "suite_a_small_grid_summary.csv")
    df_b = pd.read_csv(STAGE4_DIR / "suite_b_budget_sweep_summary.csv")
    df_c = pd.read_csv(STAGE4_DIR / "suite_c_scalability_summary.csv")
    df_5a = pd.read_csv(STAGE5A_DIR / "stage5a_robustness_summary.csv")

    plot_fig1_small_grids(df_a)
    plot_fig2_budget_sweep(df_b)
    plot_fig3_scalability(df_c)
    plot_fig4_robustness(df_5a)
    plot_fig5_spatial_cost(df_c)

    print("Generated 5 figures in:", FIG_DIR)
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")

    # Print quick statistical highlight table for report/slides
    print("\n--- Stage 6a Summary Highlights (Mean +/- 95% CI) ---")
    b3 = df_b[df_b["scenario_id"] == "3x3_budget_sweep"][
        ["agent", "return_mean", "return_ci", "optimality_gap_mean", "optimality_gap_ci", "time_per_decision_ms_mean"]
    ]
    print("\n[3x3 Budget Sweep]")
    print(b3.to_string(index=False))


if __name__ == "__main__":
    main()