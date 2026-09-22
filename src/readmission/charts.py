"""Matplotlib figures for the README and results document."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#1f2937"
ACCENT = "#0e6a70"
MUTED = "#9ca3af"
WARN = "#b45309"


def _finish(fig, ax, path: Path) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reliability(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.6))
    top = max(table["mean_predicted"].max(), table["observed_rate"].max()) * 1.1
    ax.plot([0, top], [0, top], color=MUTED, linestyle="--", linewidth=1, label="perfect calibration")
    ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", color=ACCENT, label="model (test, deciles)")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Mean predicted 30-day readmission risk")
    ax.set_ylabel("Observed readmission rate")
    ax.set_title("Calibration on the held-out test patients", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, ax, path)


def capacity_curve(curve: pd.DataFrame, chosen: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(curve["capacity_pct"], curve["recall"] * 100, color=ACCENT, label="readmissions reached (recall)")
    ax.plot(curve["capacity_pct"], curve["capacity_pct"], color=MUTED, linestyle="--", label="random selection")
    ax.axvline(chosen * 100, color=WARN, linewidth=1)
    ax.text(chosen * 100 + 1, 5, f"{chosen:.0%} capacity", color=WARN, fontsize=8)
    ax.set_xlabel("% of discharges that get a follow-up call (highest risk first)")
    ax.set_ylabel("% of 30-day readmissions reached")
    ax.set_title("What a follow-up budget buys (test patients)", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, ax, path)


def subgroup_auc(table: pd.DataFrame, path: Path) -> None:
    frame = table.dropna(subset=["roc_auc"]).reset_index(drop=True)
    labels = [f"{a}: {g} (n={n:,})" for a, g, n in zip(frame["attribute"], frame["group"], frame["encounters"])]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(frame) + 1.2))
    y = range(len(frame))
    ax.hlines(y, frame["roc_auc_ci_low"], frame["roc_auc_ci_high"], color=MUTED, linewidth=2)
    ax.scatter(frame["roc_auc"], y, color=ACCENT, zorder=3, s=20)
    ax.set_yticks(list(y), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC on test patients, 95% CI (patient bootstrap)")
    ax.set_title("Ranking quality by subgroup", loc="left", color=INK, fontsize=10)
    _finish(fig, ax, path)
