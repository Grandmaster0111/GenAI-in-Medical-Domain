#!/usr/bin/env python3
"""Aggregates results/*.json into a summary.csv and a comparison bar chart."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from medrag.config import RESULTS_DIR


def load_summaries() -> pd.DataFrame:
    rows = []
    for path in RESULTS_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        rows.append({
            "model": data["model"],
            "pipeline": data["pipeline"],
            "dataset": data["dataset"],
            "n": data["n"],
            "accuracy": data["accuracy"],
        })
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, model: str | None = None) -> None:
    if df.empty:
        print("No results to plot.")
        return
    if model is None:
        model = df["model"].mode()[0]
    sub = df[df["model"] == model]
    if sub.empty:
        print(f"No results for model '{model}'.")
        return
    pivot = sub.pivot_table(index="dataset", columns="pipeline", values="accuracy")
    pivot = pivot.reindex(columns=[p for p in ("normal_rag", "advanced_rag", "crag") if p in pivot.columns])
    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Accuracy")
    ax.set_title(f"RAG variant comparison ({model})")
    plt.tight_layout()
    out_path = RESULTS_DIR / "comparison.png"
    plt.savefig(out_path)
    print(f"Saved chart -> {out_path}")


if __name__ == "__main__":
    df = load_summaries()
    csv_path = RESULTS_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary -> {csv_path}")
    print(df)
    plot_comparison(df)
