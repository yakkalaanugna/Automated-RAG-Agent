"""
generate_results.py — Generate evaluation plots and sample output for README.

Produces pre-computed visualizations without requiring an API key,
using representative results from the adaptive RAG pipeline.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 12, "figure.dpi": 150})

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Representative results from actual runs ─────────────────────────────────

CONFIDENCE_TRAJECTORY = [0.66, 0.72, 0.66]

SUMMARY = {
    "Baseline RAG":  {"precision_at_k": 0.806, "recall_at_k": 1.000, "root_cause_match": 0.667, "iterations": 1.0, "latency": 2.5},
    "Iterative RAG": {"precision_at_k": 0.555, "recall_at_k": 1.000, "root_cause_match": 0.333, "iterations": 3.0, "latency": 31.9},
    "Adaptive RAG":  {"precision_at_k": 0.556, "recall_at_k": 1.000, "root_cause_match": 0.333, "iterations": 3.7, "latency": 54.1},
}

SAMPLE_OUTPUT = {
    "root_cause": "UE4 was released due to a failure while applying an RRCReconfiguration message",
    "confidence": 0.6581,
    "severity": "CRITICAL",
    "supporting_logs": [
        "[log1.txt] 18:34:08.417 ACR: UEC-1: UE4: Failure (code 4) while applying RRCReconfiguration",
        "[log2.txt] ERROR!! 18:34:08 rfma_impl.cpp[80]: UEC-1: UE4: Failure code 4",
        "[log1.txt] 18:34:09.500 INF: UEC-1: UE4: Cancel All Active Fsm before CTRL_DEL_UE",
    ],
    "reasoning_steps": [
        "Identified RRC Reconfiguration failure in eGate console logs at 18:34:08",
        "Correlated with UEC controller cancel FSM at 18:34:09",
        "Traced UE release trigger to rfma_impl.cpp failure code 4",
    ],
    "retrieval_scores": [0.7667, 0.5820, 0.5481, 0.4024, 0.2994],
    "iterations": [
        {"iteration": 1, "confidence": 0.66},
        {"iteration": 2, "confidence": 0.72},
        {"iteration": 3, "confidence": 0.66},
    ],
    "best_iteration": 2,
    "converged": True,
    "recommendation": "Investigate rfma_impl.cpp RRC reconfiguration handling; check UEC bearer timer settings.",
}


def plot_confidence_trajectory():
    """Plot confidence convergence across iterations."""
    fig, ax = plt.subplots(figsize=(8, 4))
    iters = range(1, len(CONFIDENCE_TRAJECTORY) + 1)
    ax.plot(iters, CONFIDENCE_TRAJECTORY, "o-", color="#1f77b4", linewidth=2.5, markersize=10)
    ax.fill_between(iters, CONFIDENCE_TRAJECTORY, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Confidence Score")
    ax.set_title("Adaptive RAG — Confidence Trajectory")
    ax.set_ylim(0, 1)
    ax.set_xticks(list(iters))
    ax.grid(True, alpha=0.3)
    for i, c in zip(iters, CONFIDENCE_TRAJECTORY):
        ax.annotate(f"{c:.2f}", (i, c), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=11)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "confidence_trajectory.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_method_comparison():
    """Plot bar chart comparing the three RAG methods."""
    methods = list(SUMMARY.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = ["precision_at_k", "recall_at_k", "root_cause_match"]
    titles = ["Precision@5", "Recall@5", "Root Cause Match"]
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        vals = [SUMMARY[m][metric] for m in methods]
        bars = ax.bar(methods, vals, color=color, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=11)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("RAG Method Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "method_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_sample_output():
    """Save sample structured JSON output."""
    path = os.path.join(RESULTS_DIR, "sample_output.json")
    with open(path, "w") as f:
        json.dump(SAMPLE_OUTPUT, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating results for README...\n")
    plot_confidence_trajectory()
    plot_method_comparison()
    save_sample_output()
    print("\nDone. All files saved to results/")
