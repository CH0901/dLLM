"""
Result Analysis & Visualization
"""
import os
import json
import glob
import numpy as np
from typing import List, Dict


def load_experiment_results(results_dir: str) -> Dict[str, dict]:
    experiments = {}
    for exp_dir in sorted(glob.glob(os.path.join(results_dir, "*"))):
        if not os.path.isdir(exp_dir):
            continue
        summary_path = os.path.join(exp_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                experiments[os.path.basename(exp_dir)] = json.load(f)
    return experiments


def print_comparison_table(experiments: Dict[str, dict]):
    print("\n" + "=" * 90)
    print(f"{'Method':<25} {'Flip Rate':>12} {'KL Div':>12} {'Final Match':>12}")
    print("-" * 90)
    for name, result in sorted(experiments.items()):
        print(
            f"{name:<25} "
            f"{result.get('avg_token_flip_rate', 0):.4f} ± {result.get('std_token_flip_rate', 0):.4f}  "
            f"{result.get('avg_kl_divergence', 0):.6f}  "
            f"{result.get('avg_final_match', 0):.4f}"
        )
    print("=" * 90)


def generate_plots(results_dir: str, output_path: str = "analysis_plots.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not installed. Skipping plots.")
        return

    experiments = load_experiment_results(results_dir)
    if not experiments:
        print("No results found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("dLLM Quantization: FP vs Quant Comparison", fontsize=14)

    names = list(experiments.keys())
    flip_rates = [experiments[n].get("avg_token_flip_rate", 0) for n in names]
    kl_divs = [experiments[n].get("avg_kl_divergence", 0) for n in names]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(names)]

    axes[0].bar(range(len(names)), flip_rates, color=colors)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Avg Token Flip Rate")
    axes[0].set_title("Token Flip Rate")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(range(len(names)), kl_divs, color=colors)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Avg KL Divergence")
    axes[1].set_title("KL Divergence")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[Analysis] Plots saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--plot_output", type=str, default="analysis_plots.png")
    args = parser.parse_args()

    experiments = load_experiment_results(args.results_dir)
    print_comparison_table(experiments)
    generate_plots(args.results_dir, args.plot_output)
