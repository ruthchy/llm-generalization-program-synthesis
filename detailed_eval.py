"""
detailed_eval_modular.py

Performs detailed evaluation of program synthesis and image generation metrics for a given experiment.
Loads test and validation metrics, maps abstractions, computes summary statistics, correlations, and
exports metric behavior cases. Designed for use with experiment directories containing config and metrics files.

Usage:
    python detailed_eval_modular.py --eval_dir /path/to/eval_dir
Example:
    python detailed_eval_modular.py --eval_dir results/length/CodeLlama_20250531_0107/inference/20250621_2247
"""
# Imports
import os
import re
import json
import yaml
import argparse
from collections import defaultdict
import hashlib
import contextlib

import numpy as np
import pandas as pd
import scipy.stats

import matplotlib.pyplot as plt
from datasets import load_dataset
from __eval import LLMCodeEvaluator
from __pipeline_hash import get_val_dataset_hash

ABSTRACTION_CATEGORIES = {
    "Basic Shape": {"line", "triangle", "square", "5 gon", "6 gon", "7 gon", "8 gon", "9 gon", "circle", "semicircle"},
    "Complex Shape": {"staircase", "zigzag", "spiral", "star"},
    "Sequence": {"connected", "separated"},
    "Concentric": {"concetric"},
    "In-a-row": {"row"},
    "Snowflake": {"snowflake"}
}

METRICS_TO_PLOT = [
    ("normalized_lev_distance", "Norm Lev Sim"),
    ("crystalbleu_score", "crystal BLEU"),
    ("ssim_score", "SSIM"),
    ("dreamsim_score", "DreamSim"),
    ("pixel_precision", "Precision"),
    ("pixel_recall", "Recall"),
    ("pixel_f1", "F1 Score")
]

# Helper
def classify_abstraction(description):
    desc = description.lower()
    # Check all categories except Basic Shape first
    for label in ["Snowflake", "In-a-row", "Concentric", "Sequence", "Complex Shape"]:
        keywords = ABSTRACTION_CATEGORIES[label]
        if any(word in desc for word in keywords):
            return label
    # Only assign Basic Shape if none of the above matched
    if any(word in desc for word in ABSTRACTION_CATEGORIES["Basic Shape"]):
        return "Basic Shape"
    return "Label Failure"

def load_config_and_metrics(eval_dir):
    config = yaml.safe_load(open(os.path.join(eval_dir, 'config.yaml')))
    with open(os.path.join(eval_dir, 'detailed_metrics.jsonl')) as f:
        metrics = [json.loads(line) for line in f]
    return config, metrics

def get_parent_eval_dir(eval_dir):
    # Remove everything after /inference to get parent directory
    parts = eval_dir.split(os.sep)
    if "inference" in parts:
        idx = parts.index("inference")
        return os.sep.join(parts[:idx])
    return eval_dir

def load_val_metrics(parent_eval_dir):
    val_metrics_path = os.path.join(parent_eval_dir, "val_detailed_metrics.jsonl")
    if not os.path.exists(val_metrics_path):
        print(f"Validation metrics file not found at {val_metrics_path}")
        return []
    with open(val_metrics_path) as f:
        return [json.loads(line) for line in f]

def map_abstractions(metrics, dataset):
    abstraction_counts = defaultdict(int)
    for entry in metrics:
        row = int(entry['id'].split('_')[0])
        abstraction = classify_abstraction(dataset[row]['Description'])
        entry['abstraction'] = abstraction
        abstraction_counts[abstraction] += 1
    return metrics, abstraction_counts

def save_abstraction_summaries(metrics, eval_dir, evaluator):
    abstraction_groups = defaultdict(list)
    for entry in metrics:
        abstraction_groups[entry['abstraction']].append(entry)

    for abstraction, group in abstraction_groups.items():
        summary = evaluator.generate_summary(group)
        out_path = os.path.join(eval_dir, f"evaluation_{abstraction.replace(' ', '_')}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

def filter_last_eval_step(val_metrics):
    # Keep only the last eval_step_count for each program_id
    df = pd.DataFrame(val_metrics)
    idx = df.groupby("program_id")["eval_step_count"].idxmax()
    return df.loc[idx].reset_index(drop=True)

def export_metric_behavior(metrics, eval_dir):
    perfect_all = [
        m for m in metrics
        if m.get("pixel_f1") == 1.0 and m.get("ssim_score") == 1.0 and m.get("dreamsim_score") == 0.0
    ]
    perfect_any = [
        m for m in metrics
        if (
            (m.get("pixel_f1") == 1.0 or m.get("ssim_score") == 1.0 or m.get("dreamsim_score") == 0.0)
            and not (m.get("pixel_f1") == 1.0 and m.get("ssim_score") == 1.0 and m.get("dreamsim_score") == 0.0)
        )
    ]
    zero_f1 = [m for m in metrics if m.get("pixel_f1", 1) == 0.0]

    def entries_to_df(entries):
        return pd.DataFrame([
            {
                "id": m.get("id"),
                "pixel_f1": m.get("pixel_f1"),
                "ssim_score": m.get("ssim_score"),
                "dreamsim_score": m.get("dreamsim_score")
            }
            for m in entries
        ])

    df_all = entries_to_df(perfect_all)
    df_any = entries_to_df(perfect_any)
    df_zero_f1 = entries_to_df(zero_f1)

    excel_path = os.path.join(eval_dir, "metric_behavior.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df_all.to_excel(writer, sheet_name="All_Perfect", index=False)
        df_any.to_excel(writer, sheet_name="Any_Perfect", index=False)
        df_zero_f1.to_excel(writer, sheet_name="F1_Zero", index=False)

    print(f"Saved metric behavior cases to {excel_path}")

    # Print IDs and values
    print(f"\nIDs where F1=1, SSIM=1, DreamSim=0: {df_all['id'].tolist()}")
    print(f"\nIDs where at least one metric is perfect (F1=1 or SSIM=1 or DreamSim=0): {df_any['id'].tolist()}")
    print(f"Found {len(df_zero_f1)} entries with pixel_f1 == 0.0")

def spearman_corr(metrics, key1, key2):
    """Calculate Spearman correlation between two metric keys, ignoring None and NaN."""
    vals1 = []
    vals2 = []
    for m in metrics:
        v1 = m.get(key1)
        v2 = m.get(key2)
        if v1 is not None and v2 is not None and not (pd.isna(v1) or pd.isna(v2)):
            vals1.append(v1)
            vals2.append(v2)
    if len(vals1) < 2:
        print(f"Not enough data to compute Spearman correlation between {key1} and {key2}.")
        return None
    corr, pval = scipy.stats.spearmanr(vals1, vals2)
    print(f"Spearman correlation between {key1} and {key2}: {corr:.4f} (p={pval:.4g})")
    return corr, pval
    
# Plotting Functions
def plot_metrics(metrics, eval_dir):
    plots_dir = os.path.join(eval_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for ax, (key, label) in zip(axes, METRICS_TO_PLOT):
        values = [m.get(key) for m in metrics if m.get(key) is not None]
        ax.hist(values, bins=30, color='#607196')
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.grid(True)
    
    # Remove any unused axes
    if len(METRICS_TO_PLOT) < len(axes):
        for j in range(len(METRICS_TO_PLOT), len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "all_metrics_histograms.png"))
    plt.close()

def plot_precision_recall(metrics, eval_dir):
    plots_dir = os.path.join(eval_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.DataFrame([
        {
            "id": m["id"],
            "pixel_precision": m.get("pixel_precision"),
            "pixel_recall": m.get("pixel_recall"),
            "pixel_f1": m.get("pixel_f1"),
            "abstraction": m.get("abstraction", "Unknown")
        }
        for m in metrics
        if all(k in m and m[k] is not None for k in ["pixel_precision", "pixel_recall", "pixel_f1"])
    ])

    def _scatter(df, title, path):
        plt.figure(figsize=(7, 5))
        plt.scatter(df["pixel_precision"], df["pixel_recall"], c=df["pixel_f1"], cmap="viridis", s=100)
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.title(title)
        plt.colorbar(label="F1 Score")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    _scatter(df, "Precision-Recall Scatter (All)", os.path.join(plots_dir, "precision_recall_all.png"))

    for abstraction, group_df in df.groupby("abstraction"):
        if not group_df.empty:
            out_path = os.path.join(plots_dir, f"precision_recall_{abstraction.replace(' ', '_')}.png")
            _scatter(group_df, f"PR Scatter ({abstraction})", out_path)

# Main Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', type=str, required=True)
    args = parser.parse_args()
    eval_dir = args.eval_dir

    evaluator = LLMCodeEvaluator()
    config, metrics = load_config_and_metrics(eval_dir)
    dataset = load_dataset(config['data']['dataset_id'], split='test')
    print(f"Loaded {len(metrics)} metrics and {len(dataset)} dataset entries.")

    # Load validation metrics
    parent_eval_dir = get_parent_eval_dir(eval_dir)
    val_metrics = load_val_metrics(parent_eval_dir)
    val_config = os.path.join(parent_eval_dir, 'config.yaml') 
    val_dataset_hash = get_val_dataset_hash(val_config)
    
    if val_metrics:
        val_df = filter_last_eval_step(val_metrics)
        if "image_metrics" in val_df.columns:
            image_metrics_df = val_df["image_metrics"].apply(pd.Series)
            val_df = pd.concat([val_df.drop(columns=["image_metrics"]), image_metrics_df], axis=1)

        val_df["prog_hash"] = val_df["program_id"].apply(lambda x: x.split("_")[1])
        val_df["Description"] = val_df["prog_hash"].map(val_dataset_hash)
        missing = val_df["Description"].isna().sum()
        if missing > 0:
            print(f"Warning: {missing} validation entries could not be matched to a Description.")

        val_df["abstraction"] = val_df["Description"].apply(classify_abstraction)

        # Rename for consistency
        if "norm_lev_dist" in val_df.columns:
            val_df = val_df.rename(columns={"norm_lev_dist": "normalized_lev_distance"})
        print(f"Loaded {len(val_df)} validation metrics (last eval step only).")
        # Now you can compare val_df to metrics/test set as needed
    else:
        print("No validation metrics loaded.")

    # Prepare output file
    out_txt = os.path.join(eval_dir, "detailed_eval_report.txt")
    with open(out_txt, "w") as f, contextlib.redirect_stdout(f):
        
        config, metrics = load_config_and_metrics(eval_dir)
        dataset = load_dataset(config['data']['dataset_id'], split='test')
        print(f"Loaded {len(metrics)} metrics and {len(dataset)} dataset entries.")

        # 1 - Map abstractions
        print("Mapping abstractions...")
        metrics, abstraction_counts = map_abstractions(metrics, dataset)

        print("\nAbstraction Group Counts:")
        for k, v in abstraction_counts.items():
            print(f"{k:<20} {v:<6}")

        # 2 - Calculate evaluation metrics per abstraction
        print("\nCalculating evaluation metrics per abstraction...")
        if not any(f.startswith("evaluation_") for f in os.listdir(eval_dir)):
            save_abstraction_summaries(metrics, eval_dir, evaluator)
        else:
            print("evaluation_*.json already exists — skipping summary.")

        # 3 - Metric behavior
        plot_metrics(metrics, eval_dir)
        plot_precision_recall(metrics, eval_dir)

        # 3.1 - Export metric behavior
        export_metric_behavior(metrics, eval_dir)

        # 3.2 - Spearman correlation
        ### between Program metrics
        spearman_corr(metrics, "normalized_lev_distance", "crystalbleu_score")
        ### between image metrics
        spearman_corr(metrics, "ssim_score", "dreamsim_score")
        spearman_corr(metrics, "pixel_f1", "ssim_score")
        spearman_corr(metrics, "pixel_f1", "dreamsim_score")
        ### between Program and image metrics
        spearman_corr(metrics, "normalized_lev_distance", "pixel_f1")
        spearman_corr(metrics, "crystalbleu_score", "pixel_f1")

        # 4 - Compare validation metrics to test metrics (last eval step:candidate program 1)
        test_first_candidates = [m for m in metrics if m.get("id", "").endswith("_1")]
        df_test = pd.DataFrame(test_first_candidates)
        df_val = val_df  # already filtered

        compare_metrics = ["normalized_lev_distance", "crystalbleu_score", "ssim_score", "dreamsim_score", "pixel_precision", "pixel_recall", "pixel_f1"]

        print("\n=== Overall Means (Test First Candidates) ===")
        print(df_test[compare_metrics].mean())

        print("\n=== Overall Means (Validation Last Step) ===")
        print(df_val[compare_metrics].mean())

        print("\n=== Spearman Correlations (Test First Candidates) ===")
        for m1, m2 in [("normalized_lev_distance", "crystalbleu_score"),
                    ("ssim_score", "dreamsim_score"),
                    ("pixel_f1", "ssim_score"),
                    ("pixel_f1", "dreamsim_score"),
                    ("normalized_lev_distance", "pixel_f1"),
                    ("crystalbleu_score", "pixel_f1")]:
            corr, pval = scipy.stats.spearmanr(df_test[m1], df_test[m2], nan_policy='omit')
            print(f"{m1} vs {m2}: {corr:.4f} (p={pval:.4g})")

        print("\n=== Spearman Correlations (Validation Last Step) ===")
        for m1, m2 in [("normalized_lev_distance", "crystalbleu_score"),
                    ("ssim_score", "dreamsim_score"),
                    ("pixel_f1", "ssim_score"),
                    ("pixel_f1", "dreamsim_score"),
                    ("normalized_lev_distance", "pixel_f1"),
                    ("crystalbleu_score", "pixel_f1")]:
            corr, pval = scipy.stats.spearmanr(df_val[m1], df_val[m2], nan_policy='omit')
            print(f"{m1} vs {m2}: {corr:.4f} (p={pval:.4g})")

        print("\n=== Per Abstraction Means and Correlations (Test First Candidates) ===")
        for abstraction, group in df_test.groupby("abstraction"):
            print(f"\nAbstraction: {abstraction}")
            print(group[compare_metrics].mean())
            for m1, m2 in [("normalized_lev_distance", "crystalbleu_score"),
                        ("ssim_score", "dreamsim_score"),
                        ("pixel_f1", "ssim_score"),
                        ("pixel_f1", "dreamsim_score"),
                        ("normalized_lev_distance", "pixel_f1"),
                        ("crystalbleu_score", "pixel_f1")]:
                corr, pval = scipy.stats.spearmanr(group[m1], group[m2], nan_policy='omit')
                print(f"{m1} vs {m2}: {corr:.4f} (p={pval:.4g})")

        print("\n=== Per Abstraction Means and Correlations (Validation Last Step) ===")
        for abstraction, group in df_val.groupby("abstraction"):
            print(f"\nAbstraction: {abstraction}")
            print(group[compare_metrics].mean())
            for m1, m2 in [("normalized_lev_distance", "crystalbleu_score"),
                        ("ssim_score", "dreamsim_score"),
                        ("pixel_f1", "ssim_score"),
                        ("pixel_f1", "dreamsim_score"),
                        ("normalized_lev_distance", "pixel_f1"),
                        ("crystalbleu_score", "pixel_f1")]:
                corr, pval = scipy.stats.spearmanr(group[m1], group[m2], nan_policy='omit')
                print(f"{m1} vs {m2}: {corr:.4f} (p={pval:.4g})")

    print(f"\nDetailed evaluation report saved to {out_txt}")

if __name__ == "__main__":
    main()
