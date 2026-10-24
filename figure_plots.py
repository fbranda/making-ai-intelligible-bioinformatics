"""Regenerate manuscript-style Figures 2 and 3 from case_study.py outputs.

Usage: python figure_plots.py [--input-dir .] [--output-dir figures]

These plots reconstruct the reported analyses; they are not guaranteed to
be pixel-identical to images embedded in the manuscript. Input files are
created by `python case_study.py`. No results are hard-coded here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def observed_calibration(prob, labels, *, bins=10, min_count=15, seed=2026, bootstrap=500):
    """Return (mean predicted probability, observed fraction, row bootstrap CI).

    Bins with fewer than `min_count` observations are omitted. Bootstrap
    intervals are descriptive row-level intervals, not chromosome-clustered
    uncertainty intervals; locus dependence is not accounted for.
    """
    rng = np.random.default_rng(seed)
    p, y = np.asarray(prob).reshape(-1), np.asarray(labels).reshape(-1)
    if len(p) != len(y):
        raise ValueError("probability and label arrays must have equal lengths")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("predicted probabilities must be finite and in [0, 1]")
    if np.any(~np.isin(y, [0, 1])):
        raise ValueError("labels must be binary")
    idx = np.minimum(np.floor(p * bins).astype(int), bins - 1)
    rows = []
    for bin_id in range(bins):
        selected = idx == bin_id
        if selected.sum() < min_count:
            continue
        frac = y[selected].mean()
        samples = rng.choice(y[selected], (bootstrap, int(selected.sum())), replace=True)
        low, high = np.quantile(samples.mean(axis=1), [0.025, 0.975])
        rows.append((p[selected].mean(), frac, low, high))
    if not rows:
        return np.empty((0, 4))
    return np.asarray(rows)


def _results(input_dir):
    path = Path(input_dir) / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; first run: python case_study.py")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def figure2(input_dir, output_dir, results):
    """Calibration, ECE uncertainty and selective prediction."""
    src = Path(input_dir)
    raw = np.load(src / "curve_p_te.npy", allow_pickle=False)
    iso = np.load(src / "curve_p_iso.npy", allow_pickle=False)
    y = np.load(src / "curve_y_te.npy", allow_pickle=False)
    unc = results["P5_calibration"]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    for pp, name, color in [(raw, "GBM raw", "#CA7D37"), (iso, "GBM isotonic", "#376D9B")]:
        curve = observed_calibration(pp, y)
        if not len(curve):
            continue
        ax.plot(curve[:, 0], curve[:, 1], "o-", color=color, linewidth=1.7,
                label=f"{name} (ECE {unc[name]['ece']:.3f})")
        ax.fill_between(curve[:, 0], curve[:, 2], curve[:, 3], color=color, alpha=0.13)
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1), title="A  |  Calibration")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    keys = [name for name in ["GBM raw", "GBM isotonic", "GBM Platt", "MLP single", "MLP 5-ensemble"] if name in unc]
    xx = np.arange(len(keys))
    yy = np.array([unc[k]["ece"] for k in keys])
    lo = np.array([unc[k]["ece_ci"][0] for k in keys]); hi = np.array([unc[k]["ece_ci"][1] for k in keys])
    ax.bar(xx, yy, color="#6487A0", width=0.68)
    ax.errorbar(xx, yy, yerr=[np.maximum(0, yy - lo), np.maximum(0, hi - yy)],
                fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax.set_xticks(xx, [k.replace(" ", "\n", 1) for k in keys], fontsize=8)
    ax.set(ylabel="Expected calibration error", title="B  |  Calibration error")
    ax.set_ylim(bottom=0)

    ax = axes[2]
    selective = results["P5_selective"]
    cov = np.asarray(selective["coverage"]); acc = np.asarray(selective["accuracy"])
    ax.plot(cov, acc, "o-", color="#376D9B", markersize=3)
    ax.axhline(acc[-1], linestyle=":", color="black", linewidth=1,
               label=f"Full coverage accuracy: {acc[-1]:.2f}")
    ax.set(xlabel="Coverage (fraction retained)", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1),
           title="C  |  Selective prediction")
    ax.legend(fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    output = Path(output_dir) / "figure2_calibration_uncertainty.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def figure3(output_dir, results):
    """Benchmark split comparison, explanation recovery/deletion, agreement."""
    p2, p3 = results["P2"], results["P3"]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.1), constrained_layout=True)

    ax = axes[0]
    cat = ["GBM", "Locus\nprior", "Motif-only\nLR", "Shuffled\nlabels"]
    row = [p2[k] for k in ["gbm_row", "locus_prior_row", "motif_lr_row", "shuffled_row"]]
    chr_split = [p2[k] for k in ["gbm_chrom", "locus_prior_chrom", "motif_lr_chrom", "shuffled_chrom"]]
    xx = np.arange(4)
    ax.bar(xx - .19, row, .38, label="Random row", color="#D08B53")
    ax.bar(xx + .19, chr_split, .38, label="Held-out chromosomes", color="#5689B5")
    ax.set_xticks(xx, cat)
    ax.set(ylabel="AUROC", ylim=(0, 1), title="A  |  Benchmark integrity")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.legend(fontsize=8)

    ax = axes[1]
    names = ["IG", "KernelSHAP", "LIME", "Ablation", "TreeSHAP (GBM)"]
    recover = np.array([p3["recovery_top10"][nm] for nm in names])
    del_keys = ["IG", "KernelSHAP", "LIME", "Ablation", "TreeSHAP (GBM, on GBM)"]
    deletion = np.array([p3["deletion_auc"][nm] for nm in del_keys])
    xi = np.arange(len(names))
    bar_width = .35
    ax.bar(xi - bar_width / 2, recover, bar_width, label="Motif k-mer recovery", color="#559A75")
    ax.axhline(10 / 256, color="#559A75", linestyle=":", linewidth=1.2)
    ax.set(ylabel="Top-10 motif recovery", ylim=(0, 1.05), title="B  |  Biological alignment vs. faithfulness")
    ax.set_xticks(xi, ["IG", "Kernel\nSHAP", "LIME", "Ablation", "Tree\nSHAP"], fontsize=8)
    twin = ax.twinx()
    twin.bar(xi + bar_width / 2, deletion, bar_width, label="Deletion AUC (lower better)", color="#A4A7AD")
    twin.set(ylabel="Deletion AUC", ylim=(0, 1.05))
    # Random-order reference varies by model and is not directly comparable.
    mlp_ref = p3["deletion_auc"].get("Random (MLP)")
    gbm_ref = p3["deletion_auc"].get("Random (GBM)")
    if mlp_ref is not None:
        twin.hlines(mlp_ref, -0.45, 3.45, colors="#BD5C55", linestyles="--", linewidth=1.2,
                    label="Random order (MLP)")
    if gbm_ref is not None:
        twin.hlines(gbm_ref, 3.55, 4.45, colors="#8C3330", linestyles="--", linewidth=1.2,
                    label="Random order (GBM)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower left", fontsize=7)

    ax = axes[2]
    corr = np.asarray(p3["spearman"], dtype=float)
    label_names = p3["names"]
    if corr.shape != (len(label_names), len(label_names)):
        raise ValueError("P3 Spearman matrix dimensions do not match method names")
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(label_names)), label_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(label_names)), label_names, fontsize=8)
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i, j]) > .65 else "black")
    ax.set_title("C  |  Attribution rank agreement")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Spearman correlation")

    output = Path(output_dir) / "figure3_simulated_case_study.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=".", help="Directory with case_study.py generated files")
    parser.add_argument("--output-dir", default="figures", help="Output directory for plots")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = _results(args.input_dir)
    for fig in [figure2(args.input_dir, out, data), figure3(out, data)]:
        print(f"Saved: {fig}")


if __name__ == "__main__":
    main()
