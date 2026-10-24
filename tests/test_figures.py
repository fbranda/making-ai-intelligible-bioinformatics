"""Lightweight checks of the plotting script, not a full scientific rerun."""
import json
from pathlib import Path

import numpy as np

from figure_plots import observed_calibration, figure2, figure3


def test_calibration_summary():
    p = np.tile(np.array([0.25, 0.75]), 40)
    y = np.tile(np.array([0, 1]), 40)
    out = observed_calibration(p, y, min_count=15, bootstrap=20)
    assert out.shape == (2, 4)
    assert np.all(out[:, 2] <= out[:, 1])
    assert np.all(out[:, 1] <= out[:, 3])


def test_plotting_with_fixture(tmp_path):
    p = np.linspace(.03, .97, 120)
    y = (p > .5).astype(int)
    np.save(tmp_path / 'curve_p_te.npy', p)
    np.save(tmp_path / 'curve_p_iso.npy', np.clip(p * .9 + .05, 0, 1))
    np.save(tmp_path / 'curve_y_te.npy', y)
    keys = ['GBM raw', 'GBM isotonic', 'GBM Platt', 'MLP single', 'MLP 5-ensemble']
    names = ['IG', 'KernelSHAP', 'LIME', 'Ablation', 'TreeSHAP (GBM)']
    rr = {
        'P2': {
            'gbm_row': .86, 'gbm_chrom': .80, 'locus_prior_row': .88,
            'locus_prior_chrom': .50, 'motif_lr_row': .79, 'motif_lr_chrom': .80,
            'shuffled_row': .51, 'shuffled_chrom': .50,
        },
        'P3': {
            'names': names,
            'recovery_top10': {nm: .25 for nm in names},
            'deletion_auc': {nm: .50 for nm in ['IG', 'KernelSHAP', 'LIME', 'Ablation', 'TreeSHAP (GBM, on GBM)', 'Random (MLP)', 'Random (GBM)']},
            'spearman': np.eye(5).tolist(),
        },
        'P5_calibration': {name: {'ece': .08, 'ece_ci': [.04, .12]} for name in keys},
        'P5_selective': {'coverage': [.1, .5, 1.0], 'accuracy': [.95, .9, .8]},
    }
    (tmp_path / 'results.json').write_text(json.dumps(rr))
    assert figure2(tmp_path, tmp_path, rr).is_file()
    assert figure3(tmp_path, rr).is_file()
    assert (tmp_path / 'figure2_calibration_uncertainty.png').stat().st_size > 10_000
    assert (tmp_path / 'figure3_simulated_case_study.png').stat().st_size > 10_000
