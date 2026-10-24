# Methods and scientific interpretation

This repository accompanies **Making AI Intelligible in Bioinformatics: A Practical Guide for Computational Researchers**. The code illustrates methodological principles; it does **not** analyze observational genomic data or establish performance in real applications.

## Synthetic task and planted ground truth

- **Samples:** 12,000 simulated DNA sequences, each 100 base pairs long.
- **Dependence:** 300 loci on ten artificial chromosome groups, with 40 closely related sequences per locus. Each locus has a shared template with approximately 10% base mutations per sequence and a locus-specific class prior.
- **Labels:** simulated binary transcription-factor binding outcomes.
- **Planted signal:** the `GATAAG` motif is inserted with probability 70% among positive examples and 10% among negative examples. Binding labels are generated before motif insertion, so the motif is a predictive signal constructed by the generator—not evidence of a real biological binding mechanism.
- **Features:** counts of all 256 possible DNA 4-mers.

The locus-specific outcome prior and shared sequence templates make naive row-wise partitioning susceptible to dependence between train and test observations. This is a teaching example of potentially misleading benchmarking rather than a claim about leakage in any specific experimental dataset.

## Partitioning and models

The primary partition uses chromosome groups 1–6 for training, 7–8 for calibration, and 9–10 for final evaluation. For comparison, a separate randomly shuffled row split is applied to the same synthetic data and is intentionally considered susceptible to locus-level information overlap. The script evaluates a gradient-boosting classifier (120 estimators, maximum depth three), a locus-identity prior baseline, motif 4-mer logistic regression, shuffled-label controls, and multilayer perceptrons with five random seeds.

### Six practical strategies in the implementation

| Practice | Reproducible component |
|---|---|
| 1. Structured reporting | Explicit simulation parameters and partitioning (see supplementary DOME and model-card templates). |
| 2. Benchmark interrogation | Row split versus chromosome-held-out split, shuffled-label controls, locus and motif baselines. |
| 3. Post-hoc explanation validation | Integrated gradients, KernelSHAP, LIME, one-feature ablation, TreeSHAP; motif recovery, rank agreement, deletion curves. |
| 4. Expert-in-the-loop | A **synthetic motif-based proxy**, not feedback from an actual domain expert. |
| 5. Uncertainty | Raw/isotonic/Platt calibration, AUROC, ECE, Brier score, bootstrap summaries, conformal sets, selective prediction. |
| 6. Shareable interface | Local Gradio demonstration using the exported artifact. |

## Interpretation boundaries

- The analysis is **synthetic**. It neither tests real transcription-factor binding nor supports biological discovery or clinical decisions.
- Explanations describe learned model behaviour and correlations, **not causal mechanisms**.
- Comparing attributions between a GBM and a neural network is partially a comparison of different trained models, not solely different attribution algorithms.
- `KernelSHAP` is intentionally run with a restricted sampling budget. Its conclusions are sensitive to choices of background data, perturbation model, and computational budget.
- Row-wise bootstrap calibration intervals are **descriptive** and do not account for clustering within loci/chromosomes.
- Split-conformal marginal guarantees rely on exchangeability; performance under chromosome-level distribution shift should be interpreted empirically, not assumed to carry a formal 90% guarantee.
- The code produces a **proxy**, not a genuine blinded expert validation of predictions.
- A single master random generator seed does not guarantee bit-for-bit agreement across operating systems or library versions.

For claims and interpretations, the manuscript and Supplementary Methods remain the authoritative scientific sources.
