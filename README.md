# Making AI Intelligible in Bioinformatics

### A reproducible, synthetic transcription-factor-binding case study

**Companion code to the manuscript:**  
*Making AI Intelligible in Bioinformatics: A Practical Guide for Computational Researchers*

**Authors:** Francesco Branda · Mohamed Mustaf Ahmed · Massimo Ciccozzi · Zhinya Kawa Othman · Fabio Scarpa

---

Artificial intelligence models can be accurate without being biologically interpretable, statistically reliable, or intelligible to the people using them. This repository provides the **Python implementation of an illustrative synthetic regulatory-genomics example** used in the accompanying manuscript to demonstrate six practical strategies for making AI outputs more transparent and reproducible.

The workflow combines **benchmark integrity, post-hoc explanation checks, expert-knowledge proxies, calibration, conformal prediction, and a local Gradio demonstration**. It has been designed for methodological teaching and transparent inspection—not biological inference or clinical deployment.

> **Synthetic-data notice:** All 12,000 sequences, class labels, chromosome groups, and loci are simulated. There are no patient samples or experimentally observed binding labels in the analysis. No results in this repository should be interpreted as validated performance on real biological data.

## At a glance

| Component | Implementation |
|---|---|
| Task | Simulated binary transcription-factor-binding prediction |
| Dataset | 12,000 sequences, 100 bp each; 300 loci on ten simulated chromosome groups |
| Known motif | `GATAAG`, planted probabilistically as a predictive signal |
| Representation | All 256 DNA 4-mer count features |
| Primary evaluation | Chromosomes 1–6 training; 7–8 calibration; 9–10 testing |
| Models and baselines | Gradient boosting, five MLP seeds, motif-only logistic regression, locus prior, shuffled-label controls |
| Interpretability | Integrated gradients, KernelSHAP, LIME, ablation, TreeSHAP |
| Reliability assessment | AUROC, calibration/ECE, Brier score, split conformal prediction, selective prediction |
| Interface | Local Gradio demonstration with predicted probability, prediction set, and top attributed k-mers |

## Quick start

Use **Python 3.11** in a fresh virtual environment. The full analysis is computationally intensive and its run time varies by system and dependency versions.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Simulate data, train models, and calculate metrics
python case_study.py

# 4. Generate manuscript-style Figures 2 and 3
python figure_plots.py

# 5. Launch the example interface on your own computer
python interface_app.py
```

After the first command, the analysis writes `results.json`, `model.joblib`, and three `curve_*.npy` files to the working directory. The plotting command saves two PNG files in `figures/`. The interface loads **the model created by the first script** and displays a local Gradio URL in the terminal. The interface is **not** a hosted public website.

**Important:** Run `case_study.py` first. The repository deliberately does not contain a pre-trained model or claim that its published results have been reproduced in every possible software environment.

## Repository structure

```text
making-ai-intelligible-bioinformatics/
├── README.md                      # Repository home page
├── case_study.py                  # Full synthetic simulation and analysis
├── figure_plots.py               # Figure 2 and Figure 3 reconstruction
├── interface_app.py              # Local Gradio interactive demonstration
├── requirements.txt              # Suggested dependency compatibility ranges
├── CITATION.cff                   # Citation metadata (no fabricated DOI)
├── .gitignore                     # Excludes generated outputs and environments
├── .github/workflows/checks.yml  # Source syntax + plotting fixture CI
├── docs/
│   ├── methods.md                 # Synthetic design and six practices
│   ├── reproducibility.md         # Verification protocol and methodological limits
│   ├── publishing.md              # Steps to publish the repository
│   └── changes.md                 # Changes relative to the provided scripts
├── tests/
│   └── test_figures.py            # Lightweight plotting checks
└── figures/                       # Regenerated figure outputs go here
```

## The six practices in one workflow

1. **Structured reporting.** Define and disclose synthetic-data provenance, splits, model choices and evaluation metrics.
2. **Benchmark interrogation.** Contrast an intentionally vulnerable random-row split with a chromosome-held-out evaluation and simple/negative controls.
3. **Post-hoc explanation validation.** Compare attribution algorithms, biologically planted k-mer recovery, method agreement, and deletion-based faithfulness.
4. **Expert-in-the-loop interpretation.** Use the planted motif as a *proxy* for domain knowledge; no real expert judgment was collected in this case study.
5. **Uncertainty communication.** Measure calibration and predictive reliability; construct split-conformal label sets and explore selective prediction.
6. **Shareable interface.** Present a probability, a conformal set, and local TreeSHAP contributions through a small Gradio application.

A more detailed methodological account appears in [docs/methods.md](docs/methods.md). The simulated case study is an educational example, **not a validated interpretability framework or real-data benchmark**.

## Outputs and manuscript correspondence

| Output | Corresponds to |
|---|---|
| `results.json` | Numeric case-study results and methodological comparisons discussed in the manuscript |
| `figures/figure2_calibration_uncertainty.png` | **Figure 2:** Calibration, ECE and selective prediction |
| `figures/figure3_simulated_case_study.png` | **Figure 3:** Model benchmarking, motif recovery/deletion, and attribution agreement |
| `model.joblib` | Components required by the local **Practice 6** interface |
| Gradio interface | Interactive illustration described in **Supplementary Methods S2–S4** |

The figure script recreates the plotted analyses from saved results; it is **not guaranteed to reproduce the journal figure images pixel-for-pixel**. Manuscript Figure 1 (the conceptual framework overview) is not generated by this analysis script.

### Illustrative finding reported in the manuscript

The accompanying manuscript reports a gradient-boosting AUROC of approximately **0.86 with random-row partitioning versus 0.80 when chromosome groups are held out**. The comparison illustrates why sample dependence and partitioning must be scrutinized before predictive gains are interpreted. These are reported results from an illustrative synthetic example; use `results.json` from a verified full run to check precise values.

## Reproducibility and appropriate use

The code uses `numpy.random.default_rng(2026)` and fixed model seeds where specified. A fixed seed supports reproducibility, but **does not guarantee exactly matching numerical outputs across library versions and operating systems**. `requirements.txt` gives compatibility ranges, not a verified lockfile; a version-pinned environment should be committed after a clean full rerun.

The computationally expensive analysis, particularly repeated KernelSHAP and LIME calculations, **has not been independently rerun in full while preparing this GitHub package**. The included checks cover Python syntax and plotting with synthetic fixtures; they do not prove equality of all manuscript results. See [docs/reproducibility.md](docs/reproducibility.md) for the verification checklist and limitations.

**Safety and scope:** The output is neither a genomic diagnostic nor a clinical decision aid. Attributions are descriptive/correlational, not causal. The local interface makes no guarantees about downstream hosting logs or infrastructure privacy. Load `model.joblib` only from trusted sources; do not accept untrusted pickle-based model artifacts.

## Citation

If this repository supports a publication or project, please cite the associated manuscript:

> Branda F, Ahmed MM, Ciccozzi M, Othman ZK, Scarpa F. *Making AI Intelligible in Bioinformatics: A Practical Guide for Computational Researchers*. Manuscript. Publication details and DOI to be added when available.

A [CITATION.cff](CITATION.cff) file is included for GitHub's **Cite this repository** feature. It intentionally does not invent a journal citation, DOI, or permanent archive URL.

## Licensing and contributions

Copyright and licensing must be agreed upon by the applicable code rights holders before a reuse license is assigned. **No open-source license has been selected automatically** in this preparation package. See [docs/publishing.md](docs/publishing.md) for release steps. Bug reports and suggestions should relate to scientific reproducibility or software operation; this research demonstration is not maintained as a clinical or production service.
