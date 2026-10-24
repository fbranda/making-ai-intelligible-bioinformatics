# Reproducibility and verification status

## Environment

Recommended starting point: **Python 3.11** in an isolated virtual environment. `requirements.txt` records plausible compatibility ranges rather than a formally tested lockfile. For journal-grade archival reproducibility, perform a full clean-machine rerun, log dependency versions with `python -m pip freeze > environment-verified.txt`, and tag the exact tested commit.

## Running the analysis

```bash
python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python case_study.py
python figure_plots.py
python interface_app.py
```

`case_study.py` is computationally substantial because it evaluates several explanation methods on 80 observations, including repeated KernelSHAP and LIME calls; execution time is hardware and dependency dependent. **Do not assume a sub-minute runtime.**

## Generated files

| File | Description |
|---|---|
| `results.json` | Structured numerical results for benchmarking, attribution analysis, calibration, and conformal assessment. |
| `model.joblib` | Fitted gradient-boosting model, isotonic calibrator, conformal quantile, and feature dictionary used by the interface. |
| `curve_p_te.npy`, `curve_p_iso.npy`, `curve_y_te.npy` | Held-out probabilities and observed labels used to recreate calibration graphics. |
| `figures/figure2_calibration_uncertainty.png` | Regenerated visual summary of calibration and selective prediction. |
| `figures/figure3_simulated_case_study.png` | Regenerated visual summary of benchmark integrity and explanation comparison. |

Files are deliberately excluded from version control by `.gitignore` until a full rerun is reviewed. An archived reproducibility release may include generated figures and `results.json` **only after** numerical consistency checks; `model.joblib` should be distributed only after reviewing provenance and security implications.

## Verification protocol prior to publication

1. Run the pipeline in a clean environment, record `pip freeze`, Python version, CPU, and actual elapsed time.
2. Verify all numerical claims and significant digits in the paper against the regenerated `results.json`.
3. Check that `figure_plots.py` accurately represents those values, and compare regenerated figures with the manuscript originals. Identical styling is not guaranteed.
4. Start Gradio locally; test valid and invalid input, model loading, button behaviour, output shapes, and conformal-label consistency.
5. Optionally attach a frozen environment file or a container image after the full rerun.
6. Make a versioned GitHub release, record its URL in manuscript Data Availability, and consider Zenodo archival for a persistent DOI.

## Current verification level of this repository package

- Python source-code syntax has been checked.
- Figure reconstruction has a small independent **synthetic fixture test**; passing that test does not validate the manuscript numbers.
- The full model fitting plus all SHAP/LIME operations has **not** been independently rerun as part of packaging this repository. The upstream source code must be evaluated in the final locked environment.
- No public Gradio deployment has been claimed.

## Trust and privacy

`joblib` uses pickle-based loading: **never open model artifacts from untrusted sources**. The interface does not explicitly write sequences to a dataset; this is not a promise that hosting infrastructure, telemetry, access logs, or upstream services will never retain information. For demonstrations, use synthetic sequences only. Public deployment requires a separate privacy, security, access-control, and workload review.
