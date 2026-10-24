# Changes relative to the user-supplied Python files

The original modelling and evaluation calculations in `case_study.py` have been kept intact. Its introductory usage description was corrected: the original script does **not** generate `fig_*.png` files, but does generate `results.json`, `model.joblib`, and three NPY arrays.

In `interface_app.py`, the **Clear** button now clears all five interface outputs, including the caveat text field (rather than leaving that field unchanged). The numerical prediction algorithm and conformal set calculation were not changed.

A **new** `figure_plots.py` was added because the accompanying Supplementary Methods mention such a script, but it was not in the supplied code attachments. It creates manuscript-style calibration and case-study charts from the arrays and JSON produced by `case_study.py`. Its display choices and row-wise bootstrap plotting are documented and must be independently checked against the original figure exports before substituting the images into the manuscript.

Additional new files (`README.md`, `requirements.txt`, `CITATION.cff`, documentation, tests, and CI workflow) provide a ready-to-upload repository structure. No generated numerical results or journal DOI were fabricated.
