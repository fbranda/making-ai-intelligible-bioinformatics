# Publishing checklist for GitHub

1. Create a **new repository** named `making-ai-intelligible-bioinformatics` (suggested name). Set the description to: `Reproducible synthetic TF-binding case study illustrating six practical strategies for interpretable AI in bioinformatics.`
2. Upload the files from the archive's root, rather than uploading the ZIP as a single opaque file. Preserve `.github/workflows/`.
3. Configure repository topics: `bioinformatics`, `explainable-ai`, `interpretability`, `reproducibility`, `genomics`, `machine-learning`, `shap`, `lime`, `conformal-prediction`, `gradio`.
4. Agree on a code license with all applicable rights holders. A public GitHub repository **does not automatically grant permission to reuse code**. This package does not select a license on behalf of the authors.
5. In a clean Python 3.11 environment, run the complete analysis. Compare the calculated values and regenerated figures with the journal manuscript. Pin the verified dependency versions.
6. Add an `environment-verified.txt` or lockfile after that run. Only then enable claims of full computational reproducibility.
7. Correct manuscript wording about compute time and supplement references if the complete run produces different results. Where the paper promises a public URL, insert the **real** URL after the repository exists.
8. When ready, create a tagged release (e.g. `v1.0.0`) and optionally archive it on Zenodo. Update `CITATION.cff` with the final journal, DOI and publication details when available.

## Git CLI approach (alternative to the GitHub website)

In the extracted repository directory:

```bash
git init
git add .
git commit -m "Initial reproducible research repository"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/making-ai-intelligible-bioinformatics.git
git push -u origin main
```

The remote repository must be created in your GitHub account before the push command. Replace the sample username with the actual account or organisation name.

## Suggested manuscript code-availability sentence

> The Python source code, illustrative simulation pipeline, figure-generation script, and local Gradio interface accompanying the synthetic transcription-factor-binding case study are available in the public GitHub repository [insert URL and versioned release]. The simulations use synthetic data only; the code has no validated clinical or biological deployment use.

Do not include a fabricated repository URL or DOI in the manuscript; add these only after they exist.
