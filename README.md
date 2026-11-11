# Coronary disease models across hospitals

Can a model trained on Cleveland Clinic records identify angiographic coronary disease in a separate Budapest cohort? This exploratory portfolio project pairs a reproducible Python pipeline with an offline dashboard for model comparison.

Open `docs/index.html` in a browser to switch between logistic regression and gradient boosting, compare ROC and calibration plots, and see how missed cases and false positives change at different cutoffs.

## Results

| Model | Cleveland holdout AUROC | Budapest AUROC | Budapest minus Cleveland AUROC (95% CI) | Budapest Brier score |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.896 | 0.876 | −0.019 (−0.096 to +0.063) | 0.129 |
| Gradient boosting | 0.880 | 0.866 | −0.014 (−0.092 to +0.073) | 0.135 |

Logistic regression had the higher observed AUROC and lower Brier score in Budapest. Both intervals for the AUROC change include zero. A 50% cutoff on the logistic model flagged 78 of 106 disease-positive Budapest records and 25 of 188 disease-negative records. The cutoff illustrates an error trade-off for these cohorts.

## Reproduce

Run with Python 3.12 and Bash. The two original, processed UCI files are included under `data/`. The dashboard has no external runtime dependencies.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
bash run.sh
```

The run regenerates `docs/results.json` and `docs/index.html`. Pass a directory to `bash run.sh` to write elsewhere. The pipeline checks the source file hashes, cohort sizes, outcome counts and category codes before fitting. The dashboard embeds aggregate results and contains no row-level records.

## Study design

1. Map the angiographic disease-status label to a binary outcome. Cleveland contributes 303 records and Budapest contributes 294. The task classifies existing disease at assessment.
2. Split Cleveland 70/30 with stratification, yielding 212 training and 91 internal holdout records. All learned preprocessing and both fixed model specifications are fitted on the training partition. Budapest is scored without refitting.
3. Use five continuous and five categorical variables, including measurements obtained during an exercise test. Omit fluoroscopy because of its proximity to angiographic assessment and extensive Budapest missingness. Omit exercise ST slope and thallium because of extensive Budapest missingness. Budapest feature availability informed this harmonisation. Model parameters and decision thresholds were not selected on Budapest performance.
4. Report AUROC, average precision, Brier score, calibration curves and threshold confusion counts. Compute 95% percentile intervals from 2,000 outcome-stratified bootstrap resamples per evaluation cohort. Difference intervals subtract independently resampled site estimates and are conditional on the fitted models.

The historical cohorts are small and selected for angiographic evaluation. Differences in case mix, measurements and referral patterns may affect site comparisons. Calibration is descriptive, with up to five site-specific quantile bins and no recalibration. This is an exploratory classification benchmark for a portfolio.

## Data provenance

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. [DOI 10.24432/C52P4X](https://doi.org/10.24432/C52P4X). [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The bundled source files are unchanged copies of `processed.cleveland.data` and `processed.hungarian.data` from the [official UCI download](https://archive.ics.uci.edu/static/public/45/heart+disease.zip). Robert Detrano led collection at Cleveland Clinic Foundation and Andras Janosi led collection at the Hungarian Institute of Cardiology in Budapest. See the [original codebook](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names) for the variable definitions. The analysis maps positive outcome codes to 1 and treats the UCI missing-value markers as missing.

## Portfolio use

The initial implementation and dashboard were generated with AI assistance. Before listing the project, reproduce the results and write your own interpretation of the performance across sites. If you publish the project on GitHub, GitHub Pages can serve the dashboard from the repository's `/docs` folder.
