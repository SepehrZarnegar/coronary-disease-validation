"""Evaluate fixed coronary disease classifiers on an internal and external holdout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42
BOOTSTRAPS = 2000

FIELDS = {
    "age": "Age",
    "sex": "Recorded sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar indicator",
    "restecg": "Resting ECG result",
    "thalach": "Peak exercise heart rate",
    "exang": "Exercise-induced angina",
    "oldpeak": "Exercise ST depression",
    "slope": "Exercise ST slope",
    "ca": "Vessels seen on fluoroscopy",
    "thal": "Thallium test result",
}
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORIES = {"sex": [0, 1], "cp": [1, 2, 3, 4], "fbs": [0, 1],
              "restecg": [0, 1, 2], "exang": [0, 1]}
FEATURES = CONTINUOUS + list(CATEGORIES)
COHORTS = {
    "cleveland": (303, 139, "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b62"
                            "2b4efa0427c8"),
    "hungarian": (294, 106, "d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee"
                            "8076f73fc3d796"),
}


def read_cohort(name):
    path = ROOT / "data" / f"processed.{name}.data"
    rows, positives, checksum = COHORTS[name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"Unexpected source file contents in {path.name}")
    raw = pd.read_csv(path, header=None, na_values=["?", -9], dtype=float)
    if raw.shape != (rows, len(FIELDS) + 1):
        raise ValueError(f"Unexpected source shape in {path.name}: {raw.shape}")
    raw.columns = [*FIELDS, "diagnosis"]
    if not raw["diagnosis"].isin(range(5)).all():
        raise ValueError(f"Missing or invalid outcomes in {path.name}")
    for field, codes in CATEGORIES.items():
        if not raw[field].dropna().isin(codes).all():
            raise ValueError(f"Unexpected coding for {field} in {path.name}")
    outcome = (raw.pop("diagnosis") > 0).astype(int)
    if outcome.sum() != positives:
        raise ValueError(f"Unexpected outcome count in {path.name}")
    return raw, outcome


def make_pipeline(estimator):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(categories=list(CATEGORIES.values()),
                                 handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, CONTINUOUS),
        ("categorical", categorical, list(CATEGORIES)),
    ])
    return Pipeline([("preprocess", preprocess), ("classify", estimator)])


def bootstrap_auc(outcome, scores, rng):
    """Resample positive and negative rows separately within an evaluation cohort."""
    positives = np.flatnonzero(outcome == 1)
    negatives = np.flatnonzero(outcome == 0)
    positive_draws = rng.choice(positives, (BOOTSTRAPS, len(positives)))
    negative_draws = rng.choice(negatives, (BOOTSTRAPS, len(negatives)))
    indices = np.column_stack((positive_draws, negative_draws))
    return np.array([roc_auc_score(outcome[draw], scores[draw]) for draw in indices])


def interval(samples):
    return np.quantile(samples, [.025, .975]).tolist()


def cohort_summary(raw, outcome):
    return {
        "n": len(raw),
        "positives": int(outcome.sum()),
        "ageMean": float(raw.age.mean()),
        "female": int(raw.sex.eq(0).sum()),
        "missing": {name: int(raw[name].isna().sum()) for name in FIELDS},
    }


def calibration_bins(outcome, scores):
    # Quantile edges are site-specific. Ties stay in the same bin.
    cuts = np.unique(np.quantile(scores, [.2, .4, .6, .8]))
    groups = np.searchsorted(cuts, scores, side="right")
    bins = []
    for group in np.unique(groups):
        keep = groups == group
        bins.append([float(scores[keep].mean()), float(outcome[keep].mean()), int(keep.sum())])
    return bins


def threshold_table(outcome, scores):
    rows = []
    positive = outcome == 1
    for step in range(101):
        flagged = scores >= step / 100
        rows.append([
            int(np.sum(flagged & positive)),
            int(np.sum(flagged & ~positive)),
            int(np.sum(~flagged & positive)),
            int(np.sum(~flagged & ~positive)),
        ])
    return rows


def evaluate(outcome, scores):
    fpr, tpr, _ = roc_curve(outcome, scores)
    precision, recall, _ = precision_recall_curve(outcome, scores)
    return {
        "auc": roc_auc_score(outcome, scores),
        "ap": average_precision_score(outcome, scores),
        "brier": brier_score_loss(outcome, scores),
        "meanScore": float(np.mean(scores)),
        "roc": list(zip(fpr.tolist(), tpr.tolist())),
        "pr": list(zip(recall[::-1].tolist(), precision[::-1].tolist())),
        "calibration": calibration_bins(outcome, scores),
        "thresholds": threshold_table(outcome, scores),
    }


def analyze():
    cleveland, y_cleveland = read_cohort("cleveland")
    budapest, y_budapest = read_cohort("hungarian")
    train, holdout, y_train, y_holdout = train_test_split(
        cleveland, y_cleveland, test_size=.30, stratify=y_cleveland, random_state=SEED
    )
    splits = {"internal": (holdout, y_holdout), "external": (budapest, y_budapest)}
    models = {
        "logistic": LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        "boosting": GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=SEED),
    }
    results = {
        "train": cohort_summary(train, y_train),
        "sites": {key: cohort_summary(*split) for key, split in splits.items()},
        "features": [[name, label, name in FEATURES] for name, label in FIELDS.items()],
        "models": {},
    }
    for key, estimator in models.items():
        pipeline = make_pipeline(estimator).fit(train[FEATURES], y_train)
        sites, samples = {}, {}
        for offset, (site, (frame, outcome)) in enumerate(splits.items()):
            probability = pipeline.predict_proba(frame[FEATURES])[:, 1]
            values = outcome.to_numpy()
            sites[site] = evaluate(values, probability)
            samples[site] = bootstrap_auc(values, probability, np.random.default_rng(SEED + offset))
            sites[site]["aucCI"] = interval(samples[site])
        results["models"][key] = {
            **sites,
            "aucChangeCI": interval(samples["external"] - samples["internal"]),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    results = analyze()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    template = (ROOT / "dashboard.template.html").read_text(encoding="utf-8")
    if template.count("/* RESULTS_JSON */") != 1:
        raise ValueError("Dashboard template must contain exactly one data marker")
    html = template.replace("/* RESULTS_JSON */", json.dumps(results, separators=(",", ":")))
    (args.out / "index.html").write_text(html, encoding="utf-8")
    for model, scores in results["models"].items():
        print(f"{model}: Cleveland AUROC {scores['internal']['auc']:.3f}; "
              f"Budapest AUROC {scores['external']['auc']:.3f}")


if __name__ == "__main__":
    main()
