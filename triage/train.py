"""Train and compare priority classifiers and resolution-time regressors.

    python -m triage.train --data data/tickets.csv
    python -m triage.train --data data/tickets.csv --features data/features.parquet   # Spark features

Each task trains Logistic Regression / Ridge, Random Forest and XGBoost on a
train split, selects the best on validation (macro-F1 / MAE), refits the winner on
train+validation and reports held-out test metrics. Writes:
    artifacts/model.joblib      - bundle loaded by the API
    artifacts/metrics.json      - model comparison + final test metrics
    artifacts/shap_*.png        - global SHAP importance plots
"""
import argparse
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, median_absolute_error, r2_score)
from sklearn.model_selection import train_test_split

from .data import PRIORITIES
from .features import FEATURE_COLUMNS, build_features
from .model import SEED, Explainer, classifiers, make_pipeline, regressors

# NumPy 2.0.x on macOS (Accelerate BLAS) emits spurious matmul warnings; results are unaffected.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def load(data_path, features_path=None):
    raw = pd.read_csv(data_path)
    if features_path:
        feats = pd.read_parquet(features_path).set_index("ticket_id")
        X = feats.loc[raw["ticket_id"], FEATURE_COLUMNS].reset_index(drop=True)
        print(f"loaded Spark features from {features_path}")
    else:
        t = time.time()
        X = build_features(raw)
        print(f"built features with pandas in {time.time() - t:.1f}s")
    y_cls = raw["priority"].map({p: i for i, p in enumerate(PRIORITIES)}).to_numpy()
    y_reg = raw["resolution_hours"].to_numpy()
    return X, y_cls, y_reg


def cls_metrics(y, pred):
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "weighted_f1": round(f1_score(y, pred, average="weighted"), 4)}


def reg_metrics(y_hours, pred_hours):
    return {"mae_hours": round(mean_absolute_error(y_hours, pred_hours), 3),
            "median_ae_hours": round(float(median_absolute_error(y_hours, pred_hours)), 3),
            "rmse_hours": round(float(np.sqrt(mean_squared_error(y_hours, pred_hours))), 3),
            "r2_log": round(r2_score(np.log1p(y_hours), np.log1p(pred_hours)), 4)}


def compare(task, candidates, X, y, split, fit_target, predict, score, better, pipe_args):
    tr, va, te = split
    results = {}
    for name, est in candidates.items():
        t = time.time()
        pipe = make_pipeline(est, *pipe_args).fit(X.iloc[tr], fit_target(y[tr]))
        results[name] = {"val": score(y[va], predict(pipe, X.iloc[va])),
                         "test": score(y[te], predict(pipe, X.iloc[te])),
                         "train_seconds": round(time.time() - t, 1)}
        print(f"  [{task}] {name:<20} val={results[name]['val']}  ({results[name]['train_seconds']}s)")
    best = better(results)
    print(f"  [{task}] best on validation: {best}")
    return best, results


def save_importance_plot(importance, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = importance[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(items) + 1.2))
    ax.barh([i["feature"] for i in items], [i["mean_abs_shap"] for i in items], color="#3b6fd8")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/tickets.csv")
    ap.add_argument("--features", default=None, help="optional Spark-built features parquet")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, y_cls, y_reg = load(args.data, args.features)
    idx = np.arange(len(X))
    tr, rest = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=y_cls)
    va, te = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=y_cls[rest])
    split = (tr, va, te)
    trva = np.concatenate([tr, va])
    print(f"train={len(tr):,} val={len(va):,} test={len(te):,}")

    # --- Priority classification -------------------------------------------------
    cls_cands = {"baseline_majority": DummyClassifier(strategy="most_frequent"), **classifiers()}
    best_cls, cls_results = compare(
        "priority", cls_cands, X, y_cls, split, fit_target=lambda y: y,
        predict=lambda p, X_: p.predict(X_), score=cls_metrics,
        better=lambda r: max((k for k in r if k != "baseline_majority"), key=lambda k: r[k]["val"]["macro_f1"]),
        pipe_args=("classification", PRIORITIES))
    cls_pipe = make_pipeline(classifiers()[best_cls], "classification", PRIORITIES).fit(X.iloc[trva], y_cls[trva])
    cls_pred = cls_pipe.predict(X.iloc[te])

    # --- Resolution-time regression (trained on log1p hours: heavy right skew) -----
    reg_cands = {"baseline_median": DummyRegressor(strategy="median"), **regressors()}
    best_reg, reg_results = compare(
        "resolution", reg_cands, X, y_reg, split, fit_target=np.log1p,
        predict=lambda p, X_: np.expm1(p.predict(X_)).clip(0.25), score=reg_metrics,
        better=lambda r: min((k for k in r if k != "baseline_median"), key=lambda k: r[k]["val"]["mae_hours"]),
        pipe_args=("regression",))
    reg_pipe = make_pipeline(regressors()[best_reg], "regression").fit(X.iloc[trva], np.log1p(y_reg[trva]))
    reg_pred = np.expm1(reg_pipe.predict(X.iloc[te])).clip(0.25)

    # --- SHAP -------------------------------------------------------------------
    background = X.iloc[tr].sample(200, random_state=SEED)
    sample = X.iloc[te].sample(min(500, len(te)), random_state=SEED)
    t = time.time()
    cls_imp = Explainer(cls_pipe, background).global_importance(sample)
    reg_imp = Explainer(reg_pipe, background).global_importance(sample)
    print(f"SHAP global importance in {time.time() - t:.1f}s")
    save_importance_plot(cls_imp, f"Priority ({best_cls}) - top SHAP features", f"{args.out}/shap_priority.png")
    save_importance_plot(reg_imp, f"Resolution time ({best_reg}) - top SHAP features",
                         f"{args.out}/shap_resolution.png")

    metrics = {
        "data": {"rows": len(X), "train": len(tr), "val": len(va), "test": len(te)},
        "priority": {
            "selected_model": best_cls, "comparison": cls_results,
            "final_test": cls_metrics(y_cls[te], cls_pred),
            "classification_report": classification_report(y_cls[te], cls_pred, target_names=PRIORITIES,
                                                           output_dict=True),
            "confusion_matrix": {"labels": PRIORITIES,
                                 "matrix": confusion_matrix(y_cls[te], cls_pred).tolist()},
            "shap_importance": cls_imp,
        },
        "resolution_time": {
            "selected_model": best_reg, "comparison": reg_results,
            "final_test": reg_metrics(y_reg[te], reg_pred), "shap_importance": reg_imp,
        },
    }
    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump({"priority_pipeline": cls_pipe, "resolution_pipeline": reg_pipe, "classes": PRIORITIES,
                 "background": background, "models": {"priority": best_cls, "resolution": best_reg},
                 "metrics": {"priority": metrics["priority"]["final_test"],
                             "resolution_time": metrics["resolution_time"]["final_test"]},
                 "trained_at": pd.Timestamp.now(tz="UTC").isoformat()},
                f"{args.out}/model.joblib", compress=3)

    print("\n=== Held-out test ===")
    print(f"priority   ({best_cls}): {metrics['priority']['final_test']}")
    print(f"resolution ({best_reg}): {metrics['resolution_time']['final_test']}")
    print(classification_report(y_cls[te], cls_pred, target_names=PRIORITIES, digits=3))
    print(f"artifacts -> {args.out}/")


if __name__ == "__main__":
    main()
