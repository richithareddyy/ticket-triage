"""Rolling, forward-only evaluation on the Apache Jira bugs built by `triage.jira`.

    python -m triage.temporal --data data/jira_bugs.parquet --out artifacts/jira_temporal

Protocol
    Prediction time  FILED_GRACE (15 min) after a bug is filed. Bugs already resolved by
                     then never reach a queue and are left out.
    Inputs           summary, description, component and attachments as of prediction
                     time (rebuilt from the changelog), project, filing time, and how many
                     bugs the reporter filed in the previous 30 days.
    Targets          priority: the priority PRIORITY_HORIZON (14 days) after filing.
                     resolution: hours to first resolution, capped at RESOLUTION_CAP_HOURS.
                     A bug still open at the cap is labelled with the cap, never zero.
    Label maturity   A row trains a model with cutoff c only if its label had settled by c:
                     filed + 14 days for priority, filed + cap for resolution, even when it
                     resolved sooner. Otherwise recent quick fixes would be kept and recent
                     slow ones dropped.
    Windows          Each window trains on every settled bug filed before c and tests on bugs
                     filed in [c, c + 1 year) whose labels had settled by the data snapshot.
    Stacking         Inner text scores are forward-only. Training rows are cut into
                     chronological blocks, and each block is scored by a text model (TF-IDF
                     included) fit only on rows whose labels settled before the block began.
                     The first block has no such rows, so it is a warm-up: it trains the final
                     text model but is left out of the model that combines the features.
    Model choice     The first window picks the model family for each task; later windows
                     report that family unchanged, alongside the baselines.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import PRIORITIES
from .features import CATEGORICAL_FEATURES, TEXT_FEATURE, build_features
from .jira import FILED_GRACE, PRIORITY_HORIZON
from .model import SEED, Explainer, TextScorer, classifiers, make_pipeline, regressors
from .train import cls_metrics, reg_metrics

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

TASKS = ("classification", "regression")
RESOLUTION_CAP_HOURS = 90 * 24
MATURITY = {"classification": PRIORITY_HORIZON, "regression": pd.Timedelta(hours=RESOLUTION_CAP_HOURS)}
# Inputs that don't come from the ticket text; channel and tier are always "unknown" in Jira.
METADATA_NUMERIC = ["hour", "day_of_week", "is_weekend", "is_business_hours", "prior_tickets_30d", "attachments"]
INNER_BLOCKS = 5
MIN_TEXT_ROWS = 500


def in_queue(df):
    """Bugs still open at prediction time."""
    resolved = df["first_resolved_at"]
    return (resolved.isna() | (resolved >= df["created_at"] + FILED_GRACE)).to_numpy()


def labels(df, task):
    """(target, time the target became known) for every row.

    Resolution hours are capped, and a bug that never resolved is labelled with the
    cap, which is what it is once the cap has passed.
    """
    ready = (df["created_at"] + MATURITY[task]).to_numpy()
    if task == "classification":
        return df["priority"].map({p: i for i, p in enumerate(PRIORITIES)}).to_numpy(), ready
    hours = (df["first_resolved_at"] - df["created_at"]).dt.total_seconds() / 3600
    return hours.clip(upper=RESOLUTION_CAP_HOURS).fillna(RESOLUTION_CAP_HOURS).to_numpy(), ready


def window_rows(created, ready, eligible, cutoff, end, snapshot):
    """Training rows settled by the cutoff, and test rows filed in the window and settled by the snapshot."""
    created = np.asarray(created, dtype="datetime64[ns]")
    train = np.flatnonzero(eligible & (ready <= np.datetime64(cutoff)))
    test = np.flatnonzero(eligible & (created >= np.datetime64(cutoff)) & (created < np.datetime64(end))
                          & (ready <= np.datetime64(snapshot)))
    return train, test


def _target(task, y):
    return y if task == "classification" else np.log1p(y)


def _text_scores(pipe, task, text, n_out):
    if task == "regression":
        return pipe.predict(text)[:, None]
    proba = pipe.predict_proba(text)
    out = np.zeros((len(text), n_out))
    out[:, pipe.classes_.astype(int)] = proba  # a class missing from early blocks scores zero
    return out


def forward_text_scores(task, text, y, created, ready, blocks=INNER_BLOCKS, min_rows=MIN_TEXT_ROWS):
    """Forward-only stacking scores for the training rows.

    Rows are split into `blocks` chronological blocks. Block k is scored by a text
    model fit on rows whose labels had settled when block k began, so a score never
    depends on a later ticket or on an outcome that wasn't known yet. Returns the
    scores and a mask of scored rows; the rest (the first block, and any block with
    fewer than `min_rows` settled rows before it) are warm-up rows.
    """
    text, y = np.asarray(text, dtype=object), np.asarray(y)
    created = np.asarray(created, dtype="datetime64[ns]")
    ready = np.asarray(ready, dtype="datetime64[ns]")
    n_out = len(PRIORITIES) if task == "classification" else 1
    scores, scored = np.zeros((len(y), n_out)), np.zeros(len(y), dtype=bool)
    order = np.argsort(created, kind="stable")
    for block in np.array_split(order, blocks)[1:]:
        start = created[block].min()
        fit_rows = ready <= start
        if fit_rows.sum() < min_rows:
            continue
        pipe = TextScorer(task)._pipe().fit(text[fit_rows], _target(task, y[fit_rows]))
        scores[block] = _text_scores(pipe, task, text[block], n_out)
        scored[block] = True
    return scores, scored


def fit_stacked(estimator, task, X, y, inner):
    """Text scores from `inner` feed the final model; the warm-up rows only train the text model."""
    scores, scored = inner
    labels_ = PRIORITIES if task == "classification" else None
    pipe = make_pipeline(clone(estimator), task, labels_, oof=scores[scored])
    pipe.fit(X.iloc[np.flatnonzero(scored)], _target(task, y[scored]))
    text = pipe.named_steps["prep"].named_transformers_["text"]
    text.fit(X, _target(task, y))
    text.oof = None  # the training scores aren't needed to predict, so don't carry them around
    return pipe


def metadata_pipeline(estimator):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), METADATA_NUMERIC),
    ])
    return Pipeline([("prep", prep), ("model", clone(estimator))])


def predict(pipe, task, X):
    return pipe.predict(X) if task == "classification" else np.expm1(pipe.predict(X)).clip(0.25)


def score(task, y, pred):
    return cls_metrics(y, pred) if task == "classification" else reg_metrics(y, pred)


def candidates(task):
    return classifiers() if task == "classification" else regressors()


def evaluate_window(task, X, y, created, ready, train, test, names):
    """Fit the named candidates on `train` with forward-only stacking and score them on `test`."""
    Xtr, ytr, Xte, yte = X.iloc[train].reset_index(drop=True), y[train], X.iloc[test], y[test]
    inner = forward_text_scores(task, Xtr[TEXT_FEATURE], ytr, created[train], ready[train])
    results, fitted = {}, {}
    for name in names:
        t = time.time()
        fitted[name] = fit_stacked(candidates(task)[name], task, Xtr, ytr, inner)
        results[name] = {**score(task, yte, predict(fitted[name], task, Xte)),
                         "train_seconds": round(time.time() - t, 1)}
    return results, fitted, inner


def baselines(task, estimator, X, y, train, test):
    """Constant, text-only and metadata-only predictions; metadata-only reuses the chosen model family."""
    Xtr, ytr, Xte, yte = X.iloc[train], y[train], X.iloc[test], y[test]
    if task == "classification":
        constant = "baseline_majority", DummyClassifier(strategy="most_frequent")
    else:
        constant = "baseline_median", DummyRegressor(strategy="median")
    results = {constant[0]: score(task, yte, constant[1].fit(Xtr, ytr).predict(Xte))}
    text_only = TextScorer(task).fit(Xtr, _target(task, ytr)).pipe_
    pred = text_only.predict(Xte[TEXT_FEATURE])
    results["baseline_text_only"] = score(task, yte, pred if task == "classification" else np.expm1(pred).clip(0.25))
    meta = metadata_pipeline(estimator).fit(Xtr, _target(task, ytr))
    results["baseline_metadata_only"] = score(task, yte, predict(meta, task, Xte))
    return results


def shap_check(pipe, task, X, background):
    """Largest gap between grouped SHAP values + base value and the model's raw output."""
    exp = Explainer(pipe, background)
    values = exp.shap_values(X)
    model, Xt = pipe.named_steps["model"], pipe.named_steps["prep"].transform(X)
    base = np.ravel(exp.explainer.expected_value)
    if values.ndim == 2:
        values, base = values[:, :, None], base[:1]
        raw = model.predict(Xt)[:, None]
    elif hasattr(model, "get_booster"):
        raw = model.predict(Xt, output_margin=True)
    elif isinstance(model, LogisticRegression):
        raw = model.decision_function(Xt)
    else:
        raw = model.predict_proba(Xt)
    grouped = np.stack([np.column_stack([v for _, _, v in exp._grouped(values[:, :, c], X)]).sum(axis=1)
                        for c in range(values.shape[2])], axis=1)
    return float(np.abs(grouped + base - raw).max()), exp.global_importance(X)


def yearly_windows(first, last):
    return [(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")) for y in range(first, last + 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/jira_bugs.parquet")
    ap.add_argument("--features", default="data/jira_bug_features.parquet", help="cache for built features")
    ap.add_argument("--out", default="artifacts/jira_temporal")
    ap.add_argument("--first-year", type=int, default=2019, help="this window picks the model family")
    ap.add_argument("--last-year", type=int, default=2025)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.data)
    snapshot = pd.Timestamp(df.attrs["snapshot_at"])
    if os.path.exists(args.features):
        X = pd.read_parquet(args.features)
    else:
        t = time.time()
        X = build_features(df)
        X.to_parquet(args.features)
        print(f"built features in {time.time() - t:.0f}s")
    created = df["created_at"].to_numpy()
    eligible = in_queue(df)
    windows = yearly_windows(args.first_year, args.last_year)
    print(f"{len(df):,} bugs, {eligible.sum():,} open at prediction time; snapshot {snapshot}")

    report = {"protocol": __doc__.split("Protocol", 1)[1].strip(), "snapshot_at": snapshot.isoformat(),
              "population": {"bugs": len(df), "open_at_prediction_time": int(eligible.sum()),
                             "never_resolved": int(df["first_resolved_at"].isna().sum()),
                             "priority_changed_after_filing": int((df["priority"] != df["filed_priority"]).sum())},
              "settings": {"filed_grace_minutes": FILED_GRACE.total_seconds() / 60,
                           "priority_horizon_days": PRIORITY_HORIZON.days,
                           "resolution_cap_hours": RESOLUTION_CAP_HOURS, "inner_blocks": INNER_BLOCKS},
              "tasks": {}}
    for task in TASKS:
        name = "priority" if task == "classification" else "resolution_time"
        y, ready = labels(df, task)
        rows, chosen, last = [], None, None
        for i, (cutoff, end) in enumerate(windows):
            train, test = window_rows(created, ready, eligible, cutoff, end, snapshot)
            names = [chosen] if chosen else list(candidates(task))
            results, fitted, inner = evaluate_window(task, X, y, created, ready, train, test, names)
            if chosen is None:
                metric = "macro_f1" if task == "classification" else "mae_hours"
                pick = max if task == "classification" else min
                chosen = pick(fitted, key=lambda m: results[m][metric])
            results.update(baselines(task, candidates(task)[chosen], X, y, train, test))
            row = {"window": f"{cutoff:%Y-%m-%d} to {end:%Y-%m-%d}", "role": "selection" if i == 0 else "test",
                   "train_rows": len(train), "warm_up_rows": int((~inner[1]).sum()), "test_rows": len(test),
                   "results": results}
            if task == "regression":
                row["test_share_at_cap"] = round(float((y[test] >= RESOLUTION_CAP_HOURS).mean()), 4)
                row["test_share_never_resolved"] = round(float(df["first_resolved_at"].iloc[test].isna().mean()), 4)
            rows.append(row)
            last = (fitted[chosen], train, test)
            print(f"[{name}] {row['window']} train={len(train):,} test={len(test):,} "
                  + " ".join(f"{m}={_headline(task, r)}" for m, r in results.items()), flush=True)

        pipe, train, test = last
        background = X.iloc[train].sample(200, random_state=SEED)
        sample = X.iloc[test].sample(min(300, len(test)), random_state=SEED)
        gap, importance = shap_check(pipe, task, sample, background)
        report["tasks"][name] = {"selected_model": chosen, "selected_on": rows[0]["window"],
                                 "windows": rows, "summary": _summarise(task, rows, chosen),
                                 "shap": {"max_additivity_gap": gap, "window": rows[-1]["window"],
                                          "importance": importance}}
        print(f"[{name}] selected {chosen}; SHAP additivity gap {gap:.2e}")

    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({k: v["summary"] for k, v in report["tasks"].items()}, indent=2))


def _headline(task, r):
    return r["macro_f1"] if task == "classification" else r["mae_hours"]


def _summarise(task, rows, chosen):
    """Mean of each method over the test windows (selection window excluded), weighted equally."""
    tests = [r for r in rows if r["role"] == "test"]
    keys = ("macro_f1", "accuracy") if task == "classification" else ("mae_hours", "median_ae_hours")
    out = {}
    for method in [chosen] + [m for m in tests[0]["results"] if m.startswith("baseline")]:
        out[method] = {k: round(float(np.mean([r["results"][method][k] for r in tests])), 4) for k in keys}
    metric = keys[0]
    best = (max if task == "classification" else min)(
        (m for m in out if m.startswith("baseline")), key=lambda m: out[m][metric])
    out["windows_where_model_beats_every_baseline"] = sum(
        all((r["results"][chosen][metric] > r["results"][b][metric]) if task == "classification"
            else (r["results"][chosen][metric] < r["results"][b][metric])
            for b in r["results"] if b.startswith("baseline")) for r in tests)
    out["test_windows"] = len(tests)
    out["strongest_baseline"] = best
    return out


if __name__ == "__main__":
    main()
