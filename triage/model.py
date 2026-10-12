"""Model definitions and SHAP explanations.

Architecture (stacked):

    clean_text --TF-IDF--> linear text model --(out-of-fold scores)--+
    categorical --one-hot----------------------------------------+-> LR / RF / XGBoost
    numeric    --scaled-----------------------------------------+

The text model turns thousands of sparse n-grams into a few dense scores
(P(priority | text) for classification, E[log hours | text] for regression).
The final model learns how those scores interact with tier, channel, timing etc.
This keeps SHAP explanations readable at the top level, and the linear text
model gives exact per-term attributions underneath.
"""
from collections import defaultdict

import numpy as np
import shap
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURE

SEED = 42


class TextScorer(BaseEstimator, TransformerMixin):
    """TF-IDF + linear model over ticket text, emitted as dense features.

    `fit_transform` returns out-of-fold predictions so the downstream model never
    sees text scores that were fit on its own labels (prevents stacking leakage);
    `transform` uses the model refit on all training rows.
    """

    def __init__(self, task="classification", labels=None, folds=5):
        self.task = task
        self.labels = labels
        self.folds = folds

    def _pipe(self):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=5000, sublinear_tf=True)
        model = (LogisticRegression(max_iter=3000, C=4.0) if self.task == "classification"
                 else Ridge(alpha=1.0))
        return Pipeline([("tfidf", vec), ("linear", model)])

    def fit(self, X, y):
        self.pipe_ = self._pipe().fit(_text(X), y)
        return self

    def fit_transform(self, X, y=None, **_):
        text = _text(X)
        cv = (StratifiedKFold(self.folds, shuffle=True, random_state=SEED) if self.task == "classification"
              else KFold(self.folds, shuffle=True, random_state=SEED))
        method = "predict_proba" if self.task == "classification" else "predict"
        oof = cross_val_predict(clone(self._pipe()), text, y, cv=cv, method=method)
        self.fit(X, y)
        return oof if oof.ndim == 2 else oof[:, None]

    def transform(self, X):
        text = _text(X)
        return self.pipe_.predict_proba(text) if self.task == "classification" else self.pipe_.predict(text)[:, None]

    def get_feature_names_out(self, input_features=None):
        if self.task == "classification":
            return np.array([f"p_{label}" for label in self.labels])
        return np.array(["log_hours"])

    def term_contributions(self, text, output_index=0, k=5):
        """Per-term tfidf * coef contributions to the text model's score; exact because the model is linear."""
        vec, lin = self.pipe_.named_steps["tfidf"], self.pipe_.named_steps["linear"]
        row = vec.transform([text])
        coef = lin.coef_[output_index] if lin.coef_.ndim == 2 else lin.coef_
        idx = row.indices
        contrib = row.data * coef[idx]
        vocab = vec.get_feature_names_out()
        order = np.argsort(-np.abs(contrib))[:k]
        return [{"term": vocab[idx[i]], "weight": round(float(contrib[i]), 4)} for i in order]


def _text(X):
    return X[TEXT_FEATURE] if hasattr(X, "columns") else X


def preprocessor(task, labels=None):
    return ColumnTransformer([
        ("text", TextScorer(task, labels), [TEXT_FEATURE]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])


def classifiers():
    return {
        "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.4,
                                                class_weight="balanced_subsample", n_jobs=-1,
                                                random_state=SEED),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.8, tree_method="hist", eval_metric="mlogloss",
                                 n_jobs=-1, random_state=SEED),
    }


def regressors():
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                               n_jobs=-1, random_state=SEED),
        "xgboost": XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.9,
                                colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED),
    }


def make_pipeline(estimator, task, labels=None):
    return Pipeline([("prep", preprocessor(task, labels)), ("model", estimator)])


def readable_names(prep):
    names = []
    for raw in prep.get_feature_names_out():
        block, name = raw.split("__", 1)
        if block == "text":
            names.append("text signal: " + (f"P({name[2:]})" if name.startswith("p_") else "est. log hours"))
        elif block == "cat":
            col = next(c for c in CATEGORICAL_FEATURES if name.startswith(c + "_"))
            names.append(f"{col} = {name[len(col) + 1:]}")
        else:
            names.append(name)
    return names


class Explainer:
    """SHAP explanations for a fitted `prep -> model` pipeline.

    Values are in the model's raw output space: log-odds (XGBoost / logistic) or
    probability (random forest) for priority, log1p(hours) for resolution time.
    One-hot columns are summed back into their source field; SHAP values are
    additive, so the grouped attributions stay exact.
    """

    def __init__(self, pipeline, background):
        self.prep = pipeline.named_steps["prep"]
        self.model = pipeline.named_steps["model"]
        self.text_scorer = self.prep.named_transformers_["text"]
        self.names = np.array(readable_names(self.prep))
        blocks = np.array([n.split("__", 1)[0] for n in self.prep.get_feature_names_out()])
        self.single_idx = np.flatnonzero(blocks != "cat")
        self.cat_idx = {c: np.flatnonzero(np.char.startswith(self.names.astype(str), c + " = "))
                        for c in CATEGORICAL_FEATURES}
        if isinstance(self.model, (LogisticRegression, Ridge)):
            self.explainer = shap.LinearExplainer(self.model, self.prep.transform(background))
        else:
            self.explainer = shap.TreeExplainer(self.model)

    def shap_values(self, X):
        """(n, features) for regressors, (n, features, classes) for classifiers."""
        values = self.explainer.shap_values(self.prep.transform(X))
        if isinstance(values, list):  # older SHAP versions return one array per class
            values = np.stack(values, axis=-1)
        return np.asarray(values)

    def _grouped(self, values, X):
        """Yield (name, display value, shap column), one-hot fields summed."""
        Xt = self.prep.transform(X)
        for i in self.single_idx:
            name = self.names[i]
            shown = Xt[:, i] if name.startswith("text signal") else X[name].to_numpy()
            yield name, shown, values[:, i]
        for col, idx in self.cat_idx.items():
            yield col, X[col].to_numpy(), values[:, idx].sum(axis=1)

    def top_contributions(self, X, class_index=None, k=6):
        """Top-k contributions per row by |SHAP|; `class_index` picks the explained class for classifiers."""
        values = self.shap_values(X)
        if values.ndim == 3:
            values = values[np.arange(len(values)), :, class_index]
        groups = list(self._grouped(values, X))
        out = []
        for r in range(len(X)):
            row = []
            for name, shown, v in sorted(groups, key=lambda g: -abs(g[2][r]))[:k]:
                categorical = name in self.cat_idx
                row.append({"feature": f"{name} = {shown[r]}" if categorical else name,
                            "value": "" if categorical else _fmt(shown[r]),
                            "shap": round(float(v[r]), 4)})
            out.append(row)
        return out

    def global_importance(self, X, k=20):
        """Mean |SHAP| over a sample, averaged over classes for classifiers."""
        values = self.shap_values(X)
        if values.ndim == 2:
            values = values[:, :, None]
        n_cls = values.shape[2]
        totals = defaultdict(float)
        for c in range(n_cls):
            for name, _, v in self._grouped(values[:, :, c], X):
                totals[name] += float(np.abs(v).mean()) / n_cls
        top = sorted(totals.items(), key=lambda t: -t[1])[:k]
        return [{"feature": name, "mean_abs_shap": round(s, 5)} for name, s in top]


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return round(float(v), 3)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return str(v)
