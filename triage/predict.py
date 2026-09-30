"""Load the trained bundle and score tickets, with optional SHAP explanations."""
import joblib
import numpy as np
import pandas as pd

from .features import build_features
from .model import Explainer

# Target first-resolution SLAs per priority (hours). Used to flag breach risk.
SLA_HOURS = {"Critical": 4, "High": 24, "Medium": 72, "Low": 168}


class Predictor:
    def __init__(self, path="artifacts/model.joblib"):
        bundle = joblib.load(path)
        self.cls = bundle["priority_pipeline"]
        self.reg = bundle["resolution_pipeline"]
        self.classes = bundle["classes"]
        self.info = {k: bundle[k] for k in ("models", "metrics", "trained_at")}
        self._background = bundle["background"]
        self._explainers = None

    @property
    def explainers(self):
        if self._explainers is None:
            self._explainers = (Explainer(self.cls, self._background), Explainer(self.reg, self._background))
        return self._explainers

    def predict(self, tickets, explain=False, top_k=6):
        df = pd.DataFrame(tickets)
        if "created_at" not in df:
            df["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None).isoformat()
        df["created_at"] = df["created_at"].fillna(pd.Timestamp.now(tz="UTC").tz_localize(None).isoformat())
        for col, default in (("prior_tickets_30d", 0), ("attachments", 0), ("category", "unknown")):
            if col not in df:
                df[col] = default
        X = build_features(df)

        proba = self.cls.predict_proba(X)
        cls_idx = proba.argmax(axis=1)
        hours = np.expm1(self.reg.predict(X)).clip(0.25)

        if explain:
            cls_exp, reg_exp = self.explainers
            pri_contrib = cls_exp.top_contributions(X, class_index=cls_idx, k=top_k)
            res_contrib = reg_exp.top_contributions(X, k=top_k)
            texts = X["clean_text"].tolist()

        results = []
        for i in range(len(X)):
            priority = self.classes[cls_idx[i]]
            r = {
                "priority": priority,
                "confidence": round(float(proba[i, cls_idx[i]]), 4),
                "priority_probabilities": {c: round(float(p), 4) for c, p in zip(self.classes, proba[i])},
                "resolution_hours": round(float(hours[i]), 2),
                "sla_target_hours": SLA_HOURS[priority],
                "sla_breach_risk": bool(hours[i] > SLA_HOURS[priority]),
            }
            if explain:
                r["explanation"] = {
                    "priority": pri_contrib[i],
                    "resolution_time": res_contrib[i],
                    # Words behind the "text signal" features, from the linear text models.
                    "priority_key_terms": cls_exp.key_terms(texts[i], output_index=int(cls_idx[i])),
                    "resolution_key_terms": reg_exp.key_terms(texts[i]),
                }
            results.append(r)
        return results
