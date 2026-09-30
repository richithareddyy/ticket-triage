"""Streamlit front end for the triage API.

    API_URL=http://localhost:5050 streamlit run ui/streamlit_app.py   # talk to a running API
    streamlit run ui/streamlit_app.py                                 # no API_URL: run the API in-process

Without API_URL the app loads the same Flask app in-process (via its test client),
so validation and responses are identical to the HTTP API. This is how the free
single-process demo runs.
"""
import datetime as dt
import math
import os
import sys
from html import escape

import altair as alt
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theme  # noqa: E402

API_URL = os.environ.get("API_URL", "").rstrip("/")
CHANNELS = ["email", "web", "chat", "phone"]
TIERS = ["free", "pro", "enterprise"]
PRODUCTS = ["Dashboard", "Mobile App", "API", "Billing Portal", "Reports", "Integrations Hub"]
CATEGORIES = ["outage", "security", "data_loss", "bug", "performance", "login_access", "billing",
              "integration", "how_to", "feature_request"]
REPO_URL = "https://github.com/richithareddyy/ticket-triage"
EXAMPLES = {
    "Enterprise outage": dict(
        subject="Production dashboard is down",
        description="Nothing loads, we get HTTP 503 on every request. All of our users are affected. "
                    "This is urgent!",
        channel="phone", product="Dashboard", customer_tier="enterprise", category="outage",
        prior_tickets_30d=2, attachments=1),
    "Duplicate charge": dict(
        subject="Charged twice",
        description="We were charged twice this month for the Pro plan. Please refund the duplicate. Thank you.",
        channel="email", product="Billing Portal", customer_tier="pro", category="billing",
        prior_tickets_30d=0, attachments=1),
    "Feature request": dict(
        subject="Would love dark mode",
        description="Please consider adding a dark mode to Reports. Our team would love it. Thanks!",
        channel="web", product="Reports", customer_tier="free", category="feature_request",
        prior_tickets_30d=0, attachments=0),
    "Repeat login issue": dict(
        subject="CAN'T LOG IN AGAIN",
        description="My two factor code is never accepted and I am now locked out. This is urgent. "
                    "Third time reporting this!!!",
        channel="chat", product="Mobile App", customer_tier="free", category="login_access",
        prior_tickets_30d=4, attachments=0),
}
MODEL_NAMES = {"xgboost": "XGBoost", "random_forest": "Random Forest", "logistic_regression": "Logistic Regression",
               "ridge": "Ridge", "baseline_majority": "Baseline (majority)", "baseline_median": "Baseline (median)"}

st.set_page_config(page_title="Ticket Triage", page_icon=":material/confirmation_number:", layout="wide",
                   initial_sidebar_state="collapsed")
theme.apply()


class ApiError(Exception):
    pass


@st.cache_resource
def local_client():
    from api.app import create_app
    return create_app().test_client()


def call(method, path, params=None, json=None):
    """Returns (status_code, json body) from the remote API or the in-process app."""
    if not API_URL:
        resp = local_client().open(path, method=method, query_string=params, json=json)
        return resp.status_code, resp.get_json()
    try:
        r = requests.request(method, f"{API_URL}{path}", params=params, json=json, timeout=30)
        return r.status_code, r.json()
    except (requests.RequestException, ValueError) as e:
        raise ApiError(str(e)) from e


@st.cache_data(ttl=60)
def get(path):
    status, body = call("GET", path)
    if status != 200:
        raise ApiError(body.get("error", f"HTTP {status}") if body else f"HTTP {status}")
    return body


FEATURE_LABELS = {
    "text_len": "Text length", "word_count": "Word count", "exclamation_count": "Exclamation marks",
    "question_count": "Question marks", "upper_ratio": "Share of capitals", "urgency_count": "Urgency words",
    "has_error_code": "Mentions an error code", "sentiment": "Sentiment", "hour": "Hour opened",
    "day_of_week": "Day of week", "is_weekend": "Opened on a weekend", "is_business_hours": "Business hours",
    "prior_tickets_30d": "Tickets, last 30 days", "attachments": "Attachments", "channel": "Channel",
    "product": "Product", "customer_tier": "Customer tier", "category": "Category",
}


def feature_label(name, value=""):
    """Readable name for a model feature; the API keeps its technical names."""
    if name.startswith("text signal: P("):
        text = f"Text reads {name[len('text signal: P('):-1]}"
        return f"{text} ({float(value):.0%})" if value != "" else text
    if name.startswith("text signal"):
        return f"Text-based estimate ({format_hours(math.expm1(float(value)))})" if value != "" \
            else "Text-based estimate"
    if " = " in name:
        col, val = name.split(" = ", 1)
        return f"{FEATURE_LABELS.get(col, col)}: {label(val)}"
    pretty = FEATURE_LABELS.get(name, name)
    return f"{pretty} ({value})" if value != "" else pretty


def label(name):
    return MODEL_NAMES.get(name, name.replace("_", " ").capitalize())


def format_hours(hours):
    if hours < 1:
        return f"{hours * 60:.0f} min"
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} days"


# --- Charts ------------------------------------------------------------------------

def contributions_chart(items, positive_label, negative_label):
    """Horizontal SHAP bars, largest impact first; warm raises the prediction, teal lowers it."""
    df = pd.DataFrame(items)
    df["label"] = df.apply(lambda r: feature_label(r["feature"], r["value"]), axis=1)
    df["direction"] = df["shap"].map(lambda v: positive_label if v > 0 else negative_label)
    chart = alt.Chart(df).mark_bar(height=12).encode(
        x=alt.X("shap:Q", title="SHAP contribution"),
        y=alt.Y("label:N", sort=df.sort_values("shap", key=abs, ascending=False)["label"].tolist(), title=None),
        color=alt.Color("direction:N", title=None,
                        scale=alt.Scale(domain=[positive_label, negative_label], range=[theme.UP, theme.DOWN])),
        tooltip=[alt.Tooltip("feature", title="Feature"), alt.Tooltip("value", title="Value"),
                 alt.Tooltip("shap:Q", title="SHAP", format="+.3f")],
    ).properties(height=alt.Step(26))
    st.altair_chart(theme.style_chart(chart), use_container_width=True)


def importance_chart(items):
    df = pd.DataFrame(items)
    df["feature"] = df["feature"].map(feature_label)
    chart = alt.Chart(df).mark_bar(height=11, color=theme.ACCENT).encode(
        x=alt.X("mean_abs_shap:Q", title="Mean |SHAP|"),
        y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("feature", title="Feature"), alt.Tooltip("mean_abs_shap:Q", title="Mean |SHAP|",
                                                                      format=".3f")],
    ).properties(height=alt.Step(23))
    st.altair_chart(theme.style_chart(chart), use_container_width=True)


def confusion_chart(labels, matrix):
    rows = [{"true": t, "pred": p, "count": matrix[i][j]}
            for i, t in enumerate(labels) for j, p in enumerate(labels)]
    df = pd.DataFrame(rows)
    df["share"] = df["count"] / df.groupby("true")["count"].transform("sum")
    base = alt.Chart(df).encode(
        x=alt.X("pred:N", sort=labels, title="Predicted", axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("true:N", sort=labels, title="Actual"),
    )
    heat = base.mark_rect(stroke=theme.PAPER, strokeWidth=2).encode(
        color=alt.Color("share:Q", legend=None, scale=alt.Scale(range=[theme.PANEL, theme.ACCENT], domain=[0, 1])),
        tooltip=[alt.Tooltip("true", title="Actual"), alt.Tooltip("pred", title="Predicted"),
                 alt.Tooltip("count:Q", title="Tickets"), alt.Tooltip("share:Q", title="Share of row", format=".0%")],
    )
    text = base.mark_text(fontSize=13, fontWeight=600).encode(
        text="count:Q",
        color=alt.condition(alt.datum.share > 0.5, alt.value("#FFFFFF"), alt.value(theme.INK)),
    )
    st.altair_chart(theme.style_chart((heat + text).properties(height=240)), use_container_width=True)


def key_terms(terms):
    if not terms:
        return
    words = "".join(
        f'<span style="color:{theme.UP if t["weight"] > 0 else theme.DOWN}">{escape(t["term"])} '
        f'{t["weight"]:+.2f}</span>' for t in terms)
    theme.html(f'<div class="tt-small" style="margin-top:.25rem">Words behind the text signal</div>'
               f'<div class="tt-terms">{words}</div>')


# --- Header ------------------------------------------------------------------------

try:
    health = get("/health")
except ApiError as e:
    theme.html('<div class="tt-title">Ticket Triage</div>')
    st.error(f"Can't reach the prediction API at `{API_URL}`. {e}")
    st.stop()

models, metrics = health["models"], health.get("metrics", {})
head_l, head_r = st.columns([3, 2], vertical_alignment="bottom")
head_l.markdown('<div class="tt-title">Ticket Triage</div>'
                '<div class="tt-sub">Priority and time-to-resolution for incoming support tickets, '
                'with the reasons behind each call.</div>', unsafe_allow_html=True)
f1 = metrics.get("priority", {}).get("macro_f1")
head_r.markdown(
    f'<div class="tt-meta"><span class="dot">●</span> {label(models["priority"])} models '
    f'{"via API" if API_URL else "loaded"}<br>trained {health["trained_at"][:10]}'
    + (f' · test macro-F1 {f1:.3f}' if f1 else "") + "</div>", unsafe_allow_html=True)

tab_triage, tab_eval = st.tabs(["Triage", "Model evaluation"])

# --- Triage: ticket → context → prediction → explanation ---------------------------

with tab_triage:
    work, side = st.columns([7, 5], gap="large")

    with work:
        h, pick = st.columns([3, 2], vertical_alignment="bottom")
        h.markdown('<div class="tt-section">Ticket</div>', unsafe_allow_html=True)
        sample = pick.selectbox("Sample ticket", list(EXAMPLES), format_func=lambda k: f"Sample: {k}",
                                label_visibility="collapsed")
        ex = EXAMPLES[sample]

        with st.form("ticket", border=False):
            subject = st.text_input("Subject", ex["subject"], max_chars=200)
            description = st.text_area("Description", ex["description"], height=108, max_chars=2000)

            theme.html('<div class="tt-section context">Context</div>')
            c1, c2, c3 = st.columns(3)
            channel = c1.selectbox("Channel", CHANNELS, index=CHANNELS.index(ex["channel"]), format_func=label)
            tier = c2.selectbox("Customer tier", TIERS, index=TIERS.index(ex["customer_tier"]), format_func=label)
            product = c3.selectbox("Product", PRODUCTS, index=PRODUCTS.index(ex["product"]))
            c1, c2, c3 = st.columns(3)
            category = c1.selectbox("Category (customer's pick)", CATEGORIES,
                                    index=CATEGORIES.index(ex["category"]), format_func=label)
            date = c2.date_input("Opened", dt.date.today())
            time = c3.time_input("Time", dt.time(10, 30))
            c1, c2, _ = st.columns(3)
            prior = c1.number_input("Tickets, last 30 days", 0, 100, ex["prior_tickets_30d"])
            attachments = c2.number_input("Attachments", 0, 50, ex["attachments"])
            submitted = st.form_submit_button("Run triage", type="primary", width="stretch")

    res = None
    with side:
        with st.container(key="panel-prediction"):
            theme.html('<div class="tt-small">Prediction</div>')
            if submitted:
                ticket = dict(subject=subject, description=description, channel=channel, customer_tier=tier,
                              product=product, category=category, prior_tickets_30d=int(prior),
                              attachments=int(attachments),
                              created_at=dt.datetime.combine(date, time).isoformat())
                if not (subject.strip() or description.strip()):
                    st.warning("Add a subject or description first.")
                else:
                    try:
                        with st.spinner("Scoring..."):
                            status, body = call("POST", "/predict", params={"explain": "true"}, json=ticket)
                        if status == 200:
                            res = body
                        else:
                            details = body.get("details") or [body.get("error", f"HTTP {status}")]
                            st.error("Couldn't score this ticket:\n" + "\n".join(f"- {d}" for d in details))
                    except ApiError as e:
                        st.error(f"The prediction request failed. {e}")

            if res:
                breach = res["sla_breach_risk"]
                sla = (f'{res["sla_target_hours"]} h target<small class="{"warn" if breach else "ok"}">'
                       f'{"at risk" if breach else "on track"}</small>')
                theme.html(
                    theme.priority_block(res["priority"], f"{res['confidence']:.0%} confidence")
                    + theme.split(res["priority_probabilities"])
                    + '<div class="tt-rows">'
                    + theme.row("Expected resolution", format_hours(res["resolution_hours"]))
                    + theme.row("SLA", sla)
                    + "</div>"
                    + f'<div class="tt-muted" style="margin-top:.6rem">Scored in {res["latency_ms"]:.0f} ms</div>')
            else:
                theme.html(
                    '<div class="tt-empty">Run triage to score this ticket.</div>'
                    + '<div class="tt-rows">'
                    + theme.row("Priority", "—", empty=True)
                    + theme.row("Expected resolution", "—", empty=True)
                    + theme.row("SLA", "—", empty=True)
                    + "</div>")

    if res:
        exp = res["explanation"]
        theme.html('<div class="tt-h">Why this prediction</div>'
                   '<div class="tt-muted">SHAP contributions from the models. Priority is in log-odds, '
                   'resolution time in log-hours.</div>')
        left, right = st.columns(2, gap="large")
        with left:
            theme.html(f'<div class="tt-h3">Priority: {escape(res["priority"])}</div>')
            contributions_chart(exp["priority"], "raises", "lowers")
            key_terms(exp["priority_key_terms"])
        with right:
            theme.html(f'<div class="tt-h3">Resolution: {format_hours(res["resolution_hours"])}</div>')
            contributions_chart(exp["resolution_time"], "slower", "faster")
            key_terms(exp["resolution_key_terms"])

# --- Model evaluation ---------------------------------------------------------------

with tab_eval:
    try:
        m = get("/model")
    except ApiError:
        theme.html('<div class="tt-muted">Evaluation results aren\'t available from the API.</div>')
        st.stop()
    p, rt = m["priority"], m["resolution_time"]
    pf, rf = p["final_test"], rt["final_test"]
    base_f1 = p["comparison"].get("baseline_majority", {}).get("test", {}).get("macro_f1")
    base_mae = rt["comparison"].get("baseline_median", {}).get("test", {}).get("mae_hours")

    theme.html('<div class="tt-muted" style="margin-top:.25rem">Held-out test set: tickets never seen during '
               'training or model selection.</div>'
               + theme.stats([
                   ("Priority macro-F1", f"{pf['macro_f1']:.3f}", f"baseline {base_f1:.3f}" if base_f1 else ""),
                   ("Priority accuracy", f"{pf['accuracy']:.1%}", ""),
                   ("Resolution MAE", f"{rf['mae_hours']:.1f} h", f"baseline {base_mae:.1f} h" if base_mae else ""),
                   ("Median error", f"{rf['median_ae_hours']:.1f} h", f"R² (log) {rf['r2_log']:.2f}"),
               ]))

    def comparison_table(block, columns, fmt):
        rows = []
        for name, v in block["comparison"].items():
            row = {"Model": label(name) + ("  ✓" if name == block["selected_model"] else "")}
            row.update({title: v["val"][key] for key, title in columns.items()})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format(fmt), width="stretch")

    theme.html('<div class="tt-h">Models compared</div>'
               '<div class="tt-muted">Validation scores. ✓ marks the model in use.</div>')
    left, right = st.columns(2, gap="large")
    with left:
        theme.html('<div class="tt-h3">Priority</div>')
        comparison_table(p, {"macro_f1": "Macro-F1", "accuracy": "Accuracy"},
                         {"Macro-F1": "{:.3f}", "Accuracy": "{:.1%}"})
    with right:
        theme.html('<div class="tt-h3">Resolution time</div>')
        comparison_table(rt, {"mae_hours": "MAE (h)", "median_ae_hours": "Median AE (h)"},
                         {"MAE (h)": "{:.2f}", "Median AE (h)": "{:.2f}"})

    left, right = st.columns([5, 6], gap="large")
    with left:
        theme.html('<div class="tt-h">Where it goes wrong</div>'
                   '<div class="tt-muted">Errors sit next to the diagonal.</div>')
        confusion_chart(p["confusion_matrix"]["labels"], p["confusion_matrix"]["matrix"])
    with right:
        theme.html('<div class="tt-h">What drives predictions</div>'
                   '<div class="tt-muted">Mean impact per feature on held-out tickets.</div>')
        target = st.radio("Target", ["Priority", "Resolution time"], horizontal=True, label_visibility="collapsed")
        importance_chart((p if target == "Priority" else rt)["shap_importance"][:10])

theme.html(f'<div class="tt-foot">Trained on synthetic tickets with a known structure; real support data is '
           f'private. <a href="{REPO_URL}">Source</a></div>')
