"""Streamlit front end for the triage API.

    API_URL=http://localhost:5050 streamlit run ui/streamlit_app.py   # talk to a running API
    streamlit run ui/streamlit_app.py                                 # no API_URL: run the API in-process

Without API_URL the app loads the same Flask app in-process (via its test client),
so validation and responses are identical to the HTTP API. This is how the free
single-process demo runs.
"""
import datetime as dt
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
PRIORITY_ORDER = list(theme.PRIORITY)
CHANNELS = ["email", "web", "chat", "phone"]
TIERS = ["free", "pro", "enterprise"]
PRODUCTS = ["Dashboard", "Mobile App", "API", "Billing Portal", "Reports", "Integrations Hub"]
CATEGORIES = ["outage", "security", "data_loss", "bug", "performance", "login_access", "billing",
              "integration", "how_to", "feature_request"]
REPO_URL = "https://github.com/richithareddyy/ticket-triage"
EXAMPLES = {
    "Production outage (enterprise)": dict(
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
    "Angry repeat customer (free tier)": dict(
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
    df["label"] = df.apply(lambda r: r["feature"] + (f" ({r['value']})" if r["value"] != "" else ""), axis=1)
    df["direction"] = df["shap"].map(lambda v: positive_label if v > 0 else negative_label)
    chart = alt.Chart(df).mark_bar(height=14).encode(
        x=alt.X("shap:Q", title="SHAP contribution"),
        y=alt.Y("label:N", sort=df.sort_values("shap", key=abs, ascending=False)["label"].tolist(), title=None),
        color=alt.Color("direction:N", title=None, legend=alt.Legend(orient="bottom", labelLimit=250),
                        scale=alt.Scale(domain=[positive_label, negative_label], range=[theme.UP, theme.DOWN])),
        tooltip=[alt.Tooltip("feature", title="Feature"), alt.Tooltip("value", title="Value"),
                 alt.Tooltip("shap:Q", title="SHAP", format="+.3f")],
    ).properties(height=alt.Step(28))
    st.altair_chart(theme.style_chart(chart), use_container_width=True)


def split_bar(probabilities):
    """One stacked bar, Low to Critical, instead of four separate bars."""
    df = pd.DataFrame({"priority": PRIORITY_ORDER, "probability": [probabilities[p] for p in PRIORITY_ORDER],
                       "order": range(len(PRIORITY_ORDER))})
    chart = alt.Chart(df).mark_bar(height=12).encode(
        x=alt.X("probability:Q", stack="zero", axis=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("priority:N", legend=None,
                        scale=alt.Scale(domain=PRIORITY_ORDER, range=[theme.PRIORITY[p] for p in PRIORITY_ORDER])),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("priority", title="Priority"), alt.Tooltip("probability:Q", title="Probability",
                                                                        format=".1%")],
    ).properties(height=16)
    st.altair_chart(theme.style_chart(chart), use_container_width=True)
    theme.html('<div class="tt-split">' + "".join(
        f'<span><b style="color:{theme.PRIORITY[p]}">{p}</b> {probabilities[p]:.0%}</span>'
        for p in PRIORITY_ORDER) + "</div>")


def importance_chart(items):
    df = pd.DataFrame(items)
    chart = alt.Chart(df).mark_bar(height=12, color=theme.ACCENT).encode(
        x=alt.X("mean_abs_shap:Q", title="Mean |SHAP|"),
        y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("feature", title="Feature"), alt.Tooltip("mean_abs_shap:Q", title="Mean |SHAP|",
                                                                      format=".3f")],
    ).properties(height=alt.Step(24))
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
        color=alt.Color("share:Q", legend=None, scale=alt.Scale(range=[theme.SUBTLE, theme.ACCENT], domain=[0, 1])),
        tooltip=[alt.Tooltip("true", title="Actual"), alt.Tooltip("pred", title="Predicted"),
                 alt.Tooltip("count:Q", title="Tickets"), alt.Tooltip("share:Q", title="Share of row", format=".0%")],
    )
    text = base.mark_text(fontSize=13, fontWeight=600).encode(
        text="count:Q",
        color=alt.condition(alt.datum.share > 0.5, alt.value("#FFFFFF"), alt.value(theme.INK)),
    )
    st.altair_chart(theme.style_chart((heat + text).properties(height=250)), use_container_width=True)


def key_terms(terms):
    if not terms:
        return
    words = "".join(
        f'<span style="color:{theme.UP if t["weight"] > 0 else theme.DOWN}">{escape(t["term"])} '
        f'{t["weight"]:+.2f}</span>' for t in terms)
    theme.html(f'<div class="tt-label" style="margin-top:.2rem">Words behind the text signal</div>'
               f'<div class="tt-terms">{words}</div>')


# --- Header ------------------------------------------------------------------------

theme.html('<div class="tt-eyebrow">Support desk · triage</div><div class="tt-title">Ticket Triage</div>'
           '<div class="tt-lede">Paste a ticket to get its priority, an expected time to resolution, '
           'and the reasons behind both.</div>')
try:
    with st.spinner("Loading models..."):
        health = get("/health")
except ApiError as e:
    st.error(f"Can't reach the prediction API at `{API_URL}`. {e}", icon=":material/cloud_off:")
    st.stop()

models = health["models"]
theme.html(f'<div class="tt-meta"><span class="ok">●</span> {"API connected" if API_URL else "Models ready"}'
           f' · priority: {label(models["priority"])} · resolution: {label(models["resolution"])}'
           f' · trained {health["trained_at"][:10]}</div>')

tab_triage, tab_model = st.tabs(["Triage", "How the model performs"])

# --- Triage ------------------------------------------------------------------------

with tab_triage:
    pick, _ = st.columns([2, 3])
    sample = pick.selectbox("Sample ticket", list(EXAMPLES),
                            help="Fills in the form. Edit anything before running triage.")
    ex = EXAMPLES[sample]

    with st.form("ticket", border=False):
        left, right = st.columns([3, 2], gap="large")
        with left:
            subject = st.text_input("Subject", ex["subject"], max_chars=200)
            description = st.text_area("Description", ex["description"], height=178, max_chars=2000)
        with right:
            # Row by row, so related fields stay together when columns stack on phones.
            c1, c2 = st.columns(2)
            channel = c1.selectbox("Channel", CHANNELS, index=CHANNELS.index(ex["channel"]), format_func=label)
            tier = c2.selectbox("Customer tier", TIERS, index=TIERS.index(ex["customer_tier"]), format_func=label)
            c1, c2 = st.columns(2)
            product = c1.selectbox("Product", PRODUCTS, index=PRODUCTS.index(ex["product"]))
            category = c2.selectbox("Category", CATEGORIES, index=CATEGORIES.index(ex["category"]),
                                    format_func=label, help="Picked by the customer, so it can be wrong.")
            c1, c2 = st.columns(2)
            date = c1.date_input("Opened", dt.date.today())
            time = c2.time_input("Time", dt.time(10, 30))
            c1, c2 = st.columns(2)
            prior = c1.number_input("Tickets, last 30 days", 0, 100, ex["prior_tickets_30d"])
            attachments = c2.number_input("Attachments", 0, 50, ex["attachments"])
        submitted = st.form_submit_button("Run triage", type="primary")

    if not submitted:
        theme.html('<div class="tt-empty">Load a sample or write a ticket, then run triage.</div>')
    else:
        ticket = dict(subject=subject, description=description, channel=channel, customer_tier=tier,
                      product=product, category=category, prior_tickets_30d=int(prior),
                      attachments=int(attachments), created_at=dt.datetime.combine(date, time).isoformat())
        res = None
        if not (subject.strip() or description.strip()):
            st.warning("Add a subject or description first.", icon=":material/edit_note:")
        else:
            try:
                with st.spinner("Scoring ticket..."):
                    status, body = call("POST", "/predict", params={"explain": "true"}, json=ticket)
                if status == 200:
                    res = body
                else:
                    details = body.get("details") or [body.get("error", f"HTTP {status}")]
                    st.error("Couldn't score this ticket:\n" + "\n".join(f"- {d}" for d in details),
                             icon=":material/error:")
            except ApiError as e:
                st.error(f"The prediction request failed. {e}", icon=":material/cloud_off:")

        if res:
            priority, breach = res["priority"], res["sla_breach_risk"]
            theme.html(theme.verdict(
                priority, f"{res['confidence']:.0%} confident",
                theme.fact("Expected resolution", format_hours(res["resolution_hours"]),
                           f"{res['resolution_hours']:.1f} hours")
                + theme.fact("SLA target", f"{res['sla_target_hours']} h",
                             "at risk" if breach else "on track", "warn" if breach else "ok")))
            theme.label("Priority split")
            split_bar(res["priority_probabilities"])

            exp = res["explanation"]
            left, right = st.columns(2, gap="large")
            with left:
                theme.heading(f"Why {priority}", "What raised or lowered the odds of this priority.")
                contributions_chart(exp["priority"], "raises", "lowers")
                key_terms(exp["priority_key_terms"])
            with right:
                theme.heading(f"Why {format_hours(res['resolution_hours'])}",
                              "What pushed the expected resolution time up or down.")
                contributions_chart(exp["resolution_time"], "slower", "faster")
                key_terms(exp["resolution_key_terms"])
            st.caption(f"Scored in {res['latency_ms']:.0f} ms. SHAP values are in log-odds for priority "
                       "and log-hours for resolution time.")

# --- Model performance --------------------------------------------------------------

with tab_model:
    try:
        m = get("/model")
    except ApiError:
        theme.html('<div class="tt-empty">Evaluation results aren\'t available from the API.</div>')
        st.stop()
    p, rt = m["priority"], m["resolution_time"]
    pf, rf = p["final_test"], rt["final_test"]
    base_f1 = p["comparison"].get("baseline_majority", {}).get("test", {}).get("macro_f1")
    base_mae = rt["comparison"].get("baseline_median", {}).get("test", {}).get("mae_hours")

    theme.heading("Held-out test set", "Tickets the models never saw during training or model selection.",
                  first=True)
    theme.html(theme.facts(
        theme.fact("Priority macro-F1", f"{pf['macro_f1']:.3f}", f"baseline {base_f1:.3f}" if base_f1 else "")
        + theme.fact("Priority accuracy", f"{pf['accuracy']:.1%}")
        + theme.fact("Resolution MAE", f"{rf['mae_hours']:.1f} h", f"baseline {base_mae:.1f} h" if base_mae else "")
        + theme.fact("Median error", f"{rf['median_ae_hours']:.1f} h", f"R² (log) {rf['r2_log']:.2f}")))

    def comparison_table(block, columns, fmt):
        rows = []
        for name, v in block["comparison"].items():
            row = {"Model": label(name) + ("  ✓" if name == block["selected_model"] else "")}
            row.update({title: v["val"][key] for key, title in columns.items()})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format(fmt), width="stretch")

    theme.heading("Models compared", "Validation scores. ✓ marks the model in use, refit on train + validation.")
    left, right = st.columns(2, gap="large")
    with left:
        theme.label("Priority")
        comparison_table(p, {"macro_f1": "Macro-F1", "accuracy": "Accuracy"},
                         {"Macro-F1": "{:.3f}", "Accuracy": "{:.1%}"})
    with right:
        theme.label("Resolution time")
        comparison_table(rt, {"mae_hours": "MAE (h)", "median_ae_hours": "Median AE (h)"},
                         {"MAE (h)": "{:.2f}", "Median AE (h)": "{:.2f}"})

    left, right = st.columns([5, 6], gap="large")
    with left:
        theme.heading("Where it goes wrong", "Mistakes sit next to the diagonal; Low and Critical are "
                                             "almost never confused.")
        confusion_chart(p["confusion_matrix"]["labels"], p["confusion_matrix"]["matrix"])
    with right:
        theme.heading("What drives predictions", "Average impact per feature across held-out tickets.")
        target = st.radio("Target", ["Priority", "Resolution time"], horizontal=True, label_visibility="collapsed")
        importance_chart((p if target == "Priority" else rt)["shap_importance"][:10])

theme.html(f'<div class="tt-foot">Trained on synthetic tickets with a known structure, since real support data '
           f'is private. <a href="{REPO_URL}">Source on GitHub</a></div>')
