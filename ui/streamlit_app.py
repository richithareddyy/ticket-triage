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

st.set_page_config(page_title="Ticket Triage", page_icon=":material/confirmation_number:", layout="wide")
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


MODEL_NAMES = {"xgboost": "XGBoost", "random_forest": "Random Forest", "logistic_regression": "Logistic Regression",
               "ridge": "Ridge", "baseline_majority": "Baseline (majority)", "baseline_median": "Baseline (median)"}


def label(name):
    return MODEL_NAMES.get(name, name.replace("_", " ").capitalize())


def format_hours(hours):
    if hours < 1:
        return f"{hours * 60:.0f} min"
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} days"


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


# --- Charts ------------------------------------------------------------------------

def contributions_chart(items, positive_label, negative_label):
    """Horizontal SHAP bars, largest impact first; warm pushes up, primary pulls down."""
    df = pd.DataFrame(items)
    df["label"] = df.apply(lambda r: r["feature"] + (f" ({r['value']})" if r["value"] != "" else ""), axis=1)
    df["direction"] = df["shap"].map(lambda v: positive_label if v > 0 else negative_label)
    chart = alt.Chart(df).mark_bar(cornerRadius=3, height=18).encode(
        x=alt.X("shap:Q", title="SHAP contribution"),
        y=alt.Y("label:N", sort=df.sort_values("shap", key=abs, ascending=False)["label"].tolist(), title=None),
        color=alt.Color("direction:N", title=None, legend=alt.Legend(orient="bottom", labelLimit=250),
                        scale=alt.Scale(domain=[positive_label, negative_label], range=[theme.UP, theme.DOWN])),
        tooltip=[alt.Tooltip("feature", title="Feature"), alt.Tooltip("value", title="Value"),
                 alt.Tooltip("shap:Q", title="SHAP", format="+.3f")],
    ).properties(height=alt.Step(30))
    st.altair_chart(theme.style_chart(chart), use_container_width=True)


def probability_chart(probabilities):
    df = pd.DataFrame({"priority": list(probabilities), "probability": list(probabilities.values())})
    bars = alt.Chart(df).encode(
        y=alt.Y("priority:N", sort=PRIORITY_ORDER[::-1], title=None),
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5),
                title=None),
    )
    chart = (bars.mark_bar(cornerRadius=3, height=16).encode(
        color=alt.Color("priority:N", legend=None,
                        scale=alt.Scale(domain=PRIORITY_ORDER, range=[theme.PRIORITY[p]["bar"] for p in PRIORITY_ORDER])),
        tooltip=[alt.Tooltip("priority", title="Priority"), alt.Tooltip("probability:Q", title="Probability",
                                                                        format=".1%")])
        + bars.mark_text(align="left", dx=6, color=theme.MUTED, fontSize=12).encode(
            text=alt.Text("probability:Q", format=".0%"))
    ).properties(height=alt.Step(30))
    st.altair_chart(theme.style_chart(chart), use_container_width=True)


def importance_chart(items):
    df = pd.DataFrame(items)
    chart = alt.Chart(df).mark_bar(cornerRadius=3, height=14, color=theme.PRIMARY).encode(
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
    heat = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("share:Q", legend=None, scale=alt.Scale(range=["#F3F5FA", theme.PRIMARY], domain=[0, 1])),
        tooltip=[alt.Tooltip("true", title="Actual"), alt.Tooltip("pred", title="Predicted"),
                 alt.Tooltip("count:Q", title="Tickets"), alt.Tooltip("share:Q", title="Share of row", format=".0%")],
    )
    text = base.mark_text(fontSize=13, fontWeight=600).encode(
        text="count:Q",
        color=alt.condition(alt.datum.share > 0.5, alt.value("#FFFFFF"), alt.value(theme.TEXT)),
    )
    st.altair_chart(theme.style_chart((heat + text).properties(height=260)), use_container_width=True)


def key_terms(terms):
    if not terms:
        return
    chips = "".join(
        f'<span class="tt-term" style="color:{theme.UP if t["weight"] > 0 else theme.DOWN}">'
        f'{t["term"]} {t["weight"]:+.2f}</span>' for t in terms)
    html(f'<p class="tt-section" style="margin-top:.4rem">Words driving the text signal</p>'
         f'<div class="tt-terms">{chips}</div>')


# --- Header + sidebar --------------------------------------------------------------

try:
    with st.spinner("Loading models..."):
        health = get("/health")
except ApiError as e:
    html('<div class="tt-header"><h1>Support Ticket Triage</h1></div>')
    st.error(f"**Can't reach the prediction API** at `{API_URL}`.\n\n{e}", icon=":material/cloud_off:")
    st.stop()

models = health["models"]
trained = health["trained_at"][:10]
html(
    '<div class="tt-header"><h1>Support Ticket Triage</h1>'
    '<p>Predict a ticket\'s priority and time to resolution, and see exactly why.</p>'
    '<div class="tt-chips">'
    + theme.status_badge("Models ready" if not API_URL else "API connected", "ok")
    + theme.status_badge(f"Priority: {label(models['priority'])}", "neutral", dot=False)
    + theme.status_badge(f"Resolution: {label(models['resolution'])}", "neutral", dot=False)
    + theme.status_badge(f"Trained {trained}", "neutral", dot=False)
    + "</div></div>")

with st.sidebar:
    st.markdown("### :material/confirmation_number: Ticket Triage")
    st.caption("An ML pipeline that classifies support-ticket priority and predicts resolution time, "
               "with SHAP explanations for every prediction.")
    st.markdown("**How to use**")
    st.markdown("1. Pick an example or write a ticket\n2. Adjust the customer details\n"
                "3. Press **Predict priority**")
    st.markdown("**Test-set performance**")
    metrics = health.get("metrics", {})
    if metrics:
        st.markdown(f"- Priority macro-F1: **{metrics['priority']['macro_f1']:.3f}**\n"
                    f"- Resolution MAE: **{metrics['resolution_time']['mae_hours']:.1f} h**")
    st.divider()
    st.markdown(f":material/code: [Source code on GitHub]({REPO_URL})")
    st.caption("Trained on a synthetic ticket dataset with a known structure; real ticket data is proprietary.")

tab_predict, tab_model = st.tabs([":material/bolt: Triage a ticket", ":material/insights: Model performance"])

# --- Triage tab --------------------------------------------------------------------

with tab_predict:
    choice = st.pills("Start from an example", list(EXAMPLES), default=list(EXAMPLES)[0],
                      help="Fills the form with a sample ticket. You can edit any field before predicting.")
    ex = EXAMPLES[choice or list(EXAMPLES)[0]]

    with st.container(key="card-form"):
        with st.form("ticket", border=False):
            theme.section_label("Ticket")
            subject = st.text_input("Subject", ex["subject"], max_chars=200, placeholder="Short summary")
            description = st.text_area("Description", ex["description"], height=110, max_chars=2000,
                                       placeholder="What is the customer reporting?")
            theme.section_label("Customer & context")
            c1, c2, c3, c4 = st.columns(4)
            channel = c1.selectbox("Channel", CHANNELS, index=CHANNELS.index(ex["channel"]), format_func=label)
            tier = c2.selectbox("Customer tier", TIERS, index=TIERS.index(ex["customer_tier"]), format_func=label)
            product = c3.selectbox("Product", PRODUCTS, index=PRODUCTS.index(ex["product"]))
            category = c4.selectbox("Category", CATEGORIES, index=CATEGORIES.index(ex["category"]),
                                    format_func=label, help="As selected by the customer, so it can be wrong.")
            theme.section_label("Timing & history")
            c5, c6, c7, c8 = st.columns(4)
            date = c5.date_input("Created date", dt.date.today())
            time = c6.time_input("Created time", dt.time(10, 30))
            prior = c7.number_input("Tickets in last 30 days", 0, 100, ex["prior_tickets_30d"])
            attachments = c8.number_input("Attachments", 0, 50, ex["attachments"])
            submitted = st.form_submit_button("Predict priority", type="primary", icon=":material/bolt:")

    st.write("")
    if not submitted:
        theme.empty_state("No prediction yet",
                          "Pick an example or write your own ticket above, then press Predict priority.")
    else:
        ticket = dict(subject=subject, description=description, channel=channel, customer_tier=tier,
                      product=product, category=category, prior_tickets_30d=int(prior),
                      attachments=int(attachments), created_at=dt.datetime.combine(date, time).isoformat())
        res = None
        if not (subject.strip() or description.strip()):
            st.warning("Add a subject or description so there's something to analyze.", icon=":material/edit:")
        else:
            try:
                with st.spinner("Scoring ticket..."):
                    status, body = call("POST", "/predict", params={"explain": "true"}, json=ticket)
                if status == 200:
                    res = body
                else:
                    details = body.get("details") or [body.get("error", f"HTTP {status}")]
                    st.error("**Please fix the following:**\n" + "\n".join(f"- {d}" for d in details),
                             icon=":material/error:")
            except ApiError as e:
                st.error(f"**The prediction request failed.** {e}", icon=":material/cloud_off:")

        if res:
            priority = res["priority"]
            breach = res["sla_breach_risk"]
            s1, s2, s3, s4 = st.columns(4)
            s1.markdown(theme.stat_card("Priority", theme.priority_badge(priority, large=True),
                                        f"{res['confidence']:.0%} confidence"), unsafe_allow_html=True)
            s2.markdown(theme.stat_card("Predicted resolution", format_hours(res["resolution_hours"]),
                                        f"{res['resolution_hours']:.1f} hours"), unsafe_allow_html=True)
            s3.markdown(theme.stat_card("SLA target", f"{res['sla_target_hours']} h",
                                        theme.status_badge("Breach risk" if breach else "Within SLA",
                                                           "warn" if breach else "ok")),
                        unsafe_allow_html=True)
            s4.markdown(theme.stat_card("Model latency", f"{res['latency_ms']:.0f} ms", "Including SHAP"),
                        unsafe_allow_html=True)
            st.write("")

            with st.container(key="card-probs"):
                theme.card_header("Priority probabilities", "How confident the classifier is in each level.")
                probability_chart(res["priority_probabilities"])

            exp = res["explanation"]
            left, right = st.columns(2, gap="medium")
            with left:
                with st.container(key="card-why-priority"):
                    theme.card_header(f"Why {priority}?",
                                      f"SHAP contributions (log-odds) that raise or lower the chance of {priority}.")
                    contributions_chart(exp["priority"], "raises", "lowers")
                    key_terms(exp["priority_key_terms"])
            with right:
                with st.container(key="card-why-time"):
                    theme.card_header("Why this resolution time?",
                                      "SHAP contributions in log-hours. Positive means slower.")
                    contributions_chart(exp["resolution_time"], "slower", "faster")
                    key_terms(exp["resolution_key_terms"])
            st.toast(f"Scored as {priority}", icon=":material/check_circle:")

# --- Model performance tab ---------------------------------------------------------

with tab_model:
    try:
        m = get("/model")
    except ApiError:
        theme.empty_state("Model metrics unavailable", "The API did not return evaluation results.")
        st.stop()
    p, rt = m["priority"], m["resolution_time"]
    pf, rf = p["final_test"], rt["final_test"]
    baseline_f1 = p["comparison"].get("baseline_majority", {}).get("test", {}).get("macro_f1")
    baseline_mae = rt["comparison"].get("baseline_median", {}).get("test", {}).get("mae_hours")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(theme.stat_card("Priority macro-F1", f"{pf['macro_f1']:.3f}",
                                f"Baseline {baseline_f1:.3f}" if baseline_f1 else ""), unsafe_allow_html=True)
    c2.markdown(theme.stat_card("Priority accuracy", f"{pf['accuracy']:.1%}", "Held-out test set"),
                unsafe_allow_html=True)
    c3.markdown(theme.stat_card("Resolution MAE", f"{rf['mae_hours']:.1f} h",
                                f"Baseline {baseline_mae:.1f} h" if baseline_mae else ""), unsafe_allow_html=True)
    c4.markdown(theme.stat_card("Resolution median error", f"{rf['median_ae_hours']:.1f} h",
                                f"R² (log) {rf['r2_log']:.2f}"), unsafe_allow_html=True)
    st.write("")

    def comparison_table(block, columns, fmt):
        rows = []
        for name, v in block["comparison"].items():
            row = {"Model": label(name)}
            row.update({title: v["val"][key] for key, title in columns.items()})
            row["Selected"] = "✓" if name == block["selected_model"] else ""
            rows.append(row)
        df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(df.style.format(fmt), width="stretch")

    with st.container(key="card-compare"):
        theme.card_header("Model comparison", "Validation-set scores. The selected model is refit on "
                                              "train + validation and evaluated once on the test set.")
        left, right = st.columns(2, gap="medium")
        with left:
            theme.section_label("Priority classification")
            comparison_table(p, {"macro_f1": "Macro-F1", "accuracy": "Accuracy"},
                             {"Macro-F1": "{:.3f}", "Accuracy": "{:.1%}"})
        with right:
            theme.section_label("Resolution time")
            comparison_table(rt, {"mae_hours": "MAE (h)", "median_ae_hours": "Median AE (h)"},
                             {"MAE (h)": "{:.2f}", "Median AE (h)": "{:.2f}"})
    st.write("")

    left, right = st.columns([5, 7], gap="medium")
    with left:
        with st.container(key="card-confusion"):
            theme.card_header("Confusion matrix", "Test set. Errors sit next to the diagonal.")
            confusion_chart(p["confusion_matrix"]["labels"], p["confusion_matrix"]["matrix"])
    with right:
        with st.container(key="card-importance"):
            theme.card_header("Global feature importance", "Mean |SHAP| across held-out tickets.")
            imp_tabs = st.tabs(["Priority", "Resolution time"])
            with imp_tabs[0]:
                importance_chart(p["shap_importance"][:10])
            with imp_tabs[1]:
                importance_chart(rt["shap_importance"][:10])
