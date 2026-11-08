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
import random
import sys

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
    """(status, body) from the remote API, or from the in-process app when API_URL is unset."""
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


def label(name):
    return MODEL_NAMES.get(name, name.replace("_", " ").capitalize())


def format_hours(hours):
    if hours < 1:
        return f"{hours * 60:.0f} min"
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} days"


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
    heat = base.mark_rect(stroke=theme.PAGE, strokeWidth=2).encode(
        color=alt.Color("share:Q", legend=None, scale=alt.Scale(range=[theme.READONLY, theme.ACCENT], domain=[0, 1])),
        tooltip=[alt.Tooltip("true", title="Actual"), alt.Tooltip("pred", title="Predicted"),
                 alt.Tooltip("count:Q", title="Tickets"), alt.Tooltip("share:Q", title="Share of row", format=".0%")],
    )
    text = base.mark_text(fontSize=13, fontWeight=600).encode(
        text="count:Q",
        color=alt.condition(alt.datum.share > 0.5, alt.value("#FFFFFF"), alt.value(theme.INK)),
    )
    st.altair_chart(theme.style_chart((heat + text).properties(height=240)), use_container_width=True)


def comparison_table(block, columns, fmt):
    rows = []
    for name, v in block["comparison"].items():
        row = {"Model": label(name) + ("  ✓" if name == block["selected_model"] else "")}
        row.update({title: v["val"][key] for key, title in columns.items()})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("Model").style.format(fmt), width="stretch")


theme.html(theme.hero())

try:
    health = get("/health")
except ApiError as e:
    st.error(f"Can't reach the prediction API at `{API_URL}`. {e}")
    st.stop()

try:
    evaluation = get("/model")
except ApiError:
    evaluation = None

FORM_FIELDS = ("subject", "description", "customer_tier", "channel", "product", "category",
               "prior_tickets_30d", "attachments")


def load_sample():
    # Fields keep stable keys and get new values in place. Without keys, a new sample changes every
    # widget's default, Streamlit rebuilds the whole form, and the page jumps.
    for field in FORM_FIELDS:
        st.session_state[f"f_{field}"] = EXAMPLES[st.session_state.sample][field]
    st.session_state.result_fresh = False


def pick_sample(name):
    st.session_state.sample = name
    load_sample()


if "f_subject" not in st.session_state:
    st.session_state.sample = list(EXAMPLES)[0]
    load_sample()

if "incident" not in st.session_state:
    st.session_state.incident = f"INC{random.randint(1, 9999999):07d}"

theme.html('<div id="open-a-ticket" class="tt-title">Open a ticket.<br><em>See where it lands.</em></div>'
           '<div class="tt-sub">Models are trained on synthetic tickets; real-data results are below.</div>')

tab_incident, tab_eval = st.tabs(["Incident", "Model evaluation"])

with tab_incident:
    # Buttons rather than a dropdown: opening a dropdown nudged the page on some browsers.
    with st.container(horizontal=True, vertical_alignment="center", gap="small", key="samples"):
        theme.html('<div class="tt-crumb">Try a sample:</div>')
        for i, name in enumerate(EXAMPLES):
            st.button(name, key=f"sample_{i}", on_click=pick_sample, args=(name,),
                      type="primary" if name == st.session_state.sample else "secondary")

    with st.form("ticket", border=False):
        main, stub = st.columns([3, 2], gap="large")
        with main:
            head_slot = st.empty()
            subject = st.text_input("Short description", max_chars=200, key="f_subject")
            description = st.text_area("Description", height=76, max_chars=2000, key="f_description")
            theme.section("Context")
            c1, c2, c3 = st.columns(3)
            tier = c1.selectbox("Customer tier", TIERS, format_func=label, key="f_customer_tier")
            channel = c2.selectbox("Channel", CHANNELS, format_func=label, key="f_channel")
            product = c3.selectbox("Product", PRODUCTS, key="f_product")
            c1, c2, c3 = st.columns(3)
            category = c1.selectbox("Category", CATEGORIES, format_func=label, key="f_category")
            date = c2.date_input("Opened", dt.date.today())
            time = c3.time_input("Time", dt.time(10, 30))
            c1, c2, _ = st.columns(3)
            prior = c1.number_input("Tickets, last 30 days", 0, 100, key="f_prior_tickets_30d")
            attachments = c2.number_input("Attachments", 0, 50, key="f_attachments")
            submitted = st.form_submit_button("Triage", type="primary", width="stretch", key="triage")
        with stub:
            theme.html('<div class="tt-stub"></div>')
            theme.section("Assessment")
            assessment_slot = st.empty()

    if submitted:
        ticket = dict(subject=subject, description=description, channel=channel, customer_tier=tier,
                      product=product, category=category, prior_tickets_30d=int(prior),
                      attachments=int(attachments), created_at=dt.datetime.combine(date, time).isoformat())
        st.session_state.result_fresh = False
        if not (subject.strip() or description.strip()):
            st.warning("Add a short description or description first.")
        else:
            try:
                status, body = call("POST", "/predict", params={"explain": "true"}, json=ticket)
                if status == 200:
                    st.session_state.update(result=body, result_fresh=True,
                                            result_time=dt.datetime.now().strftime("%H:%M"))
                else:
                    details = body.get("details") or [body.get("error", f"HTTP {status}")]
                    st.error("Couldn't triage this ticket:\n" + "\n".join(f"- {d}" for d in details))
            except ApiError as e:
                st.error(f"The prediction request failed. {e}")

    # The last result stays on screen, faded, until the next triage. Removing it would shrink the page
    # and shift everything below; the faded copy also skips the entrance animations.
    res = st.session_state.get("result")
    fresh = st.session_state.get("result_fresh", False)
    wrap = "tt-landed" if fresh else "tt-quiet tt-stale"
    stale_note = "" if fresh else '<div class="tt-stalenote">Last result. Press Triage to score this ticket.</div>'
    head_slot.markdown(f'<div class="tt-head"><span class="k">Incident</span><span class="n">'
                       f'{st.session_state.incident}</span><span class="tt-state{" done" if fresh else ""}">'
                       f'{"Triaged" if fresh else "New"}</span></div>', unsafe_allow_html=True)

    if res:
        priority, breach = res["priority"], res["sla_breach_risk"]
        hours, target = res["resolution_hours"], res["sla_target_hours"]
        assessment_slot.markdown(
            stale_note + f'<div class="{wrap}">' + theme.verdict(priority, f"{res['confidence']:.0%} confidence")
            + theme.split(res["priority_probabilities"]) + '<div style="height:.75rem"></div>'
            + theme.field("Expected resolution", format_hours(hours))
            + theme.field("SLA", f"{target} h target, {'at risk' if breach else 'on track'}",
                          theme.sla_meter(hours, target, breach))
            + theme.slam(priority) + "</div>",
            unsafe_allow_html=True)
    else:
        assessment_slot.markdown(
            '<div class="tt-await">Awaiting triage</div>'
            + theme.field("Expected resolution") + theme.field("SLA"), unsafe_allow_html=True)

    if res:
        exp = res["explanation"]
        now = st.session_state.result_time
        # A low score for some other priority is real signal, but reads oddly as a reason.
        other_levels = tuple(f"text signal: P({p})" for p in theme.PRIORITY if p != priority)
        drivers = [feature_label(c["feature"], c["value"]) for c in exp["priority"]
                   if c["shap"] > 0 and c["feature"] not in other_levels][:3]
        terms = [t["term"] for t in exp["priority_key_terms"] if t["weight"] > 0][:3]

        def factors(items):
            return [(feature_label(c["feature"], c["value"]), c["shap"]) for c in items]

        theme.html(
            '<div class="tt-trailhead">The paper trail</div>'
            '<div class="tt-trailsub">What the model noted, and what pushed each result up or down.</div>'
            + stale_note + f'<div class="{wrap}">'
            + theme.trail([
                theme.note("Triage model", now, f"Priority set to {theme.PRIORITY_CODE[priority]} "
                           f"({res['confidence']:.0%} confidence). Expected resolution {format_hours(hours)} "
                           f"against a {target} h SLA."),
                theme.note("Triage model", now, "Main factors: " + "; ".join(drivers)
                           + (f". Key terms: {', '.join(terms)}." if terms else ".")),
            ])
            + '<div class="tt-evis">'
            + theme.evidence("Priority", theme.PRIORITY_CODE[priority], priority.upper(), theme.STAMP[priority],
                             factors(exp["priority"]), exp["priority_key_terms"], "raises", "lowers")
            + theme.evidence("Expected resolution", format_hours(hours), "AT RISK" if breach else "ON TRACK",
                             theme.WARN if breach else theme.OK, factors(exp["resolution_time"]),
                             exp["resolution_key_terms"], "slower", "faster")
            + "</div>"
            + '<div class="tt-muted" style="margin-top:.7rem;text-align:center">SHAP contributions: log-odds for priority, '
            f'log-hours for resolution time. Scored in {res["latency_ms"]:.0f} ms.</div></div>')
    else:
        theme.html('<div class="tt-empty" style="margin-top:1rem">No activity yet. Run triage to add a work note.</div>')

with tab_eval:
    if evaluation is None:
        theme.html('<div class="tt-muted">Evaluation results aren\'t available from the API.</div>')
    else:
        p, rt = evaluation["priority"], evaluation["resolution_time"]
        pf, rf = p["final_test"], rt["final_test"]
        base_f1 = p["comparison"].get("baseline_majority", {}).get("test", {}).get("macro_f1")
        base_mae = rt["comparison"].get("baseline_median", {}).get("test", {}).get("mae_hours")

        theme.section("Synthetic test set")
        theme.html('<div class="tt-muted">Held-out tickets never seen during training or model selection. '
                   f'Models: {label(health["models"]["priority"])}, trained {health["trained_at"][:10]}.</div>'
                   + theme.stats([
                       ("Priority macro-F1", f"{pf['macro_f1']:.3f}", f"baseline {base_f1:.3f}" if base_f1 else ""),
                       ("Priority accuracy", f"{pf['accuracy']:.1%}", ""),
                       ("Resolution MAE", f"{rf['mae_hours']:.1f} h", f"baseline {base_mae:.1f} h" if base_mae else ""),
                       ("Median error", f"{rf['median_ae_hours']:.1f} h", f"R² (log) {rf['r2_log']:.2f}"),
                   ]))

        real = evaluation.get("real_data")
        if real:
            theme.section("Real-data check: Apache Jira")
            theme.html(f'<div class="tt-muted">{real["rows"]:,} real Apache Jira bug reports, tested on the newest '
                       'tickets. These are open-source bugs, not support tickets, and resolution times here are '
                       'no better than guessing the median.</div>'
                       + theme.stats([
                           ("Priority macro-F1", f"{real['priority']['macro_f1']:.3f}",
                            f"baseline {real['priority_baseline']['macro_f1']:.3f}"),
                           ("Priority accuracy", f"{real['priority']['accuracy']:.1%}",
                            f"baseline {real['priority_baseline']['accuracy']:.1%}"),
                           ("Resolution MAE", f"{real['resolution']['mae_hours']:,.0f} h",
                            f"baseline {real['resolution_baseline']['mae_hours']:,.0f} h"),
                           ("Median error", f"{real['resolution']['median_ae_hours']:.0f} h",
                            f"baseline {real['resolution_baseline']['median_ae_hours']:.0f} h"),
                       ]))

        theme.section("Models compared")
        theme.html('<div class="tt-muted">Validation scores. ✓ marks the model in use.</div>')
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
            theme.section("Confusion matrix")
            confusion_chart(p["confusion_matrix"]["labels"], p["confusion_matrix"]["matrix"])
        with right:
            theme.section("Feature importance")
            target = st.radio("Target", ["Priority", "Resolution time"], horizontal=True,
                              label_visibility="collapsed")
            importance_chart((p if target == "Priority" else rt)["shap_importance"][:10])

real = evaluation.get("real_data") if evaluation else None
jira = ""
if real:
    jira = (f'<p>On {real["rows"]:,} real Apache Jira bug reports, priority macro-F1 falls to '
            f'{real["priority"]["macro_f1"]:.2f} (baseline {real["priority_baseline"]["macro_f1"]:.2f}), and resolution '
            'times are no better than guessing the median. Those are open-source bugs, not support tickets.</p>')
theme.html(f"""<div class="tt-band"><h2>The honest numbers.<br><em>Not every ticket is this easy.</em></h2>
<p>This demo's tickets are synthetic, so its scores aren't evidence of how it would do in a real queue.</p>{jira}
<p>SHAP shows what influenced a prediction, not what caused a delay. <a href="{REPO_URL}">Source on GitHub</a></p></div>""")
