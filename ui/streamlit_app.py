"""Streamlit front end for the triage API.

    API_URL=http://localhost:5050 streamlit run ui/streamlit_app.py
"""
import datetime as dt
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:5050").rstrip("/")
PRIORITY_COLORS = {"Critical": "#c62828", "High": "#ef6c00", "Medium": "#f9a825", "Low": "#2e7d32"}
PRODUCTS = ["Dashboard", "Mobile App", "API", "Billing Portal", "Reports", "Integrations Hub"]
CATEGORIES = ["outage", "security", "data_loss", "bug", "performance", "login_access", "billing",
              "integration", "how_to", "feature_request"]
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

st.set_page_config(page_title="Ticket Triage", layout="wide")


@st.cache_data(ttl=60)
def get(path):
    r = requests.get(f"{API_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def contributions_chart(items, positive_label, negative_label):
    """Horizontal SHAP bars, largest impact first; red pushes up, blue pushes down."""
    df = pd.DataFrame(items)
    df["label"] = df.apply(lambda r: r["feature"] + (f" ({r['value']})" if r["value"] != "" else ""), axis=1)
    df["direction"] = df["shap"].map(lambda v: positive_label if v > 0 else negative_label)
    chart = alt.Chart(df).mark_bar(cornerRadius=2).encode(
        x=alt.X("shap:Q", title="SHAP value"),
        y=alt.Y("label:N", sort=df.sort_values("shap", key=abs, ascending=False)["label"].tolist(),
                title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("direction:N", title=None, legend=alt.Legend(orient="bottom", labelLimit=250),
                        scale=alt.Scale(domain=[positive_label, negative_label], range=["#d6453d", "#3b6fd8"])),
        tooltip=["feature", "value", alt.Tooltip("shap:Q", format="+.3f")],
    ).properties(height=alt.Step(30))
    st.altair_chart(chart, use_container_width=True)


def importance_chart(items):
    df = pd.DataFrame(items)
    chart = alt.Chart(df).mark_bar(cornerRadius=2, color="#3b6fd8").encode(
        x=alt.X("mean_abs_shap:Q", title="mean |SHAP|"),
        y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
        tooltip=["feature", alt.Tooltip("mean_abs_shap:Q", format=".3f")],
    ).properties(height=alt.Step(24))
    st.altair_chart(chart, use_container_width=True)


st.title("Support Ticket Triage")
st.caption("Predicts ticket priority and time-to-resolution, and explains each prediction with SHAP.")

try:
    health = get("/health")
    st.sidebar.success(f"API connected\n\npriority: **{health['models']['priority']}**  \n"
                       f"resolution: **{health['models']['resolution']}**")
    st.sidebar.caption(f"Trained {health['trained_at'][:16].replace('T', ' ')} UTC")
except requests.RequestException as e:
    st.sidebar.error(f"API unreachable at {API_URL}")
    st.error(f"Could not reach the prediction API: {e}")
    st.stop()

tab_predict, tab_model = st.tabs(["Triage a ticket", "Model performance"])

with tab_predict:
    example = st.selectbox("Start from an example", list(EXAMPLES))
    ex = EXAMPLES[example]
    with st.form("ticket"):
        subject = st.text_input("Subject", ex["subject"])
        description = st.text_area("Description", ex["description"], height=110)
        c1, c2, c3, c4 = st.columns(4)
        channel = c1.selectbox("Channel", ["email", "web", "chat", "phone"],
                               index=["email", "web", "chat", "phone"].index(ex["channel"]))
        tier = c2.selectbox("Customer tier", ["free", "pro", "enterprise"],
                            index=["free", "pro", "enterprise"].index(ex["customer_tier"]))
        product = c3.selectbox("Product", PRODUCTS, index=PRODUCTS.index(ex["product"]))
        category = c4.selectbox("Category (customer-selected)", CATEGORIES,
                                index=CATEGORIES.index(ex["category"]))
        c5, c6, c7, c8 = st.columns(4)
        date = c5.date_input("Created date", dt.date.today())
        time = c6.time_input("Created time", dt.time(10, 30))
        prior = c7.number_input("Tickets in last 30 days", 0, 100, ex["prior_tickets_30d"])
        attachments = c8.number_input("Attachments", 0, 50, ex["attachments"])
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        ticket = dict(subject=subject, description=description, channel=channel, customer_tier=tier,
                      product=product, category=category, prior_tickets_30d=int(prior),
                      attachments=int(attachments), created_at=dt.datetime.combine(date, time).isoformat())
        try:
            r = requests.post(f"{API_URL}/predict", params={"explain": "true"}, json=ticket, timeout=30)
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()
        if r.status_code != 200:
            st.error(r.json().get("details") or r.json().get("error"))
            st.stop()
        res = r.json()

        color = PRIORITY_COLORS[res["priority"]]
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"**Priority**<br><span style='font-size:2rem;font-weight:700;color:{color}'>"
                    f"{res['priority']}</span>", unsafe_allow_html=True)
        m2.metric("Confidence", f"{res['confidence']:.0%}")
        hrs = res["resolution_hours"]
        m3.metric("Predicted resolution", f"{hrs:.1f} h" if hrs < 48 else f"{hrs / 24:.1f} days")
        m4.metric("SLA target", f"{res['sla_target_hours']} h",
                  delta="breach risk" if res["sla_breach_risk"] else "within SLA",
                  delta_color="inverse" if res["sla_breach_risk"] else "normal")

        probs = pd.DataFrame({"priority": list(res["priority_probabilities"]),
                              "probability": list(res["priority_probabilities"].values())})
        st.altair_chart(alt.Chart(probs).mark_bar(cornerRadius=2).encode(
            x=alt.X("priority:N", sort=list(PRIORITY_COLORS)[::-1], title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("priority:N", legend=None, scale=alt.Scale(domain=list(PRIORITY_COLORS),
                                                                       range=list(PRIORITY_COLORS.values()))),
            tooltip=["priority", alt.Tooltip("probability:Q", format=".1%")],
        ).properties(height=200), use_container_width=True)

        exp = res["explanation"]
        left, right = st.columns(2)
        with left:
            st.subheader(f"Why {res['priority']}?")
            st.caption("SHAP contributions to the predicted class (log-odds). Positive pushes toward it.")
            contributions_chart(exp["priority"], f"toward {res['priority']}", f"away from {res['priority']}")
            st.caption("Words driving the text signal: " + ", ".join(
                f"`{t['term']}` ({t['weight']:+.2f})" for t in exp["priority_key_terms"]))
        with right:
            st.subheader("Why this resolution time?")
            st.caption("SHAP contributions in log-hours. Positive means slower.")
            contributions_chart(exp["resolution_time"], "slower", "faster")
            st.caption("Words driving the text signal: " + ", ".join(
                f"`{t['term']}` ({t['weight']:+.2f})" for t in exp["resolution_key_terms"]))
        st.caption(f"API latency: {res['latency_ms']} ms")

with tab_model:
    try:
        m = get("/model")
    except requests.RequestException:
        st.info("Model metrics are not available from the API.")
        st.stop()
    p, rt = m["priority"], m["resolution_time"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Priority macro-F1", f"{p['final_test']['macro_f1']:.3f}")
    c2.metric("Priority accuracy", f"{p['final_test']['accuracy']:.3f}")
    c3.metric("Resolution MAE", f"{rt['final_test']['mae_hours']:.1f} h")
    c4.metric("Resolution median AE", f"{rt['final_test']['median_ae_hours']:.1f} h")

    st.subheader("Model comparison (validation set)")
    left, right = st.columns(2)
    left.dataframe(pd.DataFrame({k: v["val"] for k, v in p["comparison"].items()}).T, width="stretch")
    right.dataframe(pd.DataFrame({k: v["val"] for k, v in rt["comparison"].items()}).T, width="stretch")

    st.subheader("Priority confusion matrix (test set)")
    labels = p["confusion_matrix"]["labels"]
    st.dataframe(pd.DataFrame(p["confusion_matrix"]["matrix"], index=[f"true {x}" for x in labels],
                              columns=[f"pred {x}" for x in labels]), width="stretch")

    st.subheader("Global feature importance (mean |SHAP|)")
    left, right = st.columns(2)
    for col, block, title in ((left, p, "Priority"), (right, rt, "Resolution time")):
        with col:
            st.caption(title)
            importance_chart(block["shap_importance"][:12])
