"""Synthetic support-ticket generator.

Real ticket corpora are proprietary, so this module generates a realistic dataset
with a known causal structure: priority and resolution time are driven by the
ticket's category, customer tier, urgency language, blast radius, channel and
timing - plus noise. The text is what a customer would actually write, and the
user-selected `category` field is deliberately noisy (customers mis-file tickets).

    python -m triage.data --n 20000 --out data/tickets.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

PRIORITIES = ["Low", "Medium", "High", "Critical"]
CHANNELS = ["email", "web", "chat", "phone"]
TIERS = ["free", "pro", "enterprise"]
PRODUCTS = ["Dashboard", "Mobile App", "API", "Billing Portal", "Reports", "Integrations Hub"]

# category -> (severity, base resolution hours, subjects, bodies)
CATEGORIES = {
    "outage": (3.0, 4.0, [
        "{product} is down", "Cannot access {product} at all", "Service outage on {product}",
        "{product} returning 503 errors", "Site not loading",
    ], [
        "{product} has been completely unavailable since {time}. We get {error} on every request.",
        "Nothing loads in {product}. The page just spins and then shows {error}.",
        "Our whole workspace is unreachable. Status page says operational but we see {error}.",
    ]),
    "security": (2.6, 10.0, [
        "Suspicious login activity", "Possible account compromise", "Unauthorized access to our account",
        "Security concern with API keys", "Phishing email claiming to be from you",
    ], [
        "We noticed logins from an unknown location ({country}) on an admin account.",
        "An API key that we rotated last week is still being accepted by {product}.",
        "A user reports seeing another customer's data in {product}. This looks like a data leak.",
    ]),
    "data_loss": (2.6, 20.0, [
        "Data missing from {product}", "Records deleted unexpectedly", "Lost all my reports",
        "Export is missing rows", "Data disappeared after sync",
    ], [
        "About {n} records vanished from {product} after the last sync with {integration}.",
        "Our saved reports are gone. Nobody on the team deleted them.",
        "The CSV export only has part of our data, roughly {n} rows are missing.",
    ]),
    "bug": (1.6, 40.0, [
        "Bug in {product}", "{feature} not working", "Error when using {feature}",
        "{feature} shows wrong values", "Unexpected behavior in {product}",
    ], [
        "When I click {feature} in {product} I get {error}. Steps to reproduce attached.",
        "{feature} shows incorrect totals since the last update. Tested on {browser}.",
        "The {feature} button does nothing on {browser}. Console shows {error}.",
    ]),
    "performance": (1.5, 30.0, [
        "{product} is very slow", "Slow load times", "Timeouts on {feature}",
        "Performance degradation in {product}", "Reports take forever",
    ], [
        "{feature} takes over {n} seconds to load, it used to be instant.",
        "We keep hitting timeouts on {product} during peak hours.",
        "Since yesterday {product} has been really sluggish for large accounts.",
    ]),
    "login_access": (1.3, 5.0, [
        "Can't log in", "Password reset not working", "Locked out of account",
        "2FA code not accepted", "SSO login failing",
    ], [
        "I reset my password but the new one is rejected with {error}.",
        "My two factor code is never accepted and I am now locked out.",
        "SSO redirects back to the login page in a loop on {browser}.",
    ]),
    "billing": (1.0, 12.0, [
        "Question about my invoice", "Charged twice", "Refund request",
        "Update payment method", "Wrong amount on invoice",
    ], [
        "We were charged twice this month for the {tier} plan. Please refund the duplicate.",
        "The invoice shows {n} seats but we only have half that many users.",
        "How do I update the card on file? The billing portal gives {error}.",
    ]),
    "integration": (1.2, 24.0, [
        "{integration} integration broken", "Sync with {integration} failing",
        "Webhook not firing", "Cannot connect {integration}", "API returns {error}",
    ], [
        "Our {integration} sync stopped working and the logs show {error}.",
        "Webhooks to our endpoint stopped arriving about {n} hours ago.",
        "Connecting {integration} fails at the OAuth step with {error}.",
    ]),
    "how_to": (0.3, 6.0, [
        "How do I export data?", "Question about {feature}", "How to add users",
        "Where is the {feature} setting?", "Need help getting started",
    ], [
        "Could you point me to the docs for {feature}? I could not find it.",
        "How do I add new team members to {product}? Thanks in advance.",
        "Is there a way to schedule {feature} to run every week?",
    ]),
    "feature_request": (0.0, 72.0, [
        "Feature request: {feature}", "Suggestion for {product}", "Would love dark mode",
        "Can you add more export formats?", "Idea to improve {feature}",
    ], [
        "It would be great if {feature} supported filtering by date.",
        "Please consider adding a dark mode to {product}. Our team would love it.",
        "Suggestion: let us customize the columns in {feature}.",
    ]),
}

URGENCY = [
    "This is urgent.", "Please fix ASAP.", "We need this resolved immediately.",
    "This is blocking our production release.", "Critical issue for our business.",
    "Our customers are affected right now.",
]
SCOPE = {
    "single": ["Only my account seems affected.", "Just me as far as I can tell.", ""],
    "team": ["Several people on my team see the same thing.", "Our whole department is affected."],
    "all": ["All of our users are affected.", "Every user in the company is impacted.",
            "This is affecting everyone in production."],
}
FRUSTRATION = ["This is unacceptable!!", "Very frustrated right now!", "Third time reporting this!!!",
               "Why does this keep happening?", "PLEASE HELP"]
POLITE = ["Thanks for your help.", "Appreciate it!", "Thank you.", "Kind regards.", ""]

FILL = {
    "feature": ["the export button", "dashboards", "scheduled reports", "user management", "search",
                "the analytics chart", "notifications", "the audit log"],
    "error": ["error 500", "HTTP 503", "ERR_TIMEOUT", "error code E1043", "a blank white screen",
              "'Something went wrong'", "401 Unauthorized"],
    "browser": ["Chrome", "Safari", "Firefox", "Edge", "the iOS app"],
    "integration": ["Salesforce", "Slack", "HubSpot", "Zapier", "Jira", "Snowflake"],
    "country": ["Brazil", "Romania", "Vietnam", "an unknown VPN"],
    "time": ["this morning", "about an hour ago", "last night", "9am"],
    "tier": ["Pro", "Enterprise", "Team"],
}

PRIORITY_MULT = {"Critical": 0.35, "High": 0.6, "Medium": 1.0, "Low": 1.6}
TIER_MULT = {"enterprise": 0.7, "pro": 0.9, "free": 1.3}
CHANNEL_MULT = {"phone": 0.8, "chat": 0.85, "web": 1.0, "email": 1.1}


def _fill(template, rng, product):
    out = template.replace("{product}", product)
    for key, options in FILL.items():
        while "{" + key + "}" in out:
            out = out.replace("{" + key + "}", options[rng.integers(len(options))], 1)
    while "{n}" in out:
        out = out.replace("{n}", str(int(rng.choice([5, 10, 30, 120, 500, 2000]))), 1)
    return out


def generate(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    cats = list(CATEGORIES)
    cat_p = np.array([4, 3, 3, 14, 9, 12, 12, 9, 20, 14], dtype=float)
    cat_p /= cat_p.sum()

    start = pd.Timestamp("2025-07-01")
    hour_p = np.array([1, 1, 1, 1, 1, 2, 3, 5, 8, 10, 10, 9, 8, 9, 9, 8, 7, 5, 4, 3, 2, 2, 1, 1], float)
    hour_p /= hour_p.sum()

    rows = []
    for i in range(n):
        category = cats[rng.choice(len(cats), p=cat_p)]
        severity, base_hours, subjects, bodies = CATEGORIES[category]
        product = PRODUCTS[rng.integers(len(PRODUCTS))]
        tier = TIERS[rng.choice(3, p=[0.45, 0.38, 0.17])]
        channel = CHANNELS[rng.choice(4, p=[0.35, 0.3, 0.25, 0.1])]
        created = (start + pd.Timedelta(days=int(rng.integers(365))) +
                   pd.Timedelta(hours=int(rng.choice(24, p=hour_p)), minutes=int(rng.integers(60))))

        # Severe categories are more likely to include urgency and a wide blast radius.
        n_urgent = rng.binomial(2, min(0.08 + 0.18 * severity, 0.7))
        scope = rng.choice(["single", "team", "all"],
                           p=[0.7, 0.2, 0.1] if severity < 1.5 else [0.35, 0.35, 0.3])
        frustrated = rng.random() < 0.12 + 0.05 * severity

        parts = [_fill(bodies[rng.integers(len(bodies))], rng, product)]
        parts += [URGENCY[j] for j in rng.choice(len(URGENCY), n_urgent, replace=False)]
        parts.append(SCOPE[scope][rng.integers(len(SCOPE[scope]))])
        if frustrated:
            parts.append(FRUSTRATION[rng.integers(len(FRUSTRATION))])
        parts.append(POLITE[rng.integers(len(POLITE))])
        description = " ".join(p for p in parts if p)
        subject = _fill(subjects[rng.integers(len(subjects))], rng, product)
        if frustrated and rng.random() < 0.3:
            subject = subject.upper()

        attachments = int(rng.poisson(1.2 if category in ("bug", "data_loss", "integration") else 0.3))
        prior = int(rng.poisson(0.8 + (1.5 if frustrated else 0)))

        # Interactions seen in real queues: free-tier customers call everything "urgent" so agents
        # discount it, and a wide blast radius only escalates incidents (not how-to questions).
        urgency_weight = 0.3 if tier == "free" else 0.9
        incident = category in ("outage", "security", "data_loss", "performance")
        latent = (severity + {"free": 0.0, "pro": 0.5, "enterprise": 1.0}[tier] + urgency_weight * n_urgent +
                  {"single": 0.0, "team": 0.5, "all": 1.2}[scope] * (1.0 if incident else 0.3) +
                  0.3 * frustrated + 0.15 * min(prior, 4) + rng.normal(0, 0.6))

        # Customers pick the wrong category ~15% of the time.
        shown_category = category if rng.random() > 0.15 else cats[rng.integers(len(cats))]

        rows.append(dict(
            ticket_id=f"T{100000 + i}", created_at=created, subject=subject, description=description,
            channel=channel, product=product, customer_tier=tier, category=shown_category,
            prior_tickets_30d=prior, attachments=attachments,
            _latent=latent, _base_hours=base_hours, _true_category=category,
        ))

    df = pd.DataFrame(rows)
    cuts = df["_latent"].quantile([0.25, 0.60, 0.88]).to_numpy()
    df["priority"] = pd.cut(df["_latent"], [-np.inf, *cuts, np.inf], labels=PRIORITIES).astype(str)

    hour = df["created_at"].dt.hour
    # Enterprise has 24/7 support, so off-hours tickets only wait longer for other tiers.
    staffed = df["customer_tier"] == "enterprise"
    weekend = (df["created_at"].dt.dayofweek >= 5) & ~staffed
    after_hours = ((hour < 8) | (hour >= 18)) & ~staffed
    # Feature requests go to the product backlog; triage priority barely changes their timeline.
    priority_mult = np.where(df["_true_category"] == "feature_request", 1.0, df["priority"].map(PRIORITY_MULT))
    hours = (df["_base_hours"] * priority_mult *
             df["customer_tier"].map(TIER_MULT) * df["channel"].map(CHANNEL_MULT) *
             np.where(weekend, 1.6, 1.0) * np.where(after_hours, 1.25, 1.0) *
             np.where(df["attachments"] > 0, 0.9, 1.0) *
             np.exp(rng.normal(0, 0.45, len(df))))
    df["resolution_hours"] = hours.clip(0.25, 500).round(2)
    return df.drop(columns=["_latent", "_base_hours", "_true_category"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/tickets.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} tickets -> {args.out}")
    print(df["priority"].value_counts().to_string())
    print(df["resolution_hours"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
