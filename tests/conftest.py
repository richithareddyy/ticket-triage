import subprocess
import sys

import pytest

from triage.data import generate

SAMPLE_TICKET = {
    "subject": "Production dashboard is down",
    "description": "Nothing loads, we get HTTP 503 on every request. All of our users are affected. This is urgent!",
    "channel": "phone", "product": "Dashboard", "customer_tier": "enterprise", "category": "outage",
    "created_at": "2026-09-27T02:15:00", "prior_tickets_30d": 2, "attachments": 1,
}


@pytest.fixture(scope="session")
def tickets():
    return generate(n=2500, seed=7)


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory, tickets):
    """Train a small model end-to-end via the real CLI."""
    root = tmp_path_factory.mktemp("run")
    data = root / "tickets.csv"
    tickets.to_csv(data, index=False)
    subprocess.run([sys.executable, "-m", "triage.train", "--data", str(data), "--out", str(root / "artifacts")],
                   check=True, capture_output=True)
    return root / "artifacts"


@pytest.fixture(scope="session")
def predictor(artifacts):
    from triage.predict import Predictor
    return Predictor(str(artifacts / "model.joblib"))
