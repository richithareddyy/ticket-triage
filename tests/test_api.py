import pytest

from api.app import create_app

from .conftest import SAMPLE_TICKET


@pytest.fixture(scope="module")
def client(predictor):
    app = create_app(predictor)
    app.testing = True
    return app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["status"] == "ok"
    assert set(r.json["models"]) == {"priority", "resolution"}


def test_predict_single(client):
    r = client.post("/predict", json=SAMPLE_TICKET)
    assert r.status_code == 200
    assert {"priority", "confidence", "resolution_hours", "sla_breach_risk", "latency_ms"} <= set(r.json)
    assert "explanation" not in r.json


def test_predict_with_explanation(client):
    r = client.post("/predict?explain=true", json=SAMPLE_TICKET)
    assert r.status_code == 200
    assert {"priority", "resolution_time", "priority_key_terms"} <= set(r.json["explanation"])


def test_predict_batch_keeps_ids(client):
    r = client.post("/predict", json={"tickets": [{**SAMPLE_TICKET, "ticket_id": "A"},
                                                  {**SAMPLE_TICKET, "ticket_id": "B"}]})
    assert r.status_code == 200
    assert [p["ticket_id"] for p in r.json["predictions"]] == ["A", "B"]


def test_optional_fields_default(client):
    minimal = {k: SAMPLE_TICKET[k] for k in ("subject", "description", "channel", "product", "customer_tier")}
    assert client.post("/predict", json=minimal).status_code == 200


@pytest.mark.parametrize("payload, message", [
    ({**SAMPLE_TICKET, "channel": "fax"}, "channel"),
    ({**SAMPLE_TICKET, "attachments": -1}, "attachments"),
    ({**SAMPLE_TICKET, "prior_tickets_30d": "3"}, "prior_tickets_30d"),
    ({**SAMPLE_TICKET, "created_at": "not a date"}, "created_at"),
    ({**SAMPLE_TICKET, "subject": "", "description": "  "}, "both empty"),
    ({k: v for k, v in SAMPLE_TICKET.items() if k != "description"}, "description"),
])
def test_validation_errors(client, payload, message):
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
    assert any(message in d for d in r.json["details"])


def test_bad_requests(client):
    assert client.post("/predict", data="nope", content_type="text/plain").status_code == 400
    assert client.post("/predict", json={"tickets": []}).status_code == 400
    assert client.post("/predict", json={"tickets": [SAMPLE_TICKET] * 101}).status_code == 400
    assert client.get("/predict").status_code == 405
    assert client.get("/nope").status_code == 404
