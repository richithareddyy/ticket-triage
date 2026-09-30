"""Streamlit UI smoke tests (in-process API mode, using the committed model)."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setenv("MODEL_PATH", os.path.join(ROOT, "artifacts", "model.joblib"))
    monkeypatch.setenv("METRICS_PATH", os.path.join(ROOT, "artifacts", "metrics.json"))
    monkeypatch.syspath_prepend(ROOT)
    at = AppTest.from_file(os.path.join(ROOT, "ui", "streamlit_app.py"), default_timeout=120)
    return at.run()


def markdown_text(at):
    return " ".join(m.value for m in at.markdown)


def test_renders_header_and_empty_state(app):
    assert not app.exception
    text = markdown_text(app)
    assert "Ticket Triage" in text and "then run triage" in text
    assert [t.label for t in app.tabs] == ["Triage", "How the model performs"]


def test_run_triage_shows_verdict_and_explanations(app):
    app.button[0].click().run()
    assert not app.exception
    text = markdown_text(app)
    assert "Critical" in text and "Priority split" in text and "Why Critical" in text
    assert "then run triage" not in text


def test_sample_ticket_fills_the_form(app):
    app.selectbox[0].set_value("Feature request").run()
    assert app.text_input[0].value == "Would love dark mode"


def test_empty_ticket_shows_warning(app):
    app.text_input[0].set_value("")
    app.text_area[0].set_value("   ")
    app.button[0].click().run()
    assert any("Add a subject or description" in w.value for w in app.warning)
    assert "Priority split" not in markdown_text(app)


def test_model_tab_renders_metrics(app):
    text = markdown_text(app)
    assert "Held-out test set" in text and "Models compared" in text and "Where it goes wrong" in text
