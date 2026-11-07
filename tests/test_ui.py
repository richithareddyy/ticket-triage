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


def test_page_leads_from_the_tickets_into_the_form(app):
    assert not app.exception
    text = markdown_text(app)
    assert "Every ticket says it's urgent." in text and 'href="#open-a-ticket"' in text
    assert 'id="open-a-ticket"' in text and "No activity yet." in text
    assert [t.label for t in app.tabs] == ["Incident", "Model evaluation"]


def test_run_triage_shows_verdict_and_explanations(app):
    app.button[0].click().run()
    assert not app.exception
    text = markdown_text(app)
    assert "1 - Critical" in text and "What the model noted" in text and "Main factors" in text
    assert "No activity yet." not in text


def test_sample_ticket_fills_the_form(app):
    app.selectbox[0].set_value("Feature request").run()
    assert app.text_input[0].value == "Would love dark mode"


def test_empty_ticket_shows_warning(app):
    app.text_input[0].set_value("")
    app.text_area[0].set_value("   ")
    app.button[0].click().run()
    assert any("Add a short description" in w.value for w in app.warning)
    assert "What the model noted" not in markdown_text(app)


def test_model_tab_renders_metrics(app):
    text = markdown_text(app)
    assert "Synthetic test set" in text and "Models compared" in text and "Confusion matrix" in text
    assert "Priority macro-F1" in text
    assert "real Apache Jira bug reports" in text and "0.360" in text


def test_new_sample_keeps_last_result_faded(app):
    app.button[0].click().run()
    app.selectbox[0].set_value("Feature request").run()
    text = markdown_text(app)
    assert "Last result. Press Triage to score this ticket." in text
    assert "What the model noted" in text and "tt-stale" in text
