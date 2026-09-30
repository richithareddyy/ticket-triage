"""Streamlit UI smoke tests (in-process API mode, using the committed model)."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402
from streamlit.testing.v1 import element_tree  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _patch_single_select_pills(monkeypatch):
    """AppTest in Streamlit 1.50 assumes button groups hold a list and iterates a single-select
    string character by character. Wrap single values so st.pills(selection_mode="single") works."""
    original = element_tree.ButtonGroup.value

    def value(self):
        v = original.fget(self)
        return [] if v is None else (list(v) if isinstance(v, (list, tuple)) else [v])

    monkeypatch.setattr(element_tree.ButtonGroup, "value", property(value))


@pytest.fixture
def app(monkeypatch):
    _patch_single_select_pills(monkeypatch)
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
    assert "Support Ticket Triage" in text and "No prediction yet" in text
    assert [t.label for t in app.tabs][:2] == [":material/bolt: Triage a ticket",
                                               ":material/insights: Model performance"]


def test_predict_shows_results(app):
    app.button[0].click().run()
    assert not app.exception
    text = markdown_text(app)
    assert "Critical" in text and "Priority probabilities" in text and "Why Critical?" in text
    assert "No prediction yet" not in text


def test_example_pills_fill_the_form(app):
    app.button_group[0].set_value(["Feature request"]).run()
    assert app.text_input[0].value == "Would love dark mode"


def test_empty_ticket_shows_warning(app):
    app.text_input[0].set_value("")
    app.text_area[0].set_value("   ")
    app.button[0].click().run()
    assert any("Add a subject or description" in w.value for w in app.warning)
    assert "Priority probabilities" not in markdown_text(app)
