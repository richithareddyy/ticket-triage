"""Visual language for the triage UI.

Base colors live in `.streamlit/config.toml` ([theme]) and are read here, so
Streamlit widgets, Altair charts and the few custom HTML elements share one
source of truth. The look borrows from support tooling: paper background, ink
text, one teal accent, monospace field labels, rules instead of boxes, and
priority color used only where it carries meaning.
"""
from html import escape

import streamlit as st


def _opt(name, fallback):
    try:
        return st.get_option(f"theme.{name}") or fallback
    except Exception:
        return fallback


ACCENT = _opt("primaryColor", "#0E6B6B")
PAPER = _opt("backgroundColor", "#FAFAF7")
SUBTLE = _opt("secondaryBackgroundColor", "#F1F0EA")
INK = _opt("textColor", "#1D2126")
RULE = _opt("borderColor", "#DDDBD3")
MUTED = _opt("grayColor", "#6A6E75")

# Ordinal priority scale; each color meets 4.5:1 contrast on the paper background.
PRIORITY = {"Low": "#3F7D58", "Medium": "#8A6D10", "High": "#B4561F", "Critical": "#B3261E"}
OK, WARN = "#3F7D58", "#B4561F"
# SHAP direction: warm raises the prediction, accent lowers it.
UP, DOWN = "#B4561F", ACCENT
MONO = '"Source Code Pro", ui-monospace, SFMono-Regular, Menlo, monospace'

CSS = f"""
<style>
.block-container {{ max-width: 1120px; padding-top: 3.4rem; padding-bottom: 2.5rem; }}

.tt-eyebrow, .tt-label {{ font-family: {MONO}; font-size: .72rem; letter-spacing: .08em;
  text-transform: uppercase; color: {MUTED}; margin: 0; }}
.tt-eyebrow {{ margin: 0 0 .2rem; }}
.tt-title {{ font-size: 2rem; font-weight: 700; margin: 0; line-height: 1.15; color: {INK}; letter-spacing: -.01em; }}
.tt-lede {{ color: {MUTED}; margin: .35rem 0 .2rem; font-size: 1.02rem; }}
.tt-meta {{ font-family: {MONO}; font-size: .78rem; color: {MUTED}; margin: 0 0 1rem; }}
.tt-meta .ok {{ color: {OK}; }}

.tt-h {{ font-size: 1.05rem; font-weight: 650; color: {INK}; margin: 1.6rem 0 .15rem;
  padding-top: .9rem; border-top: 1px solid {RULE}; }}
.tt-h.first {{ border-top: 0; padding-top: 0; margin-top: .4rem; }}
.tt-note {{ color: {MUTED}; font-size: .9rem; margin: 0 0 .6rem; }}

/* Triage verdict: a priority stripe, then facts separated by rules */
.tt-verdict {{ display: flex; flex-wrap: wrap; align-items: stretch;
  border-left: 4px solid var(--p); padding: .2rem 0 .2rem 1.1rem; margin: .6rem 0 1rem; }}
.tt-verdict .main {{ padding-right: 2rem; min-width: 11rem; }}
.tt-priority {{ font-size: 2.1rem; font-weight: 750; line-height: 1.1; color: var(--p); margin-top: .1rem; }}
.tt-verdict .conf {{ font-size: .82rem; color: {MUTED}; margin-top: .15rem; }}
.tt-fact {{ padding: 0 1.6rem; border-left: 1px solid {RULE}; min-width: 8.5rem; }}
.tt-fact .v {{ font-size: 1.35rem; font-weight: 650; color: {INK}; line-height: 1.3; margin-top: .15rem; }}
.tt-fact .s {{ font-size: .82rem; color: {MUTED}; margin-top: .1rem; }}
.tt-fact .s.ok {{ color: {OK}; }}
.tt-fact .s.warn {{ color: {WARN}; font-weight: 600; }}
.tt-facts {{ display: flex; flex-wrap: wrap; margin: .4rem 0 .6rem; }}
.tt-facts .tt-fact:first-child {{ border-left: 0; padding-left: 0; }}

.tt-split {{ font-family: {MONO}; font-size: .78rem; color: {MUTED}; display: flex; flex-wrap: wrap;
  gap: .35rem 1.2rem; margin: -.2rem 0 .4rem; }}
.tt-split b {{ font-weight: 600; }}
.tt-terms {{ font-family: {MONO}; font-size: .8rem; color: {MUTED}; margin: .1rem 0 .4rem; line-height: 1.8; }}
.tt-terms span {{ white-space: nowrap; margin-right: .9rem; }}
.tt-empty {{ color: {MUTED}; padding: 1.1rem 0 0; border-top: 1px solid {RULE}; margin-top: 1.2rem; }}
.tt-foot {{ border-top: 1px solid {RULE}; margin-top: 2.5rem; padding-top: .8rem; color: {MUTED};
  font-size: .82rem; }}

[data-testid="stFormSubmitButton"] button {{ padding: 0 1.3rem; font-weight: 600; }}
[data-baseweb="tab"] {{ font-weight: 600; }}
[data-testid="stForm"] {{ border: 0; padding: 0; }}

@media (max-width: 640px) {{
  .block-container {{ padding: 3.6rem 1rem 2rem; }}
  .tt-title {{ font-size: 1.6rem; }}
  .tt-verdict .main {{ width: 100%; padding: 0 0 .7rem; }}
  .tt-fact {{ border-left: 0; padding: .5rem 1.4rem 0 0; min-width: 45%; }}
  .tt-facts .tt-fact {{ padding-left: 0; }}
}}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def heading(text, note="", first=False):
    cls = "tt-h first" if first else "tt-h"
    note_html = f'<div class="tt-note">{escape(note)}</div>' if note else ""
    html(f'<div class="{cls}">{escape(text)}</div>{note_html}')


def label(text):
    html(f'<div class="tt-label">{escape(text)}</div>')


def fact(name, value, sub="", tone=""):
    tone_cls = f" {tone}" if tone else ""
    sub_html = f'<div class="s{tone_cls}">{escape(sub)}</div>' if sub else ""
    return (f'<div class="tt-fact"><div class="tt-label">{escape(name)}</div>'
            f'<div class="v">{escape(value)}</div>{sub_html}</div>')


def facts(items_html):
    return f'<div class="tt-facts">{items_html}</div>'


def verdict(priority, confidence_note, facts_html):
    return (f'<div class="tt-verdict" style="--p:{PRIORITY[priority]}">'
            f'<div class="main"><div class="tt-label">Priority</div>'
            f'<div class="tt-priority">{escape(priority)}</div>'
            f'<div class="conf">{escape(confidence_note)}</div></div>'
            f'{facts_html}</div>')


def style_chart(chart):
    """Shared Altair styling: no frames, quiet axes, the UI's type colors."""
    return (chart.configure(background="transparent").configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#ECEAE3", domainColor=RULE,
                            tickColor=RULE, labelFontSize=12, titleFontSize=11, titleFontWeight=500)
            .configure_legend(labelColor=INK, labelFontSize=12, symbolType="square")
            .configure_axisY(labelLimit=320))
