"""Design system for the Streamlit UI.

The base palette lives in `.streamlit/config.toml` ([theme]) and is read here, so
Streamlit widgets, Altair charts and the few custom HTML components share one
source of truth. This module adds the semantic tokens Streamlit has no slot for
(priority colors, SHAP direction colors, surfaces, shadows) and a small CSS layer.
"""
from html import escape

import altair as alt
import streamlit as st


def _opt(name, fallback):
    try:
        return st.get_option(f"theme.{name}") or fallback
    except Exception:
        return fallback


PRIMARY = _opt("primaryColor", "#3B5BDB")
BACKGROUND = _opt("backgroundColor", "#F6F7FB")
SUBTLE = _opt("secondaryBackgroundColor", "#EEF1F7")
TEXT = _opt("textColor", "#1B2333")
BORDER = _opt("borderColor", "#DCE1EA")
MUTED = _opt("grayColor", "#687287")
SURFACE = "#FFFFFF"

# Ordinal priority scale: fg/bg pairs meet WCAG AA contrast for badge text.
PRIORITY = {
    "Low": {"fg": "#1F7A4D", "bg": "#E3F4EA", "bar": "#3C9D6D"},
    "Medium": {"fg": "#7A5E00", "bg": "#FBF3D5", "bar": "#D4A72C"},
    "High": {"fg": "#A84A12", "bg": "#FCE9DC", "bar": "#DD7A37"},
    "Critical": {"fg": "#B42D2D", "bg": "#FBE3E1", "bar": "#C94A43"},
}
STATUS = {
    "ok": {"fg": "#1F7A4D", "bg": "#E3F4EA"},
    "warn": {"fg": "#A84A12", "bg": "#FCE9DC"},
    "info": {"fg": PRIMARY, "bg": "#E7ECFB"},
    "neutral": {"fg": MUTED, "bg": SUBTLE},
}
# SHAP direction: warm pushes the prediction up, primary pulls it down.
UP, DOWN = "#D2553F", PRIMARY

CSS = f"""
<style>
:root {{
  --tt-primary: {PRIMARY}; --tt-bg: {BACKGROUND}; --tt-surface: {SURFACE}; --tt-subtle: {SUBTLE};
  --tt-text: {TEXT}; --tt-muted: {MUTED}; --tt-border: {BORDER};
  --tt-radius: 12px; --tt-shadow: 0 1px 2px rgba(16, 24, 40, .05), 0 1px 3px rgba(16, 24, 40, .04);
}}
.block-container {{ max-width: 1200px; padding-top: 3.2rem; padding-bottom: 3rem; }}

/* Cards: st.container(key="card-...") */
[class*="st-key-card"] {{
  background: var(--tt-surface); border: 1px solid var(--tt-border); border-radius: var(--tt-radius);
  box-shadow: var(--tt-shadow); padding: 1.1rem 1.25rem 1.2rem; animation: tt-fade .18s ease-out;
}}
[data-testid="stMetric"] {{ background: var(--tt-surface); box-shadow: var(--tt-shadow); }}

/* Header */
.tt-header h1 {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -.01em; margin: 0 0 .25rem; padding: 0; }}
.tt-header p {{ color: var(--tt-muted); margin: 0 0 .75rem; font-size: 1rem; }}
.tt-chips {{ display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .4rem; }}

/* Section titles inside cards */
.tt-section {{ font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  color: var(--tt-muted); margin: 0 0 .35rem; }}
.tt-card-title {{ font-size: 1.1rem; font-weight: 650; margin: 0 0 .15rem; }}
.tt-card-sub {{ color: var(--tt-muted); font-size: .88rem; margin: 0 0 .6rem; }}

/* Badges / chips */
.tt-badge {{ display: inline-flex; align-items: center; gap: .35rem; border-radius: 999px;
  padding: .18rem .65rem; font-size: .82rem; font-weight: 600; line-height: 1.4; white-space: nowrap; }}
.tt-badge.lg {{ font-size: 1.35rem; padding: .25rem .9rem; font-weight: 700; }}
.tt-dot {{ width: .5rem; height: .5rem; border-radius: 50%; background: currentColor; display: inline-block; }}
.tt-terms {{ display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .25rem; }}
.tt-term {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .8rem; border-radius: 6px;
  padding: .12rem .45rem; border: 1px solid var(--tt-border); background: var(--tt-subtle); }}

/* Stat cards */
.tt-stat {{ background: var(--tt-surface); border: 1px solid var(--tt-border); border-radius: var(--tt-radius);
  box-shadow: var(--tt-shadow); padding: .9rem 1.1rem; height: 100%; min-height: 118px;
  animation: tt-fade .18s ease-out; }}
.tt-stat .label {{ color: var(--tt-muted); font-size: .85rem; font-weight: 600; margin-bottom: .35rem; }}
.tt-stat .value {{ font-size: 1.75rem; font-weight: 700; line-height: 1.2; color: var(--tt-text); }}
.tt-stat .sub {{ color: var(--tt-muted); font-size: .85rem; margin-top: .35rem; }}

/* Empty state */
.tt-empty {{ text-align: center; padding: 2.2rem 1rem; color: var(--tt-muted); border: 1.5px dashed var(--tt-border);
  border-radius: var(--tt-radius); background: rgba(255, 255, 255, .6); }}
.tt-empty .title {{ color: var(--tt-text); font-weight: 650; font-size: 1.05rem; margin-bottom: .25rem; }}

/* Controls */
button, [data-baseweb="tab"], [data-baseweb="select"] > div, input, textarea {{
  transition: background-color .15s ease, border-color .15s ease, box-shadow .15s ease, color .15s ease; }}
[data-testid="stFormSubmitButton"] button {{ min-height: 2.6rem; padding: 0 1.4rem; font-weight: 600; }}
[data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 1px solid var(--tt-border); }}
[data-baseweb="tab"] {{ font-weight: 600; padding: .55rem .9rem; border-radius: 8px 8px 0 0; }}
[data-baseweb="tab"]:hover {{ background: var(--tt-subtle); }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--tt-border); }}
[data-testid="stSidebar"] a {{ text-decoration: none; }}

@keyframes tt-fade {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: none; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
@media (max-width: 640px) {{
  .block-container {{ padding-top: 3.6rem; padding-left: 1rem; padding-right: 1rem; }}
  [class*="st-key-card"] {{ padding: .9rem .95rem 1rem; }}
  .tt-header h1 {{ font-size: 1.5rem; }}
  .tt-stat {{ min-height: 0; }}
}}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def badge(text, fg, bg, large=False, dot=True):
    size = " lg" if large else ""
    dot_html = '<span class="tt-dot"></span>' if dot else ""
    return (f'<span class="tt-badge{size}" style="color:{fg};background:{bg}">'
            f'{dot_html}{escape(str(text))}</span>')


def priority_badge(priority, large=False):
    c = PRIORITY[priority]
    return badge(priority, c["fg"], c["bg"], large=large)


def status_badge(text, kind="neutral", dot=True):
    c = STATUS[kind]
    return badge(text, c["fg"], c["bg"], dot=dot)


def stat_card(label, value_html, sub_html=""):
    sub = f'<div class="sub">{sub_html}</div>' if sub_html else ""
    return (f'<div class="tt-stat"><div class="label">{escape(label)}</div>'
            f'<div class="value">{value_html}</div>{sub}</div>')


def card_header(title, subtitle=""):
    sub = f'<p class="tt-card-sub">{escape(subtitle)}</p>' if subtitle else ""
    st.markdown(f'<p class="tt-card-title">{escape(title)}</p>{sub}', unsafe_allow_html=True)


def section_label(text):
    st.markdown(f'<p class="tt-section">{escape(text)}</p>', unsafe_allow_html=True)


def empty_state(title, body):
    st.markdown(f'<div class="tt-empty"><div class="title">{escape(title)}</div>'
                f'<div>{escape(body)}</div></div>', unsafe_allow_html=True)


def style_chart(chart):
    """Shared Altair styling so every chart matches the palette and type scale."""
    return (chart.configure(background="transparent").configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#E9EDF3", domainColor=BORDER,
                            tickColor=BORDER, labelFontSize=12, titleFontSize=12, titleFontWeight=600)
            .configure_legend(labelColor=TEXT, labelFontSize=12, symbolType="circle")
            .configure_axisY(labelLimit=320))
