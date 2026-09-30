"""Visual system for the triage UI.

Base colors live in `.streamlit/config.toml` ([theme]) and are read here, so
Streamlit widgets, Altair charts and the few custom HTML elements share one
source of truth.

    Type      Source Sans for UI; Source Code Pro only for data (terms, values).
              Scale 12 / 13 / 14 / 16 / 20 / 26 / 36 px. Sentence case everywhere.
    Spacing   4 px base: 4, 8, 12, 16, 24, 32, 48.
    Radius    4 px on inputs, buttons and the prediction panel; nothing else rounds.
    Surfaces  paper page (#FAFAF7); white inputs = editable;
              tinted panel (#F3F2EC) = model output. One level only, no nesting.
    Lines     1 px rules (#D9D7CF) separate sections instead of boxes.
    Color     one accent (teal) for actions, focus, links and "lowers";
              priority colors appear only on the prediction itself.
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
INK = _opt("textColor", "#1D2126")
RULE = _opt("borderColor", "#D9D7CF")
MUTED = _opt("grayColor", "#62666D")
PANEL = "#F3F2EC"
TRACK = "#E6E4DC"

# Ordinal priority scale; each color meets 4.5:1 contrast on paper and panel.
PRIORITY = {"Low": "#3F7D58", "Medium": "#8A6D10", "High": "#B4561F", "Critical": "#B3261E"}
OK, WARN = "#3F7D58", "#B4561F"
# SHAP direction: warm raises the prediction, accent lowers it.
UP, DOWN = "#B4561F", ACCENT
MONO = '"Source Code Pro", ui-monospace, SFMono-Regular, Menlo, monospace'

CSS = f"""
<style>
.block-container {{ max-width: 1180px; padding: 2.75rem 2rem 2rem; }}

/* Header: one line of identity, metadata pushed right and quiet */
.tt-title {{ font-size: 1.625rem; font-weight: 700; line-height: 1.2; color: {INK}; letter-spacing: -.01em; }}
.tt-sub {{ color: {MUTED}; font-size: .9375rem; margin-top: .25rem; }}
.tt-meta {{ color: {MUTED}; font-size: .8125rem; text-align: right; line-height: 1.5;
  font-variant-numeric: tabular-nums; }}
.tt-meta .dot {{ color: {OK}; }}

/* Section headings inside the workspace */
.tt-section {{ font-size: 1rem; font-weight: 650; color: {INK}; line-height: 1.5; padding-bottom: .6rem; }}
.tt-section.context {{ margin-top: .5rem; padding-top: .75rem; border-top: 1px solid {RULE}; line-height: 1.5; }}
.tt-small {{ font-size: .8125rem; font-weight: 600; color: {MUTED}; }}
.tt-muted {{ color: {MUTED}; font-size: .875rem; }}

/* Prediction panel: the one tinted surface on the page */
[class*="st-key-panel"] {{ background: {PANEL}; border: 1px solid {RULE}; border-radius: 4px;
  padding: 1.25rem 1.5rem 1.1rem; }}
/* Keep the prediction in view while editing a long ticket (the wrapper spans the column). */
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-panel"]) {{ position: sticky; top: 3.75rem; z-index: 1; }}
.tt-empty {{ font-size: 1.0625rem; color: {INK}; margin: .6rem 0 .2rem; }}
.tt-priority {{ border-left: 3px solid var(--p); padding-left: .85rem; margin: .5rem 0 1rem; }}
.tt-priority .v {{ font-size: 2.25rem; font-weight: 700; line-height: 1.1; color: var(--p); }}
.tt-priority .s {{ color: {MUTED}; font-size: .875rem; margin-top: .2rem; }}
.tt-rows {{ border-top: 1px solid {RULE}; margin-top: .9rem; }}
.tt-row {{ display: flex; justify-content: space-between; align-items: baseline; gap: 1rem;
  padding: .6rem 0; border-bottom: 1px solid {RULE}; }}
.tt-row .k {{ color: {MUTED}; font-size: .875rem; }}
.tt-row .v {{ font-size: 1.0625rem; font-weight: 650; color: {INK}; text-align: right;
  font-variant-numeric: tabular-nums; }}
.tt-row .v small {{ font-size: .8125rem; font-weight: 600; margin-left: .45rem; }}
.tt-row .v small.ok {{ color: {OK}; }}
.tt-row .v small.warn {{ color: {WARN}; }}
.tt-row .v.empty {{ color: #A3A39B; font-weight: 500; }}
.tt-split {{ display: flex; flex-wrap: wrap; gap: .25rem 1rem; font-size: .8125rem; color: {MUTED};
  font-variant-numeric: tabular-nums; margin-top: .35rem; }}
.tt-split b {{ font-weight: 650; }}
.tt-bar {{ display: flex; height: 8px; background: {TRACK}; border-radius: 2px; overflow: hidden; }}
.tt-bar span {{ display: block; height: 100%; }}

/* Explanation */
.tt-h {{ font-size: 1.25rem; font-weight: 700; color: {INK}; margin: 2rem 0 .1rem; padding-top: 1.25rem;
  border-top: 1px solid {RULE}; }}
.tt-h3 {{ font-size: 1rem; font-weight: 650; color: {INK}; margin: .9rem 0 .1rem; }}
.tt-terms {{ font-family: {MONO}; font-size: .8125rem; line-height: 1.8; margin: .25rem 0 0; }}
.tt-terms span {{ white-space: nowrap; margin-right: 1rem; }}

/* Model tab: compact numbers separated by rules */
.tt-stats {{ display: flex; flex-wrap: wrap; border-top: 1px solid {RULE}; border-bottom: 1px solid {RULE};
  margin: .75rem 0 .5rem; }}
.tt-stat {{ flex: 1 1 10rem; padding: .8rem 1.25rem .8rem 0; }}
.tt-stat + .tt-stat {{ padding-left: 1.25rem; border-left: 1px solid {RULE}; }}
.tt-stat .k {{ color: {MUTED}; font-size: .8125rem; }}
.tt-stat .v {{ font-size: 1.5rem; font-weight: 700; color: {INK}; font-variant-numeric: tabular-nums; }}
.tt-stat .s {{ color: {MUTED}; font-size: .8125rem; }}

.tt-foot {{ border-top: 1px solid {RULE}; margin-top: 3rem; padding-top: .75rem; color: {MUTED};
  font-size: .8125rem; }}

/* Controls */
[data-testid="stForm"] {{ border: 0; padding: 0; }}
[data-testid="stFormSubmitButton"] button {{ min-height: 2.75rem; font-size: 1rem; font-weight: 650; }}
[data-testid="stFormSubmitButton"] button:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 2px; }}
[data-baseweb="tab"] {{ font-weight: 600; }}
[data-baseweb="tab-list"] {{ margin-bottom: .5rem; }}

@media (max-width: 760px) {{
  .block-container {{ padding: 3.5rem 1rem 2rem; }}
  .tt-meta {{ text-align: left; }}
  .tt-stat + .tt-stat {{ padding-left: 0; border-left: 0; }}
}}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def row(key, value_html, empty=False):
    cls = "v empty" if empty else "v"
    return f'<div class="tt-row"><span class="k">{escape(key)}</span><span class="{cls}">{value_html}</span></div>'


def priority_block(priority, note=""):
    return (f'<div class="tt-priority" style="--p:{PRIORITY[priority]}"><div class="v">{escape(priority)}</div>'
            f'<div class="s">{escape(note)}</div></div>')


def split(probabilities):
    """Stacked Low→Critical bar in plain HTML (no chart chrome needed for four numbers)."""
    order = list(PRIORITY)
    segs = "".join(f'<span style="width:{probabilities[p] * 100:.2f}%;background:{PRIORITY[p]}" '
                   f'title="{p} {probabilities[p]:.1%}"></span>' for p in order)
    legend = "".join(f'<span><b style="color:{PRIORITY[p]}">{p}</b> {probabilities[p]:.0%}</span>' for p in order)
    return (f'<div class="tt-bar" role="img" aria-label="Priority probabilities: '
            + ", ".join(f"{p} {probabilities[p]:.0%}" for p in order)
            + f'">{segs}</div><div class="tt-split">{legend}</div>')


def stats(items):
    cells = "".join(f'<div class="tt-stat"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div>'
                    f'<div class="s">{escape(s)}</div></div>' for k, v, s in items)
    return f'<div class="tt-stats">{cells}</div>'


def style_chart(chart):
    """Shared Altair styling: no frames, quiet axes, the UI's type colors."""
    return (chart.configure(background="transparent").configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#ECEAE3", domainColor=RULE,
                            tickColor=RULE, labelFontSize=12, titleFontSize=12, titleFontWeight=500)
            .configure_legend(labelColor=INK, labelFontSize=12, symbolType="square", orient="bottom")
            .configure_axisY(labelLimit=340))
