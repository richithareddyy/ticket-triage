"""Look and small HTML builders for the Streamlit UI: incident tickets, led by priority.

Base colors come from `.streamlit/config.toml` so widgets, charts and custom HTML
stay in sync. White fields are editable; grey read-only fields are set by the model.
"""
from html import escape

import streamlit as st

import art


def _opt(name, fallback):
    return st.get_option(f"theme.{name}") or fallback


ACCENT = _opt("primaryColor", "#173F4F")
PAGE = _opt("backgroundColor", "#EFE6D3")
INK = _opt("textColor", "#1E2A30")
RULE = _opt("borderColor", "#D3C6AA")
MUTED = _opt("grayColor", "#575C5E")
READONLY = "#EAE2D1"

PRIORITY = {"Low": "#2F6F4E", "Medium": "#7D6209", "High": "#A84A12", "Critical": "#B3261E"}
PRIORITY_CODE = {"Critical": "1 - Critical", "High": "2 - High", "Medium": "3 - Medium", "Low": "4 - Low"}
STAMP = {"Low": "#1E8A4C", "Medium": "#C28A00", "High": "#E0620D", "Critical": art.RED}
OK, WARN = "#2F6F4E", "#A84A12"
# SHAP direction: warm raises the prediction, accent lowers it.
UP, DOWN = "#A84A12", ACCENT
SERIF = '"Newsreader", Georgia, "Times New Roman", serif'
MONO = '"Source Code Pro", ui-monospace, SFMono-Regular, Menlo, monospace'

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;1,6..72,300;1,6..72,400&display=swap');
.block-container {{ max-width: 980px; padding: 0 1.5rem 0; }}
:root {{ --ease: cubic-bezier(.22, 1, .36, 1); }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMain"] {{ scroll-behavior: smooth; }}

[data-testid="stAppViewContainer"] {{ background: {PAGE}; }}
@media (min-width: 1280px) {{
  [data-testid="stAppViewContainer"] {{ background:
    url("{art.data_uri(art.LEFT_MARGIN)}") calc(50% - 570px) 0 / 140px auto repeat-y fixed,
    url("{art.data_uri(art.RIGHT_MARGIN)}") calc(50% + 570px) 0 / 140px auto repeat-y fixed, {PAGE}; }}
}}

.tt-hero {{ position: relative; left: 50%; width: 100vw; margin: -1.5rem 0 0 -50vw; height: calc(100vh + 1.5rem); min-height: 600px;
  background: {PAGE}; text-align: center; padding: 15vh 1.5rem 0; overflow: hidden; }}
.tt-hero h1 {{ position: relative; font-family: {SERIF}; font-weight: 400; font-size: clamp(2rem, 3.4vw, 3.1rem); line-height: 1.14;
  letter-spacing: -.01em; color: {INK}; margin: 0; padding: 0; }}
.tt-hero .lede {{ position: relative; max-width: 33rem; margin: .8rem auto 2.4rem; font-size: 1.05rem; line-height: 1.5; color: {MUTED}; }}
.tt-s {{ position: absolute; width: var(--w); left: calc(50% + var(--x)); top: var(--t); transform: rotate(var(--r)); pointer-events: none; }}
@media (max-width: 1180px) {{
  .tt-s.top {{ display: none; }}
  .tt-s {{ left: var(--n); scale: .7; transform-origin: left top; }}
}}
.tt-cta {{ position: relative; display: inline-block; background: {INK}; color: #FBF7EE !important; text-decoration: none !important; border-radius: 999px;
  padding: .95rem 2.5rem; font-size: 1.2rem; font-weight: 600; transition: transform .25s var(--ease), background-color .25s var(--ease) !important; }}
.tt-cta:hover {{ transform: translateY(-1px); }}

/* The fanned tickets, one per priority */
.tt-fan {{ position: absolute; left: 50%; bottom: -3.5rem; width: 0; height: 0; }}
.tt-tk {{ position: absolute; bottom: 0; left: calc(min(21rem, 24vw) / -2); width: min(21rem, 24vw); height: 13.5rem; background: #FFFDF8; border-radius: 8px; text-align: left;
  padding: .75rem 1rem; box-shadow: 0 14px 28px -16px rgba(30, 42, 48, .5); border: 1px solid {RULE}; border-top: .6rem solid var(--p); 
  transform: translateX(calc(var(--i) * min(16.5rem, 23vw))) translateY(var(--y)) rotate(var(--r)); }}
.tt-tk::before, .tt-tk::after {{ content: ""; position: absolute; top: 58%; width: 1.3rem; height: 1.3rem; border-radius: 50%; background: {PAGE}; border: 1px solid {RULE}; }}
.tt-tk::before {{ left: -.65rem; }}
.tt-tk::after {{ right: -.65rem; }}
.tt-tk .id {{ font-family: {MONO}; font-size: .75rem; color: {MUTED}; }}
.tt-tk .p {{ font-family: {SERIF}; font-size: 1.7rem; line-height: 1.15; color: var(--p); margin: .1rem 0; }}
.tt-tk .d {{ font-size: .875rem; color: {INK}; }}

.tt-title {{ font-family: {SERIF}; font-weight: 300; font-size: clamp(2.2rem, 4.2vw, 3.4rem); line-height: 1.08; letter-spacing: -.01em; color: {INK};
  text-align: center; margin: 3rem 0 .3rem; scroll-margin-top: 1.5rem; }}
.tt-sub {{ text-align: center; color: {MUTED}; font-size: 1.05rem; margin: 0 auto 1.5rem; max-width: 36rem; }}
.tt-head {{ display: flex; align-items: baseline; gap: .75rem; flex-wrap: wrap; margin-bottom: .4rem; }}
.tt-head .k {{ color: {MUTED}; font-size: .875rem; }}
.tt-head .n {{ font-family: {MONO}; font-size: 1.15rem; font-weight: 600; color: {INK}; }}
.tt-state {{ border: 1px solid {RULE}; border-radius: 3px; padding: 0 .45rem; font-size: .8125rem; color: {MUTED}; }}
.tt-state.done {{ border-color: {OK}; color: {OK}; }}

.tt-sec {{ font-family: {SERIF}; font-size: 1.15rem; font-weight: 400; color: {INK}; border-bottom: 1px solid {RULE}; padding-bottom: .3rem; margin: .4rem 0 .6rem; }}
.tt-f {{ margin-bottom: .65rem; }}
.tt-f .l {{ font-size: .875rem; color: {INK}; margin-bottom: .25rem; }}
.tt-f .v {{ background: {READONLY}; border: 1px solid {RULE}; border-radius: 3px; padding: .4rem .6rem; min-height: 2.2rem;
  font-size: .95rem; color: {INK}; display: flex; align-items: center; gap: .5rem; font-variant-numeric: tabular-nums; }}
.tt-f .v.empty {{ color: {MUTED}; }}

/* The verdict: the stub takes on the priority's color */
.tt-verdict {{ background: var(--p); color: #FFFDF8; border-radius: 4px; padding: .6rem .9rem; margin-bottom: .6rem; }}
.tt-verdict .c {{ font-size: .85rem; opacity: .95; }}
.tt-verdict .p {{ font-family: {SERIF}; font-size: 1.9rem; font-weight: 400; line-height: 1.1; }}
/* Triage: the stamp slams down, the ticket jolts, then the result lands piece by piece */
.tt-slam {{ display: block; width: 42%; margin: .9rem auto .2rem; transform: rotate(-7deg); position: relative; z-index: 5;
  animation: tt-slam .55s cubic-bezier(.3, 0, .2, 1) both; }}
@keyframes tt-slam {{
  0% {{ opacity: 0; transform: rotate(-15deg) scale(2.6); }}
  55% {{ opacity: 1; transform: rotate(-7deg) scale(.96); }}
  78% {{ transform: rotate(-7deg) scale(1.02); }}
  100% {{ opacity: 1; transform: rotate(-7deg) scale(1); }}
}}
@keyframes tt-thud {{ 0%, 100% {{ transform: none; }} 35% {{ transform: translateY(3px); }} 70% {{ transform: translateY(-1px); }} }}
@keyframes tt-rise {{ from {{ opacity: 0; transform: translateY(10px); }} }}
@keyframes tt-fade {{ from {{ opacity: 0; }} }}
@keyframes tt-grow {{ from {{ width: 0; }} }}
@keyframes tt-up {{ from {{ opacity: 0; translate: 0 70%; }} }}
[data-testid="stForm"]:has(.tt-landed) {{ animation: tt-thud .4s var(--ease) .3s; }}
.tt-landed .tt-verdict {{ animation: tt-rise .5s var(--ease) .32s both; }}
.tt-landed .tt-meter:first-of-type, .tt-landed .tt-split {{ animation: tt-rise .5s var(--ease) .42s both; }}
.tt-landed .tt-f {{ animation: tt-rise .5s var(--ease) .52s both; }}
.tt-landed .tt-f + .tt-f {{ animation-delay: .62s; }}
.tt-landed .tt-f .tt-meter span {{ animation: tt-grow .9s var(--ease) .75s both; }}

/* Everything else: gentle entrances and no rerun flicker */
.tt-hero h1, .tt-hero .lede, .tt-hero .tt-cta {{ animation: tt-rise .7s var(--ease) both; }}
.tt-hero .lede {{ animation-delay: .1s; }}
.tt-hero .tt-cta {{ animation-delay: .2s; }}
.tt-s {{ animation: tt-fade .6s var(--ease) .3s both; }}
.tt-s:nth-of-type(2n) {{ animation-delay: .45s; }}
.tt-s:nth-of-type(3n) {{ animation-delay: .6s; }}
.tt-tk {{ animation: tt-up .8s var(--ease) .25s both; }}
.tt-tk:nth-child(2) {{ animation-delay: .33s; }}
.tt-tk:nth-child(3) {{ animation-delay: .41s; }}
.tt-tk:nth-child(4) {{ animation-delay: .49s; }}
[data-stale="true"], [data-stale="true"] * {{ opacity: 1 !important; transition: none !important; }}
button, input, textarea, [data-baseweb="select"] > div, [data-baseweb="tab"], a {{
  transition: background-color .2s var(--ease), border-color .2s var(--ease), box-shadow .2s var(--ease), color .2s var(--ease); }}
.tt-await {{ border: 1.5px dashed {RULE}; border-radius: 6px; padding: 1.1rem; margin-bottom: .75rem; color: {MUTED};
  font-family: {SERIF}; font-size: 1.5rem; }}
.tt-dot {{ width: .65rem; height: .65rem; border-radius: 50%; flex: none; }}
.tt-meter {{ display: flex; height: 8px; background: {READONLY}; border-radius: 2px; overflow: hidden; margin-top: .3rem; }}
.tt-meter span {{ display: block; height: 100%; }}
.tt-split {{ display: flex; flex-wrap: wrap; gap: .1rem .9rem; font-size: .8125rem; color: {MUTED}; margin-top: .3rem;
  font-variant-numeric: tabular-nums; }}
.tt-split b {{ font-weight: 650; }}

/* The paper trail: work notes clipped along a perforated line */
.tt-trailhead {{ font-family: {SERIF}; font-weight: 300; font-size: 2rem; line-height: 1.1; color: {INK}; text-align: center; margin: 3rem 0 .2rem; }}
.tt-trailsub {{ text-align: center; color: {MUTED}; font-size: .9rem; margin-bottom: 1.5rem; }}
.tt-trail {{ position: relative; margin: 1rem 0 0 .45rem; padding-left: 1.6rem; border-left: 2px dashed {RULE}; }}
.tt-note {{ position: relative; background: #FFFDF8; border: 1px solid {RULE}; border-radius: 4px; padding: .6rem .9rem; margin-bottom: .7rem;
  animation: tt-rise .5s var(--ease) .75s both; }}
.tt-note + .tt-note {{ animation-delay: .9s; }}
.tt-note::before {{ content: ""; position: absolute; left: calc(-1.6rem - 8px); top: .85rem; width: 12px; height: 12px; border-radius: 50%;
  background: {PAGE}; border: 3px solid {art.RED}; }}
.tt-note .who {{ font-size: .8125rem; color: {MUTED}; margin-bottom: .1rem; }}
.tt-note .who b {{ color: {INK}; font-weight: 600; }}
.tt-note .t {{ font-size: .95rem; color: {INK}; line-height: 1.45; }}

/* Evidence tickets: why each prediction came out the way it did */
.tt-evis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(19rem, 1fr)); gap: 1.25rem; margin-top: 1.25rem; }}
.tt-evi {{ position: relative; background: #FFFDF8; border: 1px solid {RULE}; border-radius: 8px; padding: 1rem 1.25rem 1.1rem;
  box-shadow: 0 16px 30px -24px rgba(18, 48, 59, .5); animation: tt-rise .55s var(--ease) 1s both; }}
.tt-evi + .tt-evi {{ animation-delay: 1.12s; }}
.tt-evi .top {{ display: flex; justify-content: space-between; align-items: center; gap: .75rem; min-height: 3.4rem; }}
.tt-evi .kind {{ font-size: .8125rem; color: {MUTED}; }}
.tt-evi .res {{ font-family: {SERIF}; font-size: 1.6rem; line-height: 1.15; color: {INK}; }}
.tt-evi .mini {{ width: 6.6rem; transform: rotate(-6deg); flex: none; }}
.tt-evi .cut {{ position: relative; border-top: 2px dashed {RULE}; margin: .7rem -1.25rem .85rem; }}
.tt-evi .cut::before, .tt-evi .cut::after {{ content: ""; position: absolute; top: -11px; width: 20px; height: 20px; border-radius: 50%;
  background: {PAGE}; border: 1px solid {RULE}; }}
.tt-evi .cut::before {{ left: -11px; }}
.tt-evi .cut::after {{ right: -11px; }}
.tt-fr {{ display: grid; grid-template-columns: minmax(0, 9.5rem) minmax(0, 1fr) 3rem; align-items: center; gap: .6rem; padding: .28rem 0; }}
.tt-fr .n {{ font-size: .875rem; color: {INK}; line-height: 1.25; }}
.tt-fr .x {{ font-family: {MONO}; font-size: .78rem; color: {MUTED}; text-align: right; }}
.tt-fr .tr {{ position: relative; height: 10px; background: {READONLY}; border-radius: 2px; }}
.tt-fr .tr::before {{ content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: {MUTED}; z-index: 1; }}
.tt-fr .b {{ position: absolute; top: 0; height: 100%; animation: tt-grow .8s var(--ease) var(--d) both; }}
.tt-fr .b.up {{ left: 50%; background: {art.RED}; border-radius: 0 2px 2px 0; }}
.tt-fr .b.down {{ right: 50%; background: {ACCENT}; border-radius: 2px 0 0 2px; }}
.tt-key {{ display: flex; gap: 1rem; font-size: .8125rem; color: {MUTED}; margin-top: .45rem; }}
.tt-key i {{ display: inline-block; width: .65rem; height: .65rem; border-radius: 2px; margin-right: .3rem; }}
.tt-words {{ margin-top: .8rem; }}
.tt-chips {{ display: flex; flex-wrap: wrap; gap: .45rem .5rem; margin-top: .35rem; }}
.tt-chip {{ border: 2px solid currentColor; border-radius: 3px; padding: 0 .4rem; font-size: .8125rem; font-weight: 700; letter-spacing: .02em;
  text-transform: uppercase; background: transparent; transform: rotate(var(--t)); }}
.tt-chip small {{ font-family: {MONO}; font-weight: 500; font-size: .72rem; margin-left: .3rem; text-transform: none; }}

.tt-quiet, .tt-quiet * {{ animation: none !important; }}
.tt-stale {{ opacity: .45; filter: grayscale(.5); transition: opacity .3s var(--ease); }}
.tt-stalenote {{ color: {MUTED}; font-size: .85rem; font-style: italic; text-align: center; margin: 0 0 .5rem; }}
.st-key-samples {{ margin-bottom: .4rem; }}
.st-key-samples button {{ min-height: 2rem; padding: .2rem .8rem; font-size: .85rem; }}
.tt-empty {{ color: {MUTED}; font-size: .9rem; padding: .3rem 0; }}
.tt-h3 {{ font-size: .95rem; font-weight: 650; color: {INK}; margin: .6rem 0 .1rem; }}
.tt-muted {{ color: {MUTED}; font-size: .875rem; }}
.tt-small {{ font-size: .8125rem; font-weight: 600; color: {MUTED}; }}

.tt-stats {{ display: flex; flex-wrap: wrap; gap: 1rem 2.5rem; margin: .5rem 0; }}
.tt-stat .v {{ font-family: {SERIF}; font-size: 2.2rem; font-weight: 300; line-height: 1.1; color: {INK}; }}
.tt-stat .k {{ color: {INK}; font-size: .875rem; }}
.tt-stat .s {{ color: {MUTED}; font-size: .8125rem; }}

.tt-band {{ text-align: center; margin: 5rem 0 0; padding: 3rem 1rem 4rem; border-top: 1px solid {RULE}; }}
.tt-band h2 {{ font-family: {SERIF}; font-weight: 300; font-size: clamp(2.2rem, 4.2vw, 3.4rem); line-height: 1.08; margin: 0 0 1rem; padding: 0; color: {INK}; }}
.tt-band p {{ max-width: 38rem; margin: 0 auto .8rem; font-size: 1.05rem; line-height: 1.55; color: {MUTED}; }}

/* One big ticket: the form is the ticket, the right column is its tear-off stub */
[data-testid="stForm"] {{ background: #FFFDF8; border: 1px solid {RULE}; border-radius: 10px; padding: 1.25rem 1.5rem; overflow: visible; box-shadow: 0 20px 40px -28px rgba(18, 48, 59, .45); }}
[data-testid="stColumn"]:has(.tt-stub) {{ position: relative; border-left: 2px dashed {RULE}; padding-left: 2rem; }}
[data-testid="stColumn"]:has(.tt-stub)::before, [data-testid="stColumn"]:has(.tt-stub)::after {{ content: ""; position: absolute;
  left: -12px; width: 22px; height: 22px; border-radius: 50%; background: {PAGE}; border: 1px solid {RULE}; }}
[data-testid="stColumn"]:has(.tt-stub)::before {{ top: calc(-1.25rem - 12px); }}
[data-testid="stColumn"]:has(.tt-stub)::after {{ bottom: calc(-1.25rem - 12px); }}
[data-testid="stFormSubmitButton"] button:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 2px; }}
[data-baseweb="tab"] {{ font-weight: 600; }}
@media (max-width: 760px) {{
  .block-container {{ padding: 0 1rem 2rem; }}
  [data-testid="stForm"] {{ padding: 1rem; }}
  .tt-hero {{ height: calc(100vh + 1.5rem); min-height: 560px; }}
  .tt-tk {{ left: -5.5rem; width: 11rem; }}
  .tt-tk:nth-child(2), .tt-tk:nth-child(3) {{ display: none; }}
  .tt-tk .p {{ font-size: 1.3rem; }}
  .tt-s {{ scale: .5; }}
  [data-testid="stColumn"]:has(.tt-stub) {{ border-left: 0; border-top: 2px dashed {RULE}; padding: 1rem 0 0; }}
  [data-testid="stColumn"]:has(.tt-stub)::before, [data-testid="stColumn"]:has(.tt-stub)::after {{ display: none; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; scroll-behavior: auto !important; }} }}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def section(text):
    html(f'<div class="tt-sec">{escape(text)}</div>')


def field(name, value_html="", extra=""):
    state = "set" if value_html else "empty"
    return (f'<div class="tt-f"><div class="l">{escape(name)}</div>'
            f'<div class="v {state}">{value_html or "Not triaged"}</div>{extra}</div>')


def verdict(priority, confidence):
    return (f'<div class="tt-verdict" style="--p:{PRIORITY[priority]}"><div class="c">{escape(confidence)}</div>'
            f'<div class="p">{PRIORITY_CODE[priority]}</div></div>')


def slam(priority):
    """The animated stamp; .tt-landed in the same stub also jolts the ticket on impact."""
    src = art.data_uri(art.stamp(priority.upper(), seed=5, color=STAMP[priority]))
    return f'<img class="tt-slam" alt="Stamped {escape(priority)}" src="{src}">'


def hero():
    stack = [("Low", -1.5, "-6deg", "1.5rem", "How do I export my data?"), ("Medium", -.5, "-2deg", "0", "Charged twice this month"),
             ("High", .5, "2deg", "0", "Locked out of my account"), ("Critical", 1.5, "6deg", "1.5rem", "Production dashboard is down")]
    tickets = "".join(
        f'<div class="tt-tk" style="--p:{PRIORITY[p]};--i:{x};--r:{r};--y:{y}"><div class="id">INC000{482 + i}</div>'
        f'<div class="p">{PRIORITY_CODE[p]}</div><div class="d">{escape(text)}</div></div>'
        for i, (p, x, r, y, text) in enumerate(stack))
    # Offsets are from the page center, so stamps stay clear of the headline at any width.
    stamps = [("P1", "150px", "-37rem", "8%", "-12deg", "top", "0%"), ("CRITICAL", "250px", "23rem", "6%", "9deg", "top", "0%"),
              ("URGENT", "230px", "-41rem", "37%", "7deg", "", "3%"), ("ASAP", "170px", "27rem", "33%", "-8deg", "", "76%"),
              ("P2", "120px", "-22rem", "56%", "10deg", "", "27%"), ("HIGH", "150px", "14rem", "54%", "-6deg", "", "52%")]
    marks = "".join(f'<img class="tt-s {cls}" alt="" src="{art.data_uri(art.stamp(text, seed=i + 2))}" '
                    f'style="--w:{w};--x:{x};--t:{t};--r:{r};--n:{n}">'
                    for i, (text, w, x, t, r, cls, n) in enumerate(stamps))
    return (f'<div class="tt-hero">{marks}<h1>Every ticket says it\'s urgent.<br><em>Find out which one is.</em></h1>'
            '<div class="lede">Paste an incident. Get its priority, an expected resolution time, and the reasons behind both.</div>'
            f'<a class="tt-cta" href="#open-a-ticket" target="_self">Triage a ticket</a><div class="tt-fan">{tickets}</div></div>')


def note(who, when, text):
    return (f'<div class="tt-note"><div class="who"><b>{escape(who)}</b> · Work note · {escape(when)}</div>'
            f'<div class="t">{escape(text)}</div></div>')


def trail(notes):
    return f'<div class="tt-trail">{"".join(notes)}</div>'


def _factors(rows, up, down):
    peak = max(abs(value) for _, value in rows) or 1
    html_rows = []
    for i, (name, value) in enumerate(rows):
        side = "up" if value > 0 else "down"
        width = abs(value) / peak * 50
        html_rows.append(f'<div class="tt-fr"><div class="n">{escape(name)}</div><div class="tr">'
                         f'<span class="b {side}" style="width:{width:.1f}%;--d:{1.15 + i * .06:.2f}s" '
                         f'title="{escape(up if value > 0 else down)}"></span></div><div class="x">{value:+.2f}</div></div>')
    key = (f'<div class="tt-key"><span><i style="background:{art.RED}"></i>{escape(up)}</span>'
           f'<span><i style="background:{ACCENT}"></i>{escape(down)}</span></div>')
    return "".join(html_rows) + key


def _chips(terms):
    if not terms:
        return ""
    tilts = ("-2deg", "1.5deg", "-1deg", "2deg", "-1.5deg")
    chips = "".join(
        f'<span class="tt-chip" style="color:{"#B3261E" if t["weight"] > 0 else ACCENT};--t:{tilts[i % 5]}">'
        f'{escape(t["term"])}<small>{t["weight"]:+.2f}</small></span>' for i, t in enumerate(terms))
    return f'<div class="tt-words"><div class="tt-small">Words behind the text signal</div><div class="tt-chips">{chips}</div></div>'


def evidence(kind, result, stamp_text, stamp_color, rows, terms, up, down):
    mini = art.data_uri(art.stamp(stamp_text, seed=4, color=stamp_color))
    return (f'<div class="tt-evi"><div class="top"><div><div class="kind">{escape(kind)}</div>'
            f'<div class="res">{escape(result)}</div></div><img class="mini" alt="" src="{mini}"></div><div class="cut"></div>'
            f'{_factors(rows, up, down)}{_chips(terms)}</div>')


def split(probabilities):
    """Low→Critical stacked bar. Plain HTML; a chart is overkill for four numbers."""
    order = list(PRIORITY)
    segs = "".join(f'<span style="width:{probabilities[p] * 100:.2f}%;background:{PRIORITY[p]}" '
                   f'title="{p} {probabilities[p]:.1%}"></span>' for p in order)
    legend = "".join(f'<span><b style="color:{PRIORITY[p]}">{p}</b> {probabilities[p]:.0%}</span>' for p in order)
    aria = "Priority probabilities: " + ", ".join(f"{p} {probabilities[p]:.0%}" for p in order)
    return f'<div class="tt-meter" role="img" aria-label="{aria}">{segs}</div><div class="tt-split">{legend}</div>'


def sla_meter(hours, target, breach):
    width = min(hours / target, 1) * 100
    color = WARN if breach else OK
    return f'<div class="tt-meter"><span style="width:{width:.1f}%;background:{color}"></span></div>'


def stats(items):
    cells = "".join(f'<div class="tt-stat"><div class="v">{escape(v)}</div><div class="k">{escape(k)}</div>'
                    f'<div class="s">{escape(s)}</div></div>' for k, v, s in items)
    return f'<div class="tt-stats">{cells}</div>'


def style_chart(chart):
    """Shared Altair styling: no frames, quiet axes, the UI's type colors."""
    return (chart.configure(background="transparent").configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#E1E6EB", domainColor=RULE,
                            tickColor=RULE, labelFontSize=12, titleFontSize=12, titleFontWeight=500)
            .configure_legend(labelColor=INK, labelFontSize=12, symbolType="square", orient="bottom")
            .configure_axisY(labelLimit=340))
