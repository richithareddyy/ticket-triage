"""Rubber-stamp art as SVG data URIs, so the UI needs no image assets."""
from base64 import b64encode

RED = "#E2231A"


def stamp(text, width=None, seed=2, color=RED):
    """A worn, double-ruled rubber stamp. Sized from the text so every label fits its box."""
    width = width or 56 + 27 * len(text)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 84">
<defs><filter id="ink" x="-5%" y="-5%" width="110%" height="110%">
  <feTurbulence type="fractalNoise" baseFrequency=".7" numOctaves="3" seed="{seed}" result="n"/>
  <feColorMatrix in="n" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 -6 4.6" result="wear"/>
  <feComposite in="SourceGraphic" in2="wear" operator="in" result="worn"/>
  <feTurbulence type="fractalNoise" baseFrequency=".04" numOctaves="2" seed="{seed}" result="r"/>
  <feDisplacementMap in="worn" in2="r" scale="3"/>
</filter></defs>
<g filter="url(#ink)" fill="none" stroke="{color}">
  <rect x="4" y="4" width="{width - 8}" height="76" rx="6" stroke-width="5"/>
  <rect x="12" y="12" width="{width - 24}" height="60" rx="3" stroke-width="2"/>
  <text x="{width / 2}" y="56" text-anchor="middle" fill="{color}" stroke="none" font-size="40" font-weight="900"
   font-family="Impact, 'Arial Black', 'Helvetica Neue', Arial, sans-serif" letter-spacing="2">{text}</text>
</g></svg>"""


def _column(items):
    """A tall strip of stamps for the page margins."""
    height = 1500
    parts = []
    for i, (text, y, angle) in enumerate(items):
        svg = stamp(text, seed=i + 3).split("\n", 1)[1].rsplit("</svg>", 1)[0].replace('id="ink"', f'id="ink{i}"').replace("#ink", f"#ink{i}")
        width = 56 + 27 * len(text)
        scale = min(1, 124 / width)
        parts.append(f'<g transform="translate(70 {y}) rotate({angle}) scale({scale:.2f}) translate({-width / 2} -42)">{svg}</g>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 {height}" width="140" height="{height}">{"".join(parts)}</svg>'


LEFT_MARGIN = _column([("P1", 180, -11), ("URGENT", 560, 8), ("HIGH", 930, -6), ("ASAP", 1290, 10)])
RIGHT_MARGIN = _column([("CRITICAL", 330, 9), ("P2", 720, -10), ("ESCALATED", 1100, 6), ("P1", 1420, -8)])


def data_uri(svg):
    return "data:image/svg+xml;base64," + b64encode(svg.encode()).decode()
