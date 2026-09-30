"""
MalXplain dashboard theme.

Navy and copper on a warm off-white surface.

Navy is used for interface structure only, never as a chart series colour: it sits
outside the validated lightness band and below the chroma floor, so as a data mark it
would read as grey. Chart series use the blue and copper pair, which passes all six
accessibility checks including colour vision deficiency separation.
"""

SURFACE = "#FBFAF8"   # page background, warm off-white
PANEL   = "#FFFFFF"   # raised card surface
INK     = "#16202B"   # primary text
NAVY    = "#143A5C"   # structure, headings, sidebar
COPPER  = "#C8622B"   # accent, key figures
BLUE    = "#2F6FAD"   # chart series, links
MUTED   = "#6B7280"   # secondary text
BORDER  = "#E4E2DD"   # warm hairline
WASH    = "#F4F2EE"   # subtle fill for inset blocks

CSS = f"""
<style>
:root {{
  --surface:{SURFACE}; --panel:{PANEL}; --ink:{INK}; --navy:{NAVY};
  --copper:{COPPER}; --blue:{BLUE}; --muted:{MUTED}; --border:{BORDER}; --wash:{WASH};
}}

/* ---------- base ---------- */
.stApp {{ background:var(--surface); }}
html, body, [class*="css"] {{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:var(--ink);
}}
.block-container {{ padding-top:2.2rem; padding-bottom:4rem; max-width:1280px; }}

/* hide the default streamlit chrome so it reads as a site, not a notebook */
#MainMenu, footer, header {{ visibility:hidden; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
  background:var(--navy);
  border-right:1px solid rgba(0,0,0,.18);
}}
section[data-testid="stSidebar"] * {{ color:#E8EDF2 !important; }}
section[data-testid="stSidebar"] .sidebar-brand {{
  font-size:1.35rem; font-weight:700; letter-spacing:-.02em; color:#FFFFFF !important;
  margin-bottom:.1rem;
}}
section[data-testid="stSidebar"] .sidebar-sub {{
  font-size:.74rem; color:#9FB3C8 !important; letter-spacing:.06em;
  text-transform:uppercase; margin-bottom:1.4rem;
}}
section[data-testid="stSidebar"] .sidebar-note {{
  font-size:.72rem; line-height:1.55; color:#A9BCCF !important;
}}
section[data-testid="stSidebar"] hr {{ border-color:rgba(255,255,255,.14); }}

/* nav radio as a list of rows */
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
  padding:.44rem .6rem; border-radius:5px; margin-bottom:1px;
  transition:background .12s ease;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
  background:rgba(255,255,255,.07);
}}

/* ---------- masthead ---------- */
.masthead {{
  border-bottom:2px solid var(--navy);
  padding-bottom:1.1rem; margin-bottom:.4rem;
}}
.masthead .eyebrow {{
  font-size:.7rem; letter-spacing:.16em; text-transform:uppercase;
  color:var(--copper); font-weight:700; margin-bottom:.5rem;
}}
.masthead h1 {{
  font-size:2.5rem; font-weight:760; letter-spacing:-.035em;
  color:var(--navy); margin:0 0 .45rem 0; line-height:1.06;
}}
.masthead .lede {{
  font-size:1rem; color:var(--muted); line-height:1.55; max-width:62rem; margin:0;
}}
.masthead .byline {{
  font-size:.78rem; color:var(--muted); margin-top:.7rem;
}}

/* ---------- section headers ---------- */
.sec {{ margin:2.3rem 0 1rem 0; }}
.sec .kicker {{
  font-size:.68rem; letter-spacing:.15em; text-transform:uppercase;
  color:var(--copper); font-weight:700; margin-bottom:.3rem;
}}
.sec h2 {{
  font-size:1.35rem; font-weight:700; letter-spacing:-.018em;
  color:var(--navy); margin:0 0 .35rem 0;
}}
.sec p {{ font-size:.9rem; color:var(--muted); line-height:1.6; margin:0; max-width:64rem; }}

/* ---------- cards ---------- */
.card {{
  background:var(--panel); border:1px solid var(--border); border-radius:7px;
  padding:1.15rem 1.3rem; box-shadow:0 1px 2px rgba(20,58,92,.045);
}}
.card + .card {{ margin-top:.75rem; }}

/* stat tiles */
.tiles {{ display:flex; gap:.7rem; flex-wrap:wrap; margin:.2rem 0 .3rem 0; }}
.tile {{
  flex:1 1 0; min-width:168px; background:var(--panel);
  border:1px solid var(--border); border-top:3px solid var(--navy);
  border-radius:6px; padding:.9rem 1rem;
}}
.tile.accent {{ border-top-color:var(--copper); }}
.tile .lbl {{
  font-size:.68rem; letter-spacing:.09em; text-transform:uppercase;
  color:var(--muted); font-weight:650; margin-bottom:.35rem;
}}
.tile .val {{
  font-size:1.72rem; font-weight:720; letter-spacing:-.03em;
  color:var(--navy); line-height:1.05;
}}
.tile.accent .val {{ color:var(--copper); }}
.tile .sub {{ font-size:.72rem; color:var(--muted); margin-top:.28rem; line-height:1.4; }}

/* callouts */
.note {{
  border-left:3px solid var(--blue); background:var(--wash);
  padding:.8rem 1rem; border-radius:0 5px 5px 0; font-size:.88rem;
  line-height:1.6; color:var(--ink); margin:.6rem 0;
}}
.note.flag {{ border-left-color:var(--copper); }}
.note.key  {{ border-left-color:var(--navy); }}
.note strong {{ color:var(--navy); }}

/* finding block on the live page */
.finding {{
  background:var(--panel); border:1px solid var(--border);
  border-left:3px solid var(--copper); border-radius:0 6px 6px 0;
  padding:1.05rem 1.25rem; font-size:.95rem; line-height:1.7; color:var(--ink);
}}

/* ---------- streamlit widget polish ---------- */
div[data-testid="stMetric"] {{
  background:var(--panel); border:1px solid var(--border);
  border-radius:6px; padding:.75rem .9rem;
}}
div[data-testid="stMetricLabel"] p {{
  font-size:.7rem !important; letter-spacing:.07em; text-transform:uppercase;
  color:var(--muted) !important; font-weight:650;
}}
div[data-testid="stMetricValue"] {{
  font-size:1.5rem !important; color:var(--navy) !important;
  font-weight:700; letter-spacing:-.02em;
}}

.stButton > button {{
  background:var(--copper); color:#fff; border:none; border-radius:5px;
  font-weight:620; letter-spacing:.01em; padding:.5rem 1.1rem;
  box-shadow:0 1px 2px rgba(200,98,43,.28); transition:background .13s ease;
}}
.stButton > button:hover {{ background:#B0551F; color:#fff; }}

.stDataFrame {{ border:1px solid var(--border); border-radius:6px; }}

div[data-testid="stExpander"] {{
  border:1px solid var(--border); border-radius:6px; background:var(--panel);
}}

.stTabs [data-baseweb="tab-list"] {{ gap:1.4rem; border-bottom:1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{
  font-size:.86rem; font-weight:600; color:var(--muted); padding:.5rem 0;
}}
.stTabs [aria-selected="true"] {{ color:var(--navy) !important; }}

hr {{ border-color:var(--border); margin:1.6rem 0; }}

/* footer strip */
.foot {{
  margin-top:3rem; padding-top:1rem; border-top:1px solid var(--border);
  font-size:.74rem; color:var(--muted); line-height:1.6;
}}
</style>
"""


def masthead(title: str, lede: str, byline: str, eyebrow: str = "") -> str:
    kick = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    return (f'<div class="masthead">{kick}<h1>{title}</h1>'
            f'<p class="lede">{lede}</p><div class="byline">{byline}</div></div>')


def section(title: str, body: str = "", kicker: str = "") -> str:
    k = f'<div class="kicker">{kicker}</div>' if kicker else ""
    p = f"<p>{body}</p>" if body else ""
    return f'<div class="sec">{k}<h2>{title}</h2>{p}</div>'


def tiles(items) -> str:
    """items: list of (label, value, subtext, accent_bool)"""
    out = []
    for lbl, val, sub, acc in items:
        cls = "tile accent" if acc else "tile"
        s = f'<div class="sub">{sub}</div>' if sub else ""
        out.append(f'<div class="{cls}"><div class="lbl">{lbl}</div>'
                   f'<div class="val">{val}</div>{s}</div>')
    return f'<div class="tiles">{"".join(out)}</div>'


def note(text: str, kind: str = "") -> str:
    cls = f"note {kind}".strip()
    return f'<div class="{cls}">{text}</div>'
