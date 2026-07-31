"""Interactive rolling-vs-CAGR model comparison — embeds the standalone
HTML/SVG/JS scrubber built for LinkedIn as a second-page tab of the main
dashboard. Same file, one source of truth: also served as a public
artifact on claude.ai and copied here so dronespredictions.net hosts
its own copy with no external dependency.
"""
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Model Scrubber — Rolling vs CAGR",
    page_icon="📈",
    layout="wide",
)

APP_DIR = Path(__file__).parent.parent
HTML_PATH = APP_DIR / "static" / "model_scrubber.html"

st.title("📈 Model Scrubber")
st.markdown(
    "Interactive comparison of the two weekly-forecast models running on this "
    "dashboard: **rolling 14-day mean** vs **CAGR-based projection**. "
    "Drag the cursor across the timeline to see both models' predictions and "
    "the realised actual (when past). Second chart shows the MAPE crossover — "
    "where each model wins."
)
st.caption(
    "This is a static interactive built from 11 real backtest weeks + 12 "
    "projected weeks forward. Same file also hosted as a public artifact "
    "(shareable link at the bottom of the page)."
)

if not HTML_PATH.exists():
    st.error(
        f"⚠️ Scrubber HTML missing at `{HTML_PATH}`. "
        "The static asset didn't ship with this build — check the Dockerfile "
        "COPY step or rebuild via `bash deploy/bazzite/install.sh`."
    )
else:
    html = HTML_PATH.read_text(encoding="utf-8")
    # 1600px covers both charts + readout without inner-iframe scroll on most desktops
    components.html(html, height=1600, scrolling=True)

st.markdown("---")
st.markdown(
    "**Share this tool:** copy the standalone HTML from "
    "`drone_app/static/model_scrubber.html` and host anywhere, or use the "
    "published artifact URL for a one-click share. Source lives in "
    "[github.com/connormmc777/ukraine-drone-forecast]"
    "(https://github.com/connormmc777/ukraine-drone-forecast)."
)
