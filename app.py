"""HarvestIQ — single-page campaign command center (Track 1: Marketing).

Pick a grower → product, channel, send date, weather, vernacular copy
all auto-derive. Includes multi-variant generation (urgent / educational
/ social-proof) for the 'scale personalization' requirement.

Run with:  streamlit run app.py
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.content import generate_variants
from src.context import build_context
from src.data import load_all
from src.targeting import train

st.set_page_config(
    page_title="HarvestIQ — Marketing",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Theme / custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1410 0%, #0F1B16 100%);
    border-right: 1px solid #1F2A24;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stSlider label {
    color: #6EE7B7 !important; font-weight: 600; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.05em;
}

/* KPI strip */
.kpi {
    background: #141A18; border-radius: 14px; padding: 18px 22px;
    border: 1px solid #1F2A24;
}
.kpi-label { color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase;
             letter-spacing: 0.05em; font-weight: 600; }
.kpi-value { color: #4ADE80; font-size: 1.9rem; font-weight: 700;
             margin-top: 4px; line-height: 1.1; }
.kpi-sub   { color: #9CA3AF; font-size: 0.8rem; margin-top: 2px; }

/* Plan cards */
.card {
    background: #141A18; border-radius: 14px; padding: 22px 24px;
    border: 1px solid #1F2A24; height: 100%;
}
.card h4 { margin: 0 0 12px 0; font-size: 0.85rem; color: #9CA3AF;
           text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700; }
.card .big { font-size: 1.6rem; font-weight: 700; color: #4ADE80;
             margin-bottom: 6px; }
.card ul { padding-left: 18px; margin: 8px 0; color: #D1D5DB; }
.card li { margin-bottom: 4px; }

/* Pill / badge */
.pill { display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
        margin-right: 4px; }
.pill-green   { background: rgba(74,222,128,0.15); color: #4ADE80; }
.pill-amber   { background: rgba(251,191,36,0.15); color: #FBBF24; }
.pill-red     { background: rgba(248,113,113,0.15); color: #F87171; }
.pill-blue    { background: rgba(96,165,250,0.15); color: #60A5FA; }
.pill-grey    { background: rgba(156,163,175,0.15); color: #D1D5DB; }

/* Variant tab cards */
.variant-card {
    background: #141A18; border: 1px solid #1F2A24; border-left: 4px solid #4ADE80;
    border-radius: 10px; padding: 18px 20px; margin-bottom: 12px;
}
.variant-card .headline { font-size: 1.15rem; font-weight: 700; color: #4ADE80;
                          margin-bottom: 8px; }
.variant-card .body { color: #E5E7EB; line-height: 1.55; margin-bottom: 10px; }
.variant-card .cta  { color: #6EE7B7; font-weight: 600; }

/* Brand bar */
.brand {
    background: linear-gradient(90deg, #0F1B16 0%, #1B4332 100%);
    color: #E5E7EB; padding: 14px 22px; border-radius: 14px;
    margin-bottom: 18px; display: flex; align-items: center;
    justify-content: space-between; border: 1px solid #1F2A24;
}
.brand .title { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.01em;
                color: #4ADE80; }
.brand .sub { font-size: 0.85rem; color: #9CA3AF; }
</style>
""", unsafe_allow_html=True)


# ── Load data + train model once ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading data and training targeting model…")
def _bundle():
    d = load_all()
    g = d.grower_features()
    m = train(d)
    g["click_score"] = m.score(g)
    return d, g, m


data, growers, model = _bundle()


# ── Brand bar ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="brand">
  <div>
    <div class="title">🌾 HarvestIQ</div>
    <div class="sub">AI-powered marketing for Syngenta India · Track 1</div>
  </div>
  <div class="sub">Rabi 2025–26 · {len(growers):,} growers · model AUC {model.auc:.3f}</div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar filters ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    state = st.selectbox("State", ["All"] + sorted(growers["state"].unique()))
    crop = st.selectbox("Crop", ["All"] + sorted(growers["crop"].unique()))
    device = st.selectbox("Device", ["All", "smartphone", "keypad", "unknown"])
    language = st.selectbox("Language", ["All"] + sorted(growers["language"].unique()))
    as_of = st.date_input("As-of date", value=date(2026, 2, 15))
    top_n = st.slider("Show top-N", 10, 200, 50, step=10)
    st.markdown("---")
    st.markdown("**Targeting**")
    st.caption(f"Logistic regression on WhatsApp click history. AUC = **{model.auc:.3f}**.")

pool = growers.copy()
if state != "All":    pool = pool[pool["state"] == state]
if crop != "All":     pool = pool[pool["crop"] == crop]
if device != "All":   pool = pool[pool["device_type"] == device]
if language != "All": pool = pool[pool["language"] == language]
pool = pool.sort_values("click_score", ascending=False)


# ── KPI strip ─────────────────────────────────────────────────────────
def _kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


k1, k2, k3, k4 = st.columns(4)
_kpi(k1, "Matching growers", f"{len(pool):,}", f"of {len(growers):,} total")
_kpi(k2, "States covered", f"{pool['state'].nunique()}", f"crops: {pool['crop'].nunique()}")
_kpi(k3, "Avg click score", f"{pool['click_score'].mean():.3f}" if len(pool) else "—",
     "predicted receptivity")
_kpi(k4, "Smartphone share",
     f"{(pool['device_type'] == 'smartphone').mean()*100:.0f}%" if len(pool) else "—",
     "drives channel mix")

st.write("")  # spacer

# ── Grower picker table ───────────────────────────────────────────────
left, right = st.columns([3, 2])
with left:
    st.markdown("##### Ranked growers")
    if not len(pool):
        st.warning("No growers match these filters.")
        st.stop()
    show = pool[["grower_id", "district", "crop", "current_stage",
                 "language", "device_type", "grower_farm_size", "click_score"]].head(top_n).reset_index(drop=True)
    show.columns = ["Grower", "District", "Crop", "Stage", "Language",
                    "Device", "Farm (ac)", "Click score"]
    st.dataframe(
        show.style.background_gradient(subset=["Click score"], cmap="Greens")
                  .format({"Farm (ac)": "{:.2f}", "Click score": "{:.3f}"}),
        use_container_width=True, height=380, hide_index=True,
    )

with right:
    st.markdown("##### Select grower")
    grower_id = st.selectbox(
        "Grower ID", show["Grower"].tolist(), index=0,
        label_visibility="collapsed",
    )
    chosen = pool[pool["grower_id"] == grower_id].iloc[0]
    st.markdown(
        f"<div class='pill pill-green'>{chosen['crop']}</div> "
        f"<div class='pill pill-blue'>{chosen['current_stage']}</div> "
        f"<div class='pill pill-grey'>{chosen['language']}</div> "
        f"<div class='pill pill-grey'>{chosen['device_type']}</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.caption(f"📍 {chosen['district']}, {chosen['state']} · {chosen['tehsil']}")
    st.caption(f"👤 {int(chosen['grower_age'])} yrs · {chosen['grower_farm_size']:.2f} acres")
    st.caption(f"🎯 click score **{chosen['click_score']:.3f}** "
               f"(top {(pool['click_score'] > chosen['click_score']).mean()*100:.1f}% of filtered)")


# ── Build the plan ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching weather + generating campaign plan…")
def _ctx(grower_id: str, as_of_iso: str):
    row = growers[growers["grower_id"] == grower_id].iloc[0]
    return build_context(data, row, generate_content=True,
                         as_of=date.fromisoformat(as_of_iso))


ctx = _ctx(grower_id, as_of.isoformat())

st.write("")
st.markdown(f"### Campaign plan · {grower_id}")

a, b, c = st.columns(3)

with a:
    rec = ctx.recommendation
    st.markdown(f"""
<div class="card">
  <h4>💊 Recommended product</h4>
  <div class="big">{rec.product}</div>
  <div class="pill pill-green">score {rec.score}</div>
  <div class="pill pill-grey">{rec.weather_modifier}</div>
  <ul>{''.join(f'<li>{r}</li>' for r in rec.reasons)}</ul>
</div>
""", unsafe_allow_html=True)

with b:
    ch = ctx.channel
    st.markdown(f"""
<div class="card">
  <h4>📡 Channel &amp; timing</h4>
  <div class="big">{ch.primary_channel}</div>
  <div class="pill pill-blue">send {ctx.send_date or '—'}</div>
  <div class="pill pill-grey">backup: {ch.secondary_channel}</div>
  <ul>
    <li><b>Format:</b> {ch.format}</li>
    <li><b>Tone:</b> {ctx.tone}</li>
    <li><i>{ch.rationale}</i></li>
  </ul>
</div>
""", unsafe_allow_html=True)

with c:
    w = ctx.weather
    pressure = w.disease_pressure()
    spray = w.spray_window()
    pp = {"high": "red", "moderate": "amber", "low": "green"}[pressure]
    sp = {"good": "green", "marginal": "amber", "poor": "red"}[spray]
    st.markdown(f"""
<div class="card">
  <h4>🌤️ Live weather · {w.district}</h4>
  <div class="big">{w.temp_c:.0f}°C · {w.humidity_pct:.0f}% RH</div>
  <div class="pill pill-{pp}">{pressure} disease pressure</div>
  <div class="pill pill-{sp}">{spray} spray window</div>
  <ul>
    <li>Next 7 days: {w.rain_forecast_7d_mm:.0f} mm rain, {w.rainy_days_next_7} rainy days</li>
    <li>Source: <code>{w.source}</code> (Open-Meteo)</li>
  </ul>
</div>
""", unsafe_allow_html=True)


# ── Generated copy + variants ─────────────────────────────────────────
st.write("")
st.markdown("### Generated copy")

if ctx.content is None:
    st.warning("Content generation failed.")
else:
    c0 = ctx.content
    badge = {"gemini": "pill-green", "ollama": "pill-amber", "template": "pill-grey"}.get(c0.provider, "pill-grey")
    st.markdown(
        f"<div class='pill {badge}'>provider: {c0.provider}</div>"
        f" <span style='color:#6B7280;font-size:0.85rem;'>{c0.notes}</span>",
        unsafe_allow_html=True,
    )

    tab_default, tab_variants, tab_export = st.tabs([
        "📨 Default copy", "🎨 Variants (urgent · educational · social-proof)",
        "📥 Bulk export",
    ])

    with tab_default:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"""
<div class="variant-card">
  <div class="headline">{c0.headline}</div>
  <div class="body">{c0.body}</div>
  <div class="cta">→ {c0.cta}</div>
</div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**SMS fallback** _(≤160 chars · {len(c0.sms_fallback)} used)_")
            st.code(c0.sms_fallback, language="text")
            st.markdown("**IVR voice script**")
            st.code(c0.voice_script, language="text")

    with tab_variants:
        st.caption(
            "Three creative angles for the same farmer. Use to A/B test "
            "or rotate fatigue across a multi-touch campaign."
        )
        if st.button("✨ Generate 3 variants", type="primary"):
            with st.spinner("Generating…"):
                variants = generate_variants(ctx.to_brief())
            st.session_state["variants"] = variants

        variants = st.session_state.get("variants")
        if variants:
            cols = st.columns(len(variants))
            for col, v in zip(cols, variants):
                with col:
                    st.markdown(f"<div class='pill pill-blue'>{v.variant}</div>",
                                unsafe_allow_html=True)
                    st.markdown(f"""
<div class="variant-card">
  <div class="headline">{v.headline}</div>
  <div class="body">{v.body}</div>
  <div class="cta">→ {v.cta}</div>
</div>""", unsafe_allow_html=True)
                    with st.expander("SMS + voice"):
                        st.code(v.sms_fallback, language="text")
                        st.code(v.voice_script, language="text")
        else:
            st.info("Click **Generate 3 variants** to produce micro-targeted creative angles.")

    with tab_export:
        st.caption("Build full campaign plans for the top-K filtered growers and download as CSV.")
        n = st.slider("How many growers?", 10, min(500, len(pool)), 50, step=10)
        if st.button("Build plans"):
            rows = []
            prog = st.progress(0.0)
            for i, (_, r) in enumerate(pool.head(n).iterrows()):
                cx = build_context(data, r, generate_content=False, as_of=as_of)
                rows.append({
                    "grower_id": r["grower_id"], "state": r["state"],
                    "district": r["district"], "crop": r["crop"],
                    "stage": r["current_stage"], "language": r["language"],
                    "device": r["device_type"],
                    "click_score": round(r["click_score"], 3),
                    "product": cx.recommendation.product,
                    "reason": "; ".join(cx.recommendation.reasons),
                    "channel": cx.channel.primary_channel,
                    "send_date": cx.send_date,
                    "weather": cx.weather.summary,
                })
                prog.progress((i + 1) / n)
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True, height=320)
            st.download_button(
                "⬇️ Download CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="harvestiq_plans.csv", mime="text/csv",
            )
