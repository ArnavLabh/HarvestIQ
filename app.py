"""HarvestIQ: single-page campaign command center (Track 1: Marketing).

Two top-level tabs:
  - Campaign Studio: pick a grower, build the plan, generate copy.
  - Dashboard: analytics across the three Track-1 datasets.

Run with:  streamlit run app.py
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from src.channel import predict_best_variant, segment_click_rate
from src.content import generate, generate_image, generate_variants
from src.context import build_context
from src.data import load_all
from src.targeting import train

st.set_page_config(
    page_title="HarvestIQ Marketing",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Theme / custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1380px; }

html, body, [class*="css"] { font-feature-settings: "ss01", "cv11"; }

h1, h2, h3, h4, h5 { letter-spacing: -0.01em; }
h5 { color: #E5E7EB !important; font-weight: 600 !important;
     font-size: 0.95rem !important; margin: 18px 0 10px !important; }
h3 { color: #F3F4F6 !important; font-weight: 700 !important;
     font-size: 1.35rem !important; }

/* Input polish */
div[data-baseweb="select"] > div, .stDateInput input, .stTextInput input {
    background: #0F1614 !important; border: 1px solid #243029 !important;
    border-radius: 10px !important;
}
div[data-baseweb="select"] > div:hover, .stDateInput input:hover { border-color: #2F4A3C !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #1F2A24; }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px 8px 0 0;
    padding: 8px 18px; font-weight: 600; color: #9CA3AF;
}
.stTabs [aria-selected="true"] { color: #4ADE80 !important;
    background: rgba(74,222,128,0.06) !important; }

/* Buttons */
.stButton > button {
    border-radius: 10px; font-weight: 600; border: 1px solid #243029;
    transition: all 0.15s ease;
}
.stButton > button:hover { transform: translateY(-1px); border-color: #4ADE80; }

/* Expanders */
[data-testid="stExpander"] {
    background: #0F1614; border: 1px solid #1F2A24; border-radius: 12px;
}
[data-testid="stExpander"] summary { font-weight: 600; color: #D1D5DB; }

.kpi {
    background: linear-gradient(180deg, #141A18 0%, #11161499 100%);
    border-radius: 14px; padding: 18px 22px;
    border: 1px solid #1F2A24;
    transition: border-color 0.2s ease;
}
.kpi:hover { border-color: #2F4A3C; }
.kpi-label { color: #9CA3AF; font-size: 0.72rem; text-transform: uppercase;
             letter-spacing: 0.06em; font-weight: 700; }
.kpi-value { color: #4ADE80; font-size: 1.85rem; font-weight: 700;
             margin-top: 6px; line-height: 1.1; letter-spacing: -0.02em; }
.kpi-sub   { color: #9CA3AF; font-size: 0.78rem; margin-top: 4px; }

.card {
    background: linear-gradient(180deg, #141A18 0%, #11161499 100%);
    border-radius: 14px; padding: 20px 22px;
    border: 1px solid #1F2A24; height: 100%;
    transition: border-color 0.2s ease;
}
.card:hover { border-color: #2F4A3C; }
.card h4 { margin: 0 0 12px 0; font-size: 0.78rem; color: #9CA3AF;
           text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700; }
.card .big { font-size: 1.5rem; font-weight: 700; color: #4ADE80;
             margin-bottom: 8px; letter-spacing: -0.02em; }
.card ul { padding-left: 18px; margin: 10px 0 0 0; color: #D1D5DB;
           font-size: 0.88rem; line-height: 1.55; }
.card li { margin-bottom: 5px; }

.pill { display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.03em;
        margin-right: 5px; margin-bottom: 4px; }
.pill-green   { background: rgba(74,222,128,0.13); color: #4ADE80;
                border: 1px solid rgba(74,222,128,0.25); }
.pill-amber   { background: rgba(251,191,36,0.13); color: #FBBF24;
                border: 1px solid rgba(251,191,36,0.25); }
.pill-red     { background: rgba(248,113,113,0.13); color: #F87171;
                border: 1px solid rgba(248,113,113,0.25); }
.pill-blue    { background: rgba(96,165,250,0.13); color: #60A5FA;
                border: 1px solid rgba(96,165,250,0.25); }
.pill-grey    { background: rgba(156,163,175,0.13); color: #D1D5DB;
                border: 1px solid rgba(156,163,175,0.2); }

.variant-card {
    background: linear-gradient(180deg, #141A18 0%, #11161499 100%);
    border: 1px solid #1F2A24; border-left: 3px solid #4ADE80;
    border-radius: 12px; padding: 18px 20px; margin-bottom: 12px;
}
.variant-card .headline { font-size: 1.1rem; font-weight: 700; color: #4ADE80;
                          margin-bottom: 8px; letter-spacing: -0.01em; }
.variant-card .body { color: #E5E7EB; line-height: 1.6; margin-bottom: 12px;
                      font-size: 0.95rem; }
.variant-card .cta  { color: #6EE7B7; font-weight: 600; font-size: 0.9rem; }

.brand {
    background: linear-gradient(90deg, #0F1B16 0%, #1B4332 100%);
    color: #E5E7EB; padding: 16px 24px; border-radius: 14px;
    margin-bottom: 20px; display: flex; align-items: center;
    justify-content: space-between; border: 1px solid #1F2A24;
    box-shadow: 0 1px 0 rgba(255,255,255,0.02) inset;
}
.brand .title { font-size: 1.45rem; font-weight: 700; letter-spacing: -0.02em;
                color: #4ADE80; }
.brand .sub { font-size: 0.83rem; color: #9CA3AF; }

.section-divider { border-top: 1px solid #1F2A24; margin: 26px 0 14px; }

.input-label {
    color: #9CA3AF; font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;
}
.input-label .tag {
    background: rgba(96,165,250,0.13); color: #60A5FA;
    border: 1px solid rgba(96,165,250,0.25); padding: 1px 8px;
    border-radius: 999px; font-size: 0.62rem; margin-left: 6px;
}

.footer-note {
    background: #0F1614; border: 1px solid #1F2A24; border-radius: 12px;
    padding: 14px 18px; margin-top: 30px; color: #9CA3AF; font-size: 0.85rem;
}
.footer-note a { color: #4ADE80; text-decoration: none; font-weight: 600; }
.footer-note a:hover { text-decoration: underline; }
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
  <div class="sub">Rabi 2025-26 · {len(growers):,} growers · model AUC {model.auc:.3f}</div>
</div>
""", unsafe_allow_html=True)


def _kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


# ── Top-level tabs ────────────────────────────────────────────────────
tab_studio, tab_dash = st.tabs(["🎯 Campaign Studio", "📊 Dashboard"])


# ─────────────────────────────────────────────────────────────────────
#  CAMPAIGN STUDIO TAB
# ─────────────────────────────────────────────────────────────────────
with tab_studio:
    # ── Collapsible: filters + ranked growers table ───────────────────
    with st.expander("🔎 Audience filters and ranked growers", expanded=False):
        f1, f2, f3, f4 = st.columns(4)
        state    = f1.selectbox("State",    ["All"] + sorted(growers["state"].unique()))
        crop     = f2.selectbox("Crop",     ["All"] + sorted(growers["crop"].unique()))
        device   = f3.selectbox("Device",   ["All", "smartphone", "keypad", "unknown"])
        language = f4.selectbox("Language", ["All"] + sorted(growers["language"].unique()))

        f5, f6 = st.columns([1, 1])
        as_of = f5.date_input(
            "As-of date", value=date.today(),
            help="Determines each farmer's current growth stage from their crop calendar, and the send-date baseline.",
        )
        top_n = f6.slider("Show top-N growers", 10, 200, 50, step=10)

        pool = growers.copy()
        if state    != "All": pool = pool[pool["state"]       == state]
        if crop     != "All": pool = pool[pool["crop"]        == crop]
        if device   != "All": pool = pool[pool["device_type"] == device]
        if language != "All": pool = pool[pool["language"]    == language]
        pool = pool.sort_values("click_score", ascending=False)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        _kpi(k1, "Matching growers", f"{len(pool):,}", f"of {len(growers):,} total")
        _kpi(k2, "States covered", f"{pool['state'].nunique()}", f"crops: {pool['crop'].nunique()}")
        _kpi(k3, "Avg click score",
             f"{pool['click_score'].mean():.3f}" if len(pool) else "N/A",
             "predicted receptivity")
        _kpi(k4, "Smartphone share",
             f"{(pool['device_type']=='smartphone').mean()*100:.0f}%" if len(pool) else "N/A",
             "drives channel mix")

        st.write("")
        st.markdown("##### Ranked growers")
        if not len(pool):
            st.warning("No growers match these filters.")
            st.stop()

        show = pool[["grower_id", "district", "crop", "current_stage",
                     "language", "device_type", "grower_farm_size",
                     "click_score"]].head(top_n).reset_index(drop=True)
        show.columns = ["Grower", "District", "Crop", "Stage", "Language",
                        "Device", "Farm (ac)", "Click score"]
        st.dataframe(
            show.style.background_gradient(subset=["Click score"], cmap="Greens")
                      .format({"Farm (ac)": "{:.2f}", "Click score": "{:.3f}"}),
            use_container_width=True, height=300, hide_index=True,
        )

    # ── Grower + threat selection (always visible, above campaign plan) ─
    st.markdown(
        "##### Select grower + active threat &nbsp;"
        "<span style='color:#9CA3AF;font-size:0.75rem;font-weight:500;"
        "text-transform:uppercase;letter-spacing:0.05em;'>(Input)</span>",
        unsafe_allow_html=True,
    )
    inp1, inp2, inp3 = st.columns([2, 2, 1])
    with inp1:
        grower_id = st.selectbox("Grower ID", sorted(pool["grower_id"].tolist()), index=0)
    with inp2:
        threat = st.selectbox(
            "Active pest / disease threat",
            ["None", "Yellow rust outbreak", "Aphid pressure", "Powdery mildew",
             "Late blight risk", "Pod borer infestation", "Stem borer"],
            help="Injected as a contextual signal into the generated copy.",
        )
    chosen = pool[pool["grower_id"] == grower_id].iloc[0]
    with inp3:
        st.write("")
        st.markdown(
            f"<div class='pill pill-green'>{chosen['crop']}</div>"
            f"<div class='pill pill-blue'>{chosen['current_stage']}</div>"
            f"<div class='pill pill-grey'>{chosen['device_type']}</div>",
            unsafe_allow_html=True,
        )
    st.caption(
        f"📍 {chosen['district']}, {chosen['state']} · {chosen['tehsil']}  "
        f"&nbsp;·&nbsp; 👤 {int(chosen['grower_age'])} yrs · "
        f"{chosen['grower_farm_size']:.2f} acres  "
        f"&nbsp;·&nbsp; 🎯 click score {chosen['click_score']:.3f} "
        f"(top {(pool['click_score'] > chosen['click_score']).mean()*100:.1f}% of filtered)"
    )


    # ── Build plan ────────────────────────────────────────────────────
    @st.cache_data(show_spinner="Building plan (weather, product, channel)…")
    def _ctx(grower_id: str, as_of_iso: str, threat_key: str):
        row = growers[growers["grower_id"] == grower_id].iloc[0]
        return build_context(data, row, active_threat=threat_key,
                             as_of=date.fromisoformat(as_of_iso))

    ctx = _ctx(grower_id, as_of.isoformat(), threat)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
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
  <div class="pill pill-blue">send {ctx.send_date or 'N/A'}</div>
  <div class="pill pill-grey">backup: {ch.secondary_channel}</div>
  <ul>
    <li><b>Format:</b> {ch.format}</li>
    <li><b>Social amplification:</b> {ch.social_amplification}</li>
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
        threat_pill = (
            f'<div class="pill pill-red">⚠ {threat}</div>'
            if threat != "None" else
            '<div class="pill pill-grey">no active threat</div>'
        )
        st.markdown(f"""
<div class="card">
  <h4>🌤️ Weather + threat · {w.district}</h4>
  <div class="big">{w.temp_c:.0f}°C · {w.humidity_pct:.0f}% RH</div>
  <div class="pill pill-{pp}">{pressure} disease pressure</div>
  <div class="pill pill-{sp}">{spray} spray window</div>
  {threat_pill}
  <ul>
    <li>Next 7 days: {w.rain_forecast_7d_mm:.0f} mm rain, {w.rainy_days_next_7} rainy days</li>
    <li>Source: <code>{w.source}</code> (Open-Meteo)</li>
  </ul>
</div>
""", unsafe_allow_html=True)

    # ── Receptivity benchmark + best-variant prediction ───────────────
    st.write("")
    rec_seg = segment_click_rate(
        data.whatsapp, growers,
        device_type=chosen["device_type"],
        language=chosen["language"],
        crop=chosen["crop"],
    )
    best_variant, why = predict_best_variant(
        grower=chosen.to_dict(), click_score=float(chosen["click_score"]))

    rb1, rb2, rb3 = st.columns([1, 1, 2])
    _kpi(rb1, "Segment click rate",
         f"{rec_seg['click_rate']*100:.2f}%",
         f"{rec_seg['clicked']:,} / {rec_seg['delivered']:,} delivered · {rec_seg['level']}")
    _kpi(rb2, "Segment open rate",
         f"{rec_seg['open_rate']*100:.2f}%",
         rec_seg.get("proxy_note") or f"benchmark for {rec_seg['level']}")
    with rb3:
        st.markdown(f"""
<div class="kpi">
  <div class="kpi-label">Predicted best creative variant</div>
  <div class="kpi-value">{best_variant.replace('_',' ').title()}</div>
  <div class="kpi-sub">{why}</div>
</div>""", unsafe_allow_html=True)

    # ── Generate buttons ──────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Generate copy")
    st.caption("Nothing runs until you click.")

    gen_col1, gen_col2, gen_col3 = st.columns([1, 1, 2])
    gen_default = gen_col1.button("✨ Generate default copy",
                                  type="primary", use_container_width=True)
    gen_all = gen_col2.button("🎨 Generate 3 variants",
                              use_container_width=True)
    gen_col3.caption(
        "Default = one message in the predicted-best style. "
        "Variants = urgent + educational + social proof side-by-side."
    )

    ss_key = f"output::{grower_id}::{threat}"
    view_key = f"view::{grower_id}::{threat}"
    img_key = f"image::{grower_id}::{threat}"
    user_key = st.session_state.get("user_gemini_key", "") or None

    if gen_default:
        with st.spinner("Generating…"):
            out = generate(ctx.to_brief(variant=best_variant), api_key=user_key)
        prev = st.session_state.get(ss_key, {})
        prev["default"] = out
        st.session_state[ss_key] = prev
        st.session_state[view_key] = "default"

    if gen_all:
        with st.spinner("Generating 3 variants…"):
            outs = generate_variants(ctx.to_brief(), api_key=user_key)
        prev = st.session_state.get(ss_key, {})
        prev["variants"] = outs
        st.session_state[ss_key] = prev
        st.session_state[view_key] = "variants"

    output = st.session_state.get(ss_key, {})
    current_view = st.session_state.get(view_key)

    if not output:
        st.info("Pick a generate action above.")
    else:
        # View switcher (radio styled as segmented control). Defaults to
        # whichever output was generated most recently.
        view_options = []
        if "default" in output:  view_options.append("📨 Default copy")
        if "variants" in output: view_options.append("🎨 Variants")
        view_options.append("📥 Bulk export")

        default_label = (
            "🎨 Variants" if current_view == "variants" and "variants" in output
            else "📨 Default copy" if "default" in output
            else view_options[0]
        )
        selected = st.radio(
            "view", view_options,
            index=view_options.index(default_label),
            horizontal=True, label_visibility="collapsed",
        )

        if selected == "📨 Default copy" and "default" in output:
            c0 = output["default"]
            badge = {"gemini": "pill-green", "ollama": "pill-amber",
                     "template": "pill-grey"}.get(c0.provider, "pill-grey")
            st.markdown(
                f"<div class='pill {badge}'>provider: {c0.provider}</div>"
                f"<div class='pill pill-blue'>variant: {c0.variant}</div>"
                f" <span style='color:#9CA3AF;font-size:0.85rem;'>{c0.notes}</span>",
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(f"""
<div class="variant-card">
  <div class="headline">{c0.headline}</div>
  <div class="body">{c0.body}</div>
  <div class="cta">→ {c0.cta}</div>
</div>""", unsafe_allow_html=True)
                st.markdown("**🎨 Visual concept** _(English, for designer)_")
                st.info(c0.visual_concept or "N/A")

                img_col_a, img_col_b = st.columns([1, 3])
                gen_img = img_col_a.button(
                    "🖼️ Generate image", use_container_width=True, key="gen_img_btn",
                    help="Generate a poster image from the visual concept using Gemini.",
                )
                img_col_b.caption(
                    "Uses Gemini image preview models. Image generation is available only with a paid/upgraded Gemini API key; "
                    "if you hit a rate limit or lack image capability, add your own API key at the bottom of the page."
                )
                if gen_img and c0.visual_concept:
                    with st.spinner("Generating image (this can take ~15-30s)…"):
                        img_res = generate_image(c0.visual_concept, api_key=user_key)
                    st.session_state[img_key] = img_res

                img_res = st.session_state.get(img_key)
                if img_res:
                    if img_res.image_bytes:
                        st.image(img_res.image_bytes,
                                 caption=f"Generated · model: {img_res.model}",
                                 use_container_width=True)
                        st.download_button(
                            "⬇️ Download image", data=img_res.image_bytes,
                            file_name=f"harvestiq_{grower_id}.png",
                            mime="image/png",
                        )
                    else:
                        st.markdown(
                            f"""<div style='background:rgba(248,113,113,0.08);border:1px solid
                            rgba(248,113,113,0.3);border-radius:10px;padding:14px 16px;
                            color:#FCA5A5;font-size:0.88rem;line-height:1.55;'>
                            <b>Image generation unavailable</b><br>{img_res.error}
                            </div>""",
                            unsafe_allow_html=True,
                        )

                st.markdown("**🎬 Video script** _(English, for producer)_")
                st.info(c0.video_script or "N/A")
            with col2:
                st.markdown(f"**📱 SMS fallback** _(≤160 chars · {len(c0.sms_fallback)} used)_")
                st.code(c0.sms_fallback, language="text")
                st.markdown("**☎️ IVR voice script**")
                st.code(c0.voice_script, language="text")

        elif selected == "🎨 Variants" and "variants" in output:
            variants = output["variants"]
            cols = st.columns(len(variants))
            for col, v in zip(cols, variants):
                with col:
                    mark = " ★" if v.variant == best_variant else ""
                    st.markdown(
                        f"<div class='pill pill-blue'>{v.variant}{mark}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"""
<div class="variant-card">
  <div class="headline">{v.headline}</div>
  <div class="body">{v.body}</div>
  <div class="cta">→ {v.cta}</div>
</div>""", unsafe_allow_html=True)
                    with st.expander("SMS · voice · visual · video"):
                        st.markdown("**SMS**")
                        st.code(v.sms_fallback, language="text")
                        st.markdown("**Voice**")
                        st.code(v.voice_script, language="text")
                        st.markdown("**Visual concept**")
                        st.write(v.visual_concept or "N/A")
                        st.markdown("**Video script**")
                        st.write(v.video_script or "N/A")

        elif selected == "📥 Bulk export":
            st.caption("Build plans for the top-K filtered growers. "
                       "Demonstrates scale: thousands of micro-targeted plans, no extra human effort.")
            n = st.slider("How many growers?", 10, min(500, len(pool)), 50,
                          step=10, key="bulk_slider")
            if st.button("Build plans"):
                rows = []
                prog = st.progress(0.0)
                for i, (_, r) in enumerate(pool.head(n).iterrows()):
                    cx = build_context(data, r, active_threat=threat, as_of=as_of)
                    bv, _ = predict_best_variant(
                        grower=r.to_dict(), click_score=float(r["click_score"]))
                    rows.append({
                        "grower_id": r["grower_id"], "state": r["state"],
                        "district": r["district"], "crop": r["crop"],
                        "stage": r["current_stage"], "language": r["language"],
                        "device": r["device_type"],
                        "click_score": round(r["click_score"], 3),
                        "best_variant": bv,
                        "product": cx.recommendation.product,
                        "channel": cx.channel.primary_channel,
                        "social_amplification": cx.channel.social_amplification,
                        "send_date": cx.send_date,
                        "active_threat": threat,
                        "weather": cx.weather.summary,
                    })
                    prog.progress((i + 1) / n)
                out_df = pd.DataFrame(rows)
                st.dataframe(out_df, use_container_width=True, height=320)
                st.download_button(
                    "⬇️ Download CSV",
                    data=out_df.to_csv(index=False).encode("utf-8"),
                    file_name="harvestiq_plans.csv", mime="text/csv",
                )

    # ── BYO Gemini API key (rate-limit fallback) ──────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    with st.expander("🔑 Hitting Gemini rate limits? Use your own API key", expanded=False):
        st.markdown(
            "If you're seeing slow responses, quota errors, or want to use your own quota, "
            "paste a personal Gemini API key below. It's stored only in this browser session.<br>"
            "Get one free at <a href='https://aistudio.google.com/api-keys' target='_blank'>aistudio.google.com/api-keys</a>.",
            unsafe_allow_html=True,
        )
        key_in = st.text_input(
            "Gemini API key", type="password",
            value=st.session_state.get("user_gemini_key", ""),
            placeholder="AIza…",
            help="Used for both text and image generation. Leave blank to use the app's default key.",
        )
        kcol1, kcol2 = st.columns([1, 5])
        if kcol1.button("Save key", use_container_width=True):
            st.session_state["user_gemini_key"] = key_in.strip()
            st.success("Key saved for this session." if key_in.strip() else "Cleared, using app default.")
        if st.session_state.get("user_gemini_key"):
            kcol2.caption("✅ Using your personal key for this session.")
        else:
            kcol2.caption("Using the app's default key.")


# ─────────────────────────────────────────────────────────────────────
#  DASHBOARD TAB
# ─────────────────────────────────────────────────────────────────────
with tab_dash:
    st.markdown("##### Dataset analytics across Track-1 sources")

    d1, d2, d3, d4 = st.columns(4)
    wa_total = len(data.whatsapp)
    wa_clicked = int(data.whatsapp["clicked_status"].sum())
    overall_click = wa_clicked / wa_total if wa_total else 0.0
    funnel_imp = int(data.funnel["social_post_impression"].sum())
    funnel_leads = int(data.funnel["lead_form_submission"].sum())
    _kpi(d1, "Growers", f"{len(growers):,}",
         f"{growers['state'].nunique()} states · {growers['crop'].nunique()} crops")
    _kpi(d2, "WhatsApp messages sent", f"{wa_total:,}",
         f"{wa_clicked:,} clicks · {overall_click*100:.2f}% CTR")
    _kpi(d3, "Digital impressions", f"{funnel_imp/1e6:.2f}M",
         f"{funnel_leads:,} leads · {funnel_leads/funnel_imp*1e4:.1f} per 10k")
    _kpi(d4, "Languages covered", f"{growers['language'].nunique()}",
         f"{(growers['device_type'] == 'smartphone').mean()*100:.0f}% on smartphone")

    st.write("")

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.markdown("**Growers by crop**")
        by_crop = growers["crop"].value_counts().reset_index()
        by_crop.columns = ["crop", "growers"]
        fig = px.bar(by_crop, x="crop", y="growers",
                     color="growers", color_continuous_scale="Greens")
        fig.update_layout(template="plotly_dark", showlegend=False,
                          margin=dict(l=0, r=0, t=10, b=0), height=280,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with r1c2:
        st.markdown("**Device mix**")
        dev = growers["device_type"].value_counts().reset_index()
        dev.columns = ["device", "growers"]
        fig = px.pie(dev, names="device", values="growers", hole=0.55,
                     color_discrete_sequence=["#4ADE80", "#60A5FA", "#9CA3AF"])
        fig.update_layout(template="plotly_dark",
                          margin=dict(l=0, r=0, t=10, b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)
    with r1c3:
        st.markdown("**Language mix**")
        lang = growers["language"].value_counts().reset_index()
        lang.columns = ["language", "growers"]
        fig = px.bar(lang, x="growers", y="language", orientation="h",
                     color="growers", color_continuous_scale="Greens")
        fig.update_layout(template="plotly_dark", showlegend=False,
                          margin=dict(l=0, r=0, t=10, b=0), height=280,
                          coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("**WhatsApp engagement by crop**")
        wa_crop = data.whatsapp_funnel_by_crop().sort_values("sent", ascending=False)
        long = wa_crop.melt(id_vars="campaign_crop",
                            value_vars=["delivery_rate", "open_rate", "click_rate"],
                            var_name="metric", value_name="rate")
        fig = px.bar(long, x="campaign_crop", y="rate", color="metric",
                     barmode="group",
                     color_discrete_sequence=["#4ADE80", "#60A5FA", "#FBBF24"])
        fig.update_layout(template="plotly_dark",
                          margin=dict(l=0, r=0, t=10, b=0), height=320,
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
    with r2c2:
        st.markdown("**Click rate by device × language (top 10)**")
        wa_g = data.whatsapp.merge(
            growers[["grower_id", "device_type", "language"]], on="grower_id")
        seg = (wa_g.groupby(["device_type", "language"])
                    .agg(delivered=("delivered_status", "sum"),
                         clicked=("clicked_status", "sum"))
                    .reset_index())
        seg = seg[seg["delivered"] > 30]
        seg["click_rate"] = seg["clicked"] / seg["delivered"]
        seg = seg.sort_values("click_rate", ascending=False).head(10)
        seg["segment"] = seg["device_type"] + " · " + seg["language"]
        fig = px.bar(seg, x="click_rate", y="segment", orientation="h",
                     color="click_rate", color_continuous_scale="Greens",
                     text=seg["click_rate"].map(lambda r: f"{r*100:.1f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(template="plotly_dark",
                          margin=dict(l=0, r=0, t=10, b=0), height=320,
                          coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Digital funnel by week (4 flagship Rabi campaigns)**")
    f = data.funnel.copy()
    f["week_start_date"] = pd.to_datetime(f["week_start_date"])
    fig = px.line(f.sort_values("week_start_date"),
                  x="week_start_date", y="landing_page_visits",
                  color="campaign_crop", markers=True,
                  color_discrete_sequence=["#4ADE80", "#60A5FA", "#FBBF24", "#F472B6"])
    fig.update_layout(template="plotly_dark",
                      margin=dict(l=0, r=0, t=10, b=0), height=320,
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
