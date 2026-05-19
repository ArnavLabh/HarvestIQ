# HarvestIQ

**Track 1: AI-Powered Agricultural Marketing at Scale**
Syngenta x IIT Madras Hackathon 2026

---

## Overview

HarvestIQ is an AI marketing engine that generates personalised campaign plans for smallholder farmers across India. Given a farmer's profile, crop calendar, and live weather, it auto-selects the right product, channel, send timing, and vernacular copy, without any manual configuration per farmer.

---

## Problem Statement Coverage

| Brief Requirement | HarvestIQ Solution |
|---|---|
| Context-aware content generation | Gemini (primary) with Ollama and template fallback; output in the farmer's language and native script across WhatsApp, SMS, and IVR formats |
| Campaign targeting and timing | Logistic regression trained on WhatsApp engagement history; crop-stage-aware send-date logic per farmer |
| Predict campaign receptivity | Per-farmer click-probability score used to rank and filter the target pool |
| Scale personalisation | Three creative variants (urgent, educational, social proof) generated per farmer; bulk CSV export for thousands of plans |
| Offline and low-connectivity | Three-tier fallback: Gemini API, local Ollama model, deterministic template; weather data cached on disk |
| Feature phone and low-literacy users | Device type drives channel selection: smartphone gets WhatsApp rich media, keypad gets IVR voice call, unknown gets SMS |

---

## How It Works

1. **Filter** farmers by state, crop, device, and language from the sidebar
2. **Rank** the matching pool by predicted click probability (ML model)
3. **Select** a grower and the system auto-derives everything:
   - Recommended product based on crop, growth stage, and weather
   - Best channel and send date
   - Live weather context from Open-Meteo (temperature, humidity, disease pressure, spray window)
4. **Generate** vernacular WhatsApp body, SMS fallback, and IVR voice script
5. **Variants** tab produces three creative angles for A/B testing
6. **Bulk export** builds plans for the top-K filtered growers as a CSV

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (single-page, dark mode) |
| Targeting model | scikit-learn Logistic Regression |
| Content generation | Google Gemini API, Ollama (local fallback), template (offline fallback) |
| Weather | Open-Meteo API (free, no key required, disk-cached) |
| Language | Python 3.11 |

---

## Dataset Used (Track 1 Only)

| File | Role |
|---|---|
| `growers.csv` | Farmer profiles: state, district, language, device type, crop calendar |
| `whatsapp_campaign.csv` | Historical WhatsApp engagement: delivered, opened, clicked labels for training |
| `digital_funnel_weekly.csv` | Four flagship Rabi campaign funnels: impressions, visits, leads |

> Place the Syngenta-provided files inside `Syngenta_IITM_Hackathon_2026_dataset (1)/` before running the app.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add Gemini API key (optional, enables live content generation)
cp .env.example .env
# Edit .env and set GEMINI_API_KEY

# 3. Run
streamlit run app.py
```

Note: In the app's Content Studio (bottom of the page) there's an option to paste your Gemini API key. A free Gemini API key enables text/content generation; a paid or upgraded Gemini key also enables image generation from the app.

Without a Gemini key the app falls back to a built-in template (English), so the demo works fully offline.

---

## Project Structure

```
app.py                        Main Streamlit app (single page)
src/
    data.py                   Loads and joins the three Track-1 datasets
    targeting.py              Trains and applies the click-probability model
    product_recommender.py    Selects the best product from crop, stage, and weather
    channel.py                Maps device type and crop stage to channel and send date
    weather.py                Fetches and caches live weather from Open-Meteo
    context.py                Assembles the full campaign plan for one grower
    content.py                Generates vernacular copy with Gemini / Ollama / template
.env.example                  Template for API keys
requirements.txt              Python dependencies
```

---

## Key Design Decisions

**Resilience over perfection.** Content generation degrades gracefully from Gemini to Ollama to a deterministic template. The app never crashes due to a missing API key or network outage, which is critical for rural pilot environments.

**Transparency in every recommendation.** Each product pick and channel assignment surfaces its reasoning so a field representative or reviewer can understand and override it.

**Track 1 data only.** Retailer inventory, rep visit logs, and POS tables belong to Track 2 (Field Force Intelligence) and are deliberately excluded from this solution.
