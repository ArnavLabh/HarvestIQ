"""Content generation engine — Gemini → Ollama → template fallback.

Supports multiple creative variants per grower (the "5–10 → thousands
of micro-targeted versions" requirement in the brief).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Literal

import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

Provider = Literal["gemini", "ollama", "template"]

VARIANT_STYLES = {
    "urgent":       "Lead with urgency — the pest/disease window is closing this week. Use one short imperative line up front.",
    "educational":  "Lead with the agronomic insight — explain WHY this stage matters for this crop in this region. Position the product as the protective action.",
    "social_proof": "Lead with what other farmers in the same district are doing. Use a quietly confident, peer-driven tone.",
    "default":      "Use a warm, respectful, agronomically precise tone.",
}


@dataclass
class ContentBrief:
    crop: str
    product: str
    state: str
    district: str
    language: str
    current_stage: str
    channel: str
    format: str
    tone: str
    variant: str = "default"
    farmer_age: int | None = None
    farm_size_acres: float | None = None


@dataclass
class ContentResult:
    provider: Provider
    variant: str
    headline: str
    body: str
    cta: str
    sms_fallback: str
    voice_script: str
    notes: str = ""


SYSTEM_INSTRUCTIONS = """You are an agricultural marketing copywriter for Syngenta India.
You write to smallholder farmers (often 1-5 acres) in the language they speak.
Your copy MUST be:
- agronomically accurate (never recommend a product outside its stage window)
- specific to the farmer's crop, region, and current growth stage
- vernacular: write in the requested language using the native script, NOT romanised
- short: WhatsApp body under 60 words; SMS under 160 characters; voice script under 45 seconds spoken
- action-oriented: every message ends with a concrete next step
- trust-building: reference local context (district, season)
- never invent product claims; stick to the product category
Return STRICTLY valid JSON with keys: headline, body, cta, sms_fallback, voice_script."""


def _build_user_prompt(brief: ContentBrief) -> str:
    style = VARIANT_STYLES.get(brief.variant, VARIANT_STYLES["default"])
    return f"""Generate marketing copy for one farmer. CREATIVE STYLE: {style}

CONTEXT (JSON):
{json.dumps(asdict(brief), indent=2, ensure_ascii=False)}

OUTPUT (JSON only, no preface, no markdown fences):
{{
  "headline": "...",       // 6-10 words, attention-grabbing
  "body": "...",            // 35-55 words, WhatsApp body
  "cta": "...",             // 5-10 words, one clear next step
  "sms_fallback": "...",    // <=160 characters total
  "voice_script": "..."     // 35-45 seconds spoken, conversational
}}"""


def _try_gemini(brief: ContentBrief) -> ContentResult | None:
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=SYSTEM_INSTRUCTIONS,
            generation_config={"response_mime_type": "application/json", "temperature": 0.8},
        )
        resp = model.generate_content(_build_user_prompt(brief))
        data = json.loads(resp.text)
        return ContentResult(provider="gemini", variant=brief.variant, **data)
    except Exception as e:  # noqa: BLE001
        return ContentResult(
            provider="template", variant=brief.variant, headline="", body="",
            cta="", sms_fallback="", voice_script="", notes=f"gemini_error: {e}",
        )


def _try_ollama(brief: ContentBrief) -> ContentResult | None:
    try:
        prompt = SYSTEM_INSTRUCTIONS + "\n\n" + _build_user_prompt(brief)
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "format": "json", "options": {"temperature": 0.8}},
            timeout=60,
        )
        r.raise_for_status()
        data = json.loads(r.json()["response"])
        return ContentResult(provider="ollama", variant=brief.variant, **data)
    except Exception:  # noqa: BLE001
        return None


def _template(brief: ContentBrief) -> ContentResult:
    stage = brief.current_stage.replace("_", " ")
    intros = {
        "urgent":       f"This week is critical for your {brief.crop}.",
        "educational":  f"At the {stage} stage, your {brief.crop} is most vulnerable.",
        "social_proof": f"Many farmers in {brief.district} are protecting their {brief.crop} now.",
        "default":      f"Namaste! Your {brief.crop} crop in {brief.district} is at the {stage} stage.",
    }
    intro = intros.get(brief.variant, intros["default"])
    body = (
        f"{intro} This is the window when {brief.product} protects yield "
        f"from the threats common in {brief.state} at this time. Talk to your "
        f"nearest Syngenta retailer this week."
    )
    return ContentResult(
        provider="template", variant=brief.variant,
        headline=f"Protect your {brief.crop} this {stage}",
        body=body,
        cta=f"Visit your retailer for {brief.product} this week.",
        sms_fallback=(
            f"Syngenta: {brief.crop} in {brief.district} is at {stage}. "
            f"Use {brief.product} now. Visit your retailer."
        )[:160],
        voice_script=(
            f"Namaste. This is Syngenta. Your {brief.crop} crop in {brief.district} "
            f"is at the {stage} stage. Right now is the best week to use {brief.product} "
            f"to protect your yield. Please visit your nearest retailer."
        ),
        notes="Generated from local template (LLM unavailable).",
    )


def generate(brief: ContentBrief, *, prefer: Provider | None = None) -> ContentResult:
    order: list[Provider] = ["gemini", "ollama", "template"] if not prefer else [prefer]
    for provider in order:
        if provider == "gemini":
            r = _try_gemini(brief)
            if r and r.provider == "gemini":
                return r
        elif provider == "ollama":
            r = _try_ollama(brief)
            if r:
                return r
        elif provider == "template":
            return _template(brief)
    return _template(brief)


def generate_variants(brief: ContentBrief,
                      variants: list[str] | None = None) -> list[ContentResult]:
    """Produce N creative variants of the same brief."""
    variants = variants or ["urgent", "educational", "social_proof"]
    results = []
    for v in variants:
        b = ContentBrief(**{**asdict(brief), "variant": v})
        results.append(generate(b))
    return results
