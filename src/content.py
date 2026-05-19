"""Content generation engine — Gemini → Ollama → template fallback.

Produces complete creative brief per farmer:
  - headline + WhatsApp body + CTA
  - SMS fallback (<=160 chars)
  - IVR voice script (45 sec)
  - visual concept (poster / image description)
  - short video script (15-30 sec for social media reels)

Supports multiple creative variants for the 'scale personalisation'
requirement (5-10 → thousands of micro-targeted versions).
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


def _resolve_key(override: str | None) -> str:
    return (override or "").strip() or GEMINI_API_KEY

Provider = Literal["gemini", "ollama", "template"]

VARIANT_STYLES = {
    "urgent":       "Lead with urgency. The pest/disease window is closing this week. Use one short imperative line up front.",
    "educational":  "Lead with the agronomic insight. Explain WHY this stage matters for this crop in this region. Position the product as the protective action.",
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
    active_threat: str = "none"
    weather_summary: str = ""
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
    visual_concept: str = ""
    video_script: str = ""
    notes: str = ""


SYSTEM_INSTRUCTIONS = """You are an agricultural marketing copywriter for Syngenta India.
You write to smallholder farmers (often 1 to 5 acres) in the language they speak.
Your copy MUST be:
- agronomically accurate (never recommend a product outside its stage window)
- specific to the farmer's crop, region, current growth stage, weather, and any active pest or disease threat
- vernacular: write in the requested language using the native script, NOT romanised
- short: WhatsApp body under 60 words; SMS under 160 characters; voice script under 45 seconds spoken
- action oriented: every message ends with a concrete next step
- trust building: reference local context (district, season)
- never invent product claims; stick to the product category
- visual_concept and video_script should be in ENGLISH (these are creative briefs for designers and producers, not for the farmer)
Return STRICTLY valid JSON with keys: headline, body, cta, sms_fallback, voice_script, visual_concept, video_script."""


def _build_user_prompt(brief: ContentBrief) -> str:
    style = VARIANT_STYLES.get(brief.variant, VARIANT_STYLES["default"])
    threat_instruction = (
        f"\nACTIVE THREAT: '{brief.active_threat}' has been reported in this farmer's area. "
        "You MUST reference this specific threat by name in the headline or body. "
        "Frame the product as the direct protective response to this threat."
        if brief.active_threat and brief.active_threat.lower() not in {"none", ""}
        else ""
    )
    return f"""Generate a complete creative brief for one farmer. CREATIVE STYLE: {style}{threat_instruction}

CONTEXT (JSON):
{json.dumps(asdict(brief), indent=2, ensure_ascii=False)}

OUTPUT (JSON only, no preface, no markdown fences):
{{
  "headline": "...",         // 6-10 words, attention-grabbing, in farmer's language
  "body": "...",              // 35-55 words, WhatsApp body, in farmer's language
  "cta": "...",               // 5-10 words, one clear next step, in farmer's language
  "sms_fallback": "...",      // <=160 characters total, in farmer's language
  "voice_script": "...",      // 35-45 seconds spoken, conversational, in farmer's language
  "visual_concept": "...",    // 2-3 sentences describing a poster/image (English, for designer)
  "video_script": "..."       // 15-30 second short-form video script with shot directions (English, for producer)
}}"""


def _try_gemini(brief: ContentBrief, api_key: str | None = None) -> ContentResult | None:
    key = _resolve_key(api_key)
    if not key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
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
            cta="", sms_fallback="", voice_script="", visual_concept="",
            video_script="", notes=f"gemini_error: {e}",
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
    threat_line = (
        f" Local alert: {brief.active_threat} reported in your area."
        if brief.active_threat and brief.active_threat.lower() not in {"none", ""}
        else ""
    )
    intros = {
        "urgent":       f"This week is critical for your {brief.crop}.{threat_line}",
        "educational":  f"At the {stage} stage, your {brief.crop} is most vulnerable.{threat_line}",
        "social_proof": f"Many farmers in {brief.district} are protecting their {brief.crop} now.",
        "default":      f"Namaste! Your {brief.crop} crop in {brief.district} is at the {stage} stage.{threat_line}",
    }
    intro = intros.get(brief.variant, intros["default"])
    body = (
        f"{intro} Use {brief.product} to protect yield from the threats common "
        f"in {brief.state} this season. Talk to your nearest Syngenta retailer this week."
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
        visual_concept=(
            f"Mid-shot of a smallholder farmer in {brief.district} inspecting a healthy "
            f"{brief.crop} crop at sunrise, Syngenta {brief.product} pack visible at lower "
            f"right. Warm earthy tones, single-line headline in {brief.language} at top."
        ),
        video_script=(
            f"00:00 wide shot of {brief.crop} field in {brief.district}. "
            f"00:03 farmer looks worried at affected leaves. "
            f"00:08 voiceover names the {stage} risk. "
            f"00:15 product pack appears with one-line CTA. "
            f"00:22 farmer smiles next to healthy crop. End with Syngenta logo."
        ),
        notes="Generated from local template (LLM unavailable).",
    )


def generate(brief: ContentBrief, *, prefer: Provider | None = None,
             api_key: str | None = None) -> ContentResult:
    order: list[Provider] = ["gemini", "ollama", "template"] if not prefer else [prefer]
    for provider in order:
        if provider == "gemini":
            r = _try_gemini(brief, api_key=api_key)
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
                      variants: list[str] | None = None,
                      *, api_key: str | None = None) -> list[ContentResult]:
    variants = variants or ["urgent", "educational", "social_proof"]
    return [generate(ContentBrief(**{**asdict(brief), "variant": v}),
                     api_key=api_key) for v in variants]


# ── Image generation (Gemini image preview via google-genai SDK) ──────

@dataclass
class ImageResult:
    image_bytes: bytes | None
    model: str = ""
    error: str = ""


# Image model priority:
# 1. gemini-3.1-flash-image-preview  — fast, high-volume (Nano Banana 2)
# 2. gemini-3-pro-image-preview       — high-fidelity, thinking (Nano Banana Pro)
# 3. gemini-2.5-flash-image           — speed/efficiency fallback (Nano Banana)
IMAGE_MODELS = [
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
]

IMAGE_STYLE_GUIDE = (
    "Photorealistic, cinematic, warm earthy tones, golden-hour lighting, "
    "shallow depth of field, smallholder Indian farming context, "
    "documentary-style realism, no text or logos in the image."
)


def _friendly_image_error(model: str, exc: Exception) -> str:
    """Convert a raw API exception into a readable one-liner."""
    msg = str(exc)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return (
            f"{model} quota exhausted. "
            "This model has a free-tier limit of 0 requests. "
            "Enable billing at aistudio.google.com or add a paid API key."
        )
    if "404" in msg or "not found" in msg.lower():
        return f"{model} not found or not available in your region."
    if "403" in msg or "API_KEY" in msg:
        return "Invalid or missing Gemini API key."
    # Strip raw JSON / stack noise — keep first sentence only
    first = msg.split("\n")[0][:200]
    return first


def generate_image(visual_concept: str, *,
                   api_key: str | None = None) -> ImageResult:
    """Generate a poster image from a visual concept brief.

    Tries IMAGE_MODELS in order. Uses response_modalities=["IMAGE"] via
    the google-genai SDK — the correct path for Gemini image generation.
    """
    key = _resolve_key(api_key)
    if not key:
        return ImageResult(None, error="No Gemini API key configured. Add one in the section at the bottom of the page.")

    prompt = f"{visual_concept.strip()}\n\nStyle: {IMAGE_STYLE_GUIDE}"

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        return ImageResult(None, error=f"google-genai package not installed. Run: pip install google-genai  ({e})")

    client = genai.Client(api_key=key)
    last_err = "No models attempted."

    for model_name in IMAGE_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    import base64
                    data = part.inline_data.data
                    if isinstance(data, str):
                        data = base64.b64decode(data)
                    return ImageResult(data, model=model_name)
            last_err = f"{model_name}: response contained no image parts."
        except Exception as e:  # noqa: BLE001
            last_err = _friendly_image_error(model_name, e)
            continue

    return ImageResult(None, error=last_err)
