"""LLM-powered comment analyzer.

Replaces the previous langdetect + googletrans + TextBlob pipeline with a
single batched GPT-4o-mini call that produces, per comment:

    - language          : ISO 639-1 code or "other"
    - sentiment_label   : "positive" | "neutral" | "negative"
    - sentiment_score   : float in [-1.0, 1.0]
    - themes            : list of short tags (1-3 items, 1-2 words each)
    - translation_en    : English translation (or the original if English)

Designed for Ethiopian YouTube content: handles Amharic (Ethiopic script),
Afaan Oromo, Tigrinya, English, and mixed-script comments. The model judges
sentiment in the original language rather than translating-then-scoring,
which preserves idioms, sarcasm, and religious expressions that VADER
misreads.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable, Optional

import pandas as pd
from openai import OpenAI


logger = logging.getLogger(__name__)

# Conservative batch size — large enough to amortize prompt overhead,
# small enough to stay well under context limits and recover quickly from
# a malformed batch response.
DEFAULT_BATCH_SIZE = 20
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You analyze YouTube comments on East African content.
Comments are in English, Amharic (Ethiopic script), Afaan Oromo (Latin or Ethiopic), Tigrinya, or mixed.
They often contain sarcasm, religious expressions, slang, emoji, and code-switching between Amharic and English.

Judge sentiment as a native speaker of the comment's language would, NOT after a literal translation.
Phrases like "ግሩም ነው" or "በጣም ጥሩ" are strongly positive; "ምንም አይደለም" is dismissive; "ዋው" is excited.
Religious expressions like "እግዚአብሔር ይባርክ" are positive blessings.

Return STRICT JSON only, no commentary."""


USER_TEMPLATE = """Analyze each comment below. Return JSON of the form:
{{"results": [{{"language": str, "sentiment_label": str, "sentiment_score": float, "themes": [str], "translation_en": str}}, ...]}}

Rules:
- language: ISO 639-1 code among "en", "am", "om", "ti", or "other".
- sentiment_label: exactly one of "positive", "neutral", "negative".
- sentiment_score: float in [-1.0, 1.0]. Strong positive ~0.8, mild positive ~0.3, neutral ~0.0, mild negative ~-0.3, strong negative ~-0.8.
- themes: 1-3 short tags (1-2 words each, lowercase English) describing what the comment is about. Examples: "praise", "music quality", "lyrics", "artist appearance", "political", "nostalgia", "complaint about audio".
- translation_en: faithful English translation. If already English, return the original lightly cleaned.
- Preserve the input order. Output exactly {n} result objects.

Comments:
{comments}
"""


def _format_comments(batch: list[str]) -> str:
    return "\n".join(f"{i + 1}. {c.strip()}" for i, c in enumerate(batch))


def _analyze_batch(
    client: OpenAI,
    batch: list[str],
    model: str,
) -> list[dict]:
    """Call the model on one batch. Returns a list of dicts of length == len(batch),
    padded with neutral defaults if the model misbehaves on individual rows."""
    user_prompt = USER_TEMPLATE.format(n=len(batch), comments=_format_comments(batch))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        results = parsed.get("results", []) if isinstance(parsed, dict) else []
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse batch JSON: %s", exc)
        results = []

    # Pad or trim to match batch size, with neutral fallbacks.
    normalized: list[dict] = []
    for i, original in enumerate(batch):
        item = results[i] if i < len(results) and isinstance(results[i], dict) else {}
        normalized.append(
            {
                "language": str(item.get("language", "other"))[:5],
                "sentiment_label": _coerce_label(item.get("sentiment_label")),
                "sentiment_score": _coerce_score(item.get("sentiment_score")),
                "themes": _coerce_themes(item.get("themes")),
                "translation_en": str(item.get("translation_en") or original),
            }
        )
    return normalized


def _coerce_label(value) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"positive", "negative", "neutral"}:
            return v
    return "neutral"


def _coerce_score(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(-1.0, min(1.0, f))


def _coerce_themes(value) -> list[str]:
    if isinstance(value, list):
        cleaned = [str(x).strip().lower() for x in value if str(x).strip()]
        return cleaned[:3]
    return []


def analyze_comments(
    comments: Iterable[str] | pd.DataFrame,
    openai_api_key: str,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Analyze a list of comments (or DataFrame with a 'text' column).

    Returns a DataFrame with the original input columns plus:
        language, sentiment_label, sentiment_score, themes, translation_en
    """
    if isinstance(comments, pd.DataFrame):
        input_df = comments.copy().reset_index(drop=True)
        texts = input_df["text"].astype(str).tolist()
    else:
        texts = [str(c) for c in comments]
        input_df = pd.DataFrame({"text": texts})

    if not texts:
        return input_df.assign(
            language=[], sentiment_label=[], sentiment_score=[], themes=[], translation_en=[]
        )

    client = OpenAI(api_key=openai_api_key)

    enriched: list[dict] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            enriched.extend(_analyze_batch(client, batch, model))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Batch starting at %d failed: %s", start, exc)
            # Neutral fallback for the whole batch — don't kill the run.
            enriched.extend(
                {
                    "language": "other",
                    "sentiment_label": "neutral",
                    "sentiment_score": 0.0,
                    "themes": [],
                    "translation_en": text,
                }
                for text in batch
            )

    enriched_df = pd.DataFrame(enriched)
    out = pd.concat([input_df, enriched_df], axis=1)
    return out


def summarize_overall(
    df: pd.DataFrame,
    openai_api_key: str,
    model: str = DEFAULT_MODEL,
    max_samples: int = 60,
) -> str:
    """Produce a 3-4 sentence plain-language summary of the overall reaction,
    suitable for the top of the report. Samples comments across sentiments
    so a vocal minority doesn't dominate the prompt."""
    if df.empty:
        return "No comments to summarize."

    # Stratified sample across sentiment buckets
    samples: list[str] = []
    for label in ("positive", "negative", "neutral"):
        bucket = df[df["sentiment_label"] == label]
        n = min(len(bucket), max_samples // 3)
        if n > 0:
            samples.extend(bucket.sample(n, random_state=42)["translation_en"].astype(str).tolist())

    if not samples:
        samples = df["translation_en"].astype(str).tolist()[:max_samples]

    counts = df["sentiment_label"].value_counts().to_dict()

    client = OpenAI(api_key=openai_api_key)
    prompt = (
        "Here is a sample of English-translated YouTube comments on a video, "
        f"with overall sentiment distribution {counts}. Write 3-4 sentences "
        "for a general audience: what is the dominant reaction, what recurring "
        "themes appear, and what's a surprising or notable minority view? "
        "Be honest and specific. No hedging language.\n\nComments:\n"
        + "\n".join(f"- {s}" for s in samples)
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return (response.choices[0].message.content or "").strip()
