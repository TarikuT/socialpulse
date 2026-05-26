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

Batches run concurrently across a thread pool (each thread blocks on an
OpenAI HTTP call, ideal I/O-bound work for threads). A 170-comment video
drops from ~60s sequential to ~10s with 6 workers.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Optional

import pandas as pd
from openai import OpenAI


logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 20
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_WORKERS = 6


SYSTEM_PROMPT = """You analyze YouTube comments on East African content.
Comments are in English, Amharic (Ethiopic script), Afaan Oromo (Latin or Ethiopic), Tigrinya, or mixed.
They often contain sarcasm, religious expressions, slang, emoji, and code-switching between Amharic and English.

Judge sentiment as a native speaker of the comment's language would, NOT after a literal translation.
Phrases like "GROOM nawe" or "BETAM TIRU" are strongly positive; "MNIM AYDELLEM" is dismissive.
Religious expressions are positive blessings.

Return STRICT JSON only, no commentary."""


USER_TEMPLATE = """Analyze each comment below. Return JSON of the form:
{{"results": [{{"language": str, "sentiment_label": str, "sentiment_score": float, "themes": [str], "translation_en": str}}, ...]}}

Rules:
- language: ISO 639-1 code among "en", "am", "om", "ti", or "other".
- sentiment_label: exactly one of "positive", "neutral", "negative".
- sentiment_score: float in [-1.0, 1.0]. Strong positive ~0.8, mild positive ~0.3, neutral ~0.0, mild negative ~-0.3, strong negative ~-0.8.
- themes: 1-3 short tags (1-2 words each, lowercase English) describing what the comment is about.
- translation_en: faithful English translation. If already English, return the original lightly cleaned.
- Preserve the input order. Output exactly {n} result objects.

Comments:
{comments}
"""


def _format_comments(batch):
    return "\n".join(f"{i + 1}. {c.strip()}" for i, c in enumerate(batch))


def _coerce_label(value):
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"positive", "negative", "neutral"}:
            return v
    return "neutral"


def _coerce_score(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(-1.0, min(1.0, f))


def _coerce_themes(value):
    if isinstance(value, list):
        cleaned = [str(x).strip().lower() for x in value if str(x).strip()]
        return cleaned[:3]
    return []


def _neutral_fallback(batch):
    return [
        {
            "language": "other",
            "sentiment_label": "neutral",
            "sentiment_score": 0.0,
            "themes": [],
            "translation_en": text,
        }
        for text in batch
    ]


def _analyze_batch(client, batch, model):
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

    normalized = []
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


def analyze_comments(
    comments,
    openai_api_key,
    model=DEFAULT_MODEL,
    batch_size=DEFAULT_BATCH_SIZE,
    max_workers=DEFAULT_MAX_WORKERS,
    progress_callback=None,
):
    """Analyze a list of comments (or DataFrame with a 'text' column).

    Pass progress_callback(done_batches, total_batches) for a UI progress bar.
    Returns a DataFrame with original columns plus language, sentiment_label,
    sentiment_score, themes, translation_en.
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

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    results = [None] * len(batches)

    workers = max(1, min(max_workers, len(batches)))
    done = 0
    if progress_callback:
        try:
            progress_callback(0, len(batches))
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_analyze_batch, client, batch, model): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.warning("Batch %d failed: %s", idx, exc)
                results[idx] = _neutral_fallback(batches[idx])
            done += 1
            if progress_callback:
                try:
                    progress_callback(done, len(batches))
                except Exception:
                    pass

    enriched = []
    for batch_result in results:
        if batch_result is not None:
            enriched.extend(batch_result)

    enriched_df = pd.DataFrame(enriched)
    out = pd.concat([input_df, enriched_df], axis=1)
    return out


def summarize_overall(df, openai_api_key, model=DEFAULT_MODEL, max_samples=60):
    """Produce a 3-4 sentence summary of overall reaction, stratified across
    sentiments so a vocal minority does not dominate."""
    if df.empty:
        return "No comments to summarize."

    samples = []
    for label in ("positive", "negative", "neutral"):
        bucket = df[df["sentiment_label"] == label]
        n = min(len(bucket), max_samples // 3)
        if n > 0:
            samples.extend(
                bucket.sample(n, random_state=42)["translation_en"].astype(str).tolist()
            )

    if not samples:
        samples = df["translation_en"].astype(str).tolist()[:max_samples]

    counts = df["sentiment_label"].value_counts().to_dict()

    client = OpenAI(api_key=openai_api_key)
    prompt = (
        "Here is a sample of English-translated YouTube comments on a video, "
        f"with overall sentiment distribution {counts}. Write 3-4 sentences "
        "for a general audience: what is the dominant reaction, what recurring "
        "themes appear, and what is a surprising or notable minority view? "
        "Be honest and specific. No hedging language.\n\nComments:\n"
        + "\n".join(f"- {s}" for s in samples)
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return (response.choices[0].message.content or "").strip()
