"""LLM-powered comment analyzer.

Default model is Claude Sonnet 4.6 (validated on Ethiopian content at ~78%
agreement, kappa 0.53 — substantially better than GPT-4o-mini on negative
sentiment, which is the most commercially important class).

Auto-detects OpenAI vs Anthropic based on the model name, so the same
codepath works for cheap cost-sensitive testing (gpt-4o-mini) or production
classification (claude-sonnet-4-6). Same JSON output schema either way.

Per comment, returns:
    language          : ISO 639-1 code or "other"
    sentiment_label   : positive | neutral | negative
    sentiment_score   : float in [-1.0, 1.0]
    themes            : list of short tags (1-3 items, 1-2 words each)
    translation_en    : English translation (or original if English)

Batches run concurrently across a thread pool.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Optional

import pandas as pd


logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 20
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_WORKERS = 6


SYSTEM_PROMPT = """You analyze YouTube comments on East African content.
Comments are in English, Amharic (Ethiopic script), Afaan Oromo (Latin or Ethiopic), Tigrinya, or mixed.
They often contain sarcasm, religious expressions, slang, emoji, and code-switching between Amharic and English.

Judge sentiment as a native speaker of the comment's language would, NOT after a literal translation.
When a comment discusses heavy topics (war, conflict, hardship) but expresses pride, solidarity, or support,
classify by the SPEAKER's stance, not the topic's emotional words.

Return STRICT JSON only, no commentary, no markdown fences."""


USER_TEMPLATE = """Analyze each comment below. Return JSON of the form:
{{"results": [{{"language": str, "sentiment_label": str, "sentiment_score": float, "themes": [str], "translation_en": str}}, ...]}}

Rules:
- language: ISO 639-1 code among "en", "am", "om", "ti", or "other".
- sentiment_label: exactly one of "positive", "neutral", "negative".
- sentiment_score: float in [-1.0, 1.0]. Strong positive ~0.8, mild positive ~0.3, neutral ~0.0, mild negative ~-0.3, strong negative ~-0.8.
- themes: 1-3 short tags (1-2 words each, lowercase English).
- translation_en: faithful English translation. If already English, return original lightly cleaned.
- Preserve input order. Output exactly {n} result objects.

Comments:
{comments}
"""


def _is_anthropic(model: str) -> bool:
    return model.startswith("claude-")


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


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences if a model wraps JSON in them."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
    return raw


def _call_openai(model, system, user, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return resp.choices[0].message.content or "{}"


def _call_anthropic(model, system, user, api_key):
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0.2,
    )
    raw = "".join(b.text for b in resp.content if hasattr(b, "text"))
    return _strip_fences(raw)


def _analyze_batch(batch, model, api_key):
    user_prompt = USER_TEMPLATE.format(n=len(batch), comments=_format_comments(batch))
    if _is_anthropic(model):
        raw = _call_anthropic(model, SYSTEM_PROMPT, user_prompt, api_key)
    else:
        raw = _call_openai(model, SYSTEM_PROMPT, user_prompt, api_key)

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
    api_key=None,
    model=DEFAULT_MODEL,
    batch_size=DEFAULT_BATCH_SIZE,
    max_workers=DEFAULT_MAX_WORKERS,
    progress_callback=None,
    # Legacy keyword for callers that haven't migrated:
    openai_api_key=None,
):
    """Analyze a list of comments (or DataFrame with a 'text' column).

    api_key: pass the appropriate key for the chosen model (Anthropic for
    claude-*, OpenAI for gpt-*). The legacy `openai_api_key` parameter is
    accepted for backward compatibility but is being phased out.

    progress_callback(done_batches, total_batches): optional UI hook.

    Returns a DataFrame with original input columns plus
        language, sentiment_label, sentiment_score, themes, translation_en.
    """
    if api_key is None:
        api_key = openai_api_key
    if not api_key:
        raise ValueError("No API key provided. Pass api_key=... for the chosen model.")

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
            executor.submit(_analyze_batch, batch, model, api_key): idx
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


def summarize_overall(
    df,
    api_key=None,
    model=DEFAULT_MODEL,
    max_samples=60,
    openai_api_key=None,  # legacy
):
    """3-4 sentence audience-reaction summary for the report header.
    Stratified-samples across sentiment buckets so a vocal minority doesn't
    dominate the prompt."""
    if api_key is None:
        api_key = openai_api_key
    if not api_key:
        return "(no API key for summary)"

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
    prompt = (
        "Here is a sample of English-translated YouTube comments on a video, "
        f"with overall sentiment distribution {counts}. Write 3-4 sentences "
        "for a general audience: what is the dominant reaction, what recurring "
        "themes appear, and what is a surprising or notable minority view? "
        "Be honest and specific. No hedging language.\n\nComments:\n"
        + "\n".join(f"- {s}" for s in samples)
    )

    if _is_anthropic(model):
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip()
