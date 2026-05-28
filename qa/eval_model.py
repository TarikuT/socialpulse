"""Multi-model evaluation harness for SocialPulse.

Runs the same sentiment classification prompt through a chosen model on
already-human-labeled comments, then reports accuracy vs the human labels.

Usage from socialpulse/ directory:

    python qa/eval_model.py --model gpt-4o
    python qa/eval_model.py --model claude-sonnet-4-6
    python qa/eval_model.py --model claude-sonnet-4-6 --label-file qa/labels_seifuebs.csv

Without --label-file, runs against all qa/labels_*.csv files.

Reads API keys from .streamlit/secrets.toml (OPENAI_API_KEY, ANTHROPIC_API_KEY).
Predictions are saved to qa/preds_<model>_<labelfile>.csv so you can re-inspect
without re-spending on API calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


# ---------- Paths ----------

SCRIPT_DIR = Path(__file__).resolve().parent      # .../socialpulse/qa
PROJECT_DIR = SCRIPT_DIR.parent                    # .../socialpulse
QA_DIR = SCRIPT_DIR
SECRETS_PATH = PROJECT_DIR / ".streamlit" / "secrets.toml"


# ---------- Secrets ----------

def _load_secrets():
    """Tiny TOML parser for our flat key=value secrets file (no deps)."""
    secrets = {}
    if not SECRETS_PATH.exists():
        return secrets
    for line in SECRETS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        secrets[k.strip()] = v.strip().strip('"').strip("'")
    return secrets


# ---------- Prompts (same shape as analyzer.py) ----------

SYSTEM_PROMPT = (
    "You analyze YouTube comments on East African content. "
    "Comments are in English, Amharic (Ethiopic script), Afaan Oromo, Tigrinya, or mixed. "
    "Judge sentiment as a native speaker of the comment's language would, NOT after literal translation. "
    "Phrases of praise are positive; dismissive expressions are negative or neutral. "
    "When a comment discusses a heavy topic (war, conflict, hardship) but expresses pride, "
    "solidarity, or support, classify it by the SPEAKER's stance, not the topic's emotional words. "
    "Return STRICT JSON only."
)

USER_TEMPLATE = (
    'For each comment, return: {{"results": [{{"sentiment_label": "positive"|"neutral"|"negative"}}, ...]}}.\n'
    "Rules:\n"
    "- exactly one of positive, neutral, negative\n"
    "- Preserve input order. Output exactly {n} result objects.\n\n"
    "Comments:\n{comments}\n"
)


def _format_comments(batch):
    return "\n".join(f"{i + 1}. {c}" for i, c in enumerate(batch))


# ---------- Model adapters ----------

def call_openai(model, texts, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    prompt = USER_TEMPLATE.format(n=len(texts), comments=_format_comments(texts))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return resp.choices[0].message.content or "{}"


def call_anthropic(model, texts, api_key):
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    prompt = USER_TEMPLATE.format(n=len(texts), comments=_format_comments(texts))
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT + ' Return only a JSON object, no markdown fences, no preamble.',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    # Anthropic returns content as list of blocks
    raw = "".join(b.text for b in resp.content if hasattr(b, "text"))
    # Strip ```json fences if model adds them
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def predict_batch(model, batch, secrets):
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        raw = call_openai(model, batch, secrets["OPENAI_API_KEY"])
    elif model.startswith("claude-"):
        raw = call_anthropic(model, batch, secrets["ANTHROPIC_API_KEY"])
    else:
        raise ValueError(f"Unknown model family: {model}")

    try:
        parsed = json.loads(raw)
        results = parsed.get("results", []) if isinstance(parsed, dict) else []
    except json.JSONDecodeError:
        results = []

    labels = []
    for i in range(len(batch)):
        item = results[i] if i < len(results) and isinstance(results[i], dict) else {}
        lbl = str(item.get("sentiment_label", "neutral")).strip().lower()
        if lbl not in {"positive", "neutral", "negative"}:
            lbl = "neutral"
        labels.append(lbl)
    return labels


# ---------- Main eval ----------

def evaluate(label_file, model, secrets, batch_size=20, max_workers=6):
    df = pd.read_csv(label_file)
    valid = df[df["human_label"] != "skip"].copy().reset_index(drop=True)
    texts = valid["text"].astype(str).tolist()

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    predictions = [None] * len(batches)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(predict_batch, model, b, secrets): i for i, b in enumerate(batches)}
        done = 0
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                predictions[i] = fut.result()
            except Exception as exc:
                print(f"  Batch {i} failed: {exc}", file=sys.stderr)
                predictions[i] = ["neutral"] * len(batches[i])
            done += 1
            print(f"  [{done}/{len(batches)}] batches complete", end="\r", file=sys.stderr)
    print(file=sys.stderr)

    flat_preds = [p for batch_p in predictions for p in batch_p]
    valid["new_pred"] = flat_preds
    valid["agree_new"] = valid["new_pred"] == valid["human_label"]
    valid["agree_old"] = valid["gpt_label"] == valid["human_label"]

    elapsed = time.time() - t0
    return valid, elapsed


def print_metrics(valid, model_name, baseline_col="gpt_label"):
    from sklearn.metrics import cohen_kappa_score
    n = len(valid)
    new_acc = valid["agree_new"].sum() / n
    old_acc = valid["agree_old"].sum() / n
    new_k = cohen_kappa_score(valid["human_label"], valid["new_pred"], labels=["positive", "neutral", "negative"])
    old_k = cohen_kappa_score(valid["human_label"], valid[baseline_col], labels=["positive", "neutral", "negative"])

    print(f"\n  {'metric':<18s}  {'NEW (' + model_name + ')':<22s}  {'BASELINE (gpt-4o-mini)':<24s}")
    print(f"  {'-'*18}  {'-'*22}  {'-'*24}")
    print(f"  {'agreement':<18s}  {new_acc:.1%}  ({int(valid['agree_new'].sum())}/{n})        {old_acc:.1%}  ({int(valid['agree_old'].sum())}/{n})")
    print(f"  {'Cohens kappa':<18s}  {new_k:+.3f}                {old_k:+.3f}")

    print(f"\n  Per-class recall (when human says X, model says X):")
    for cls in ["positive", "neutral", "negative"]:
        bucket = valid[valid["human_label"] == cls]
        if len(bucket) == 0:
            continue
        new_r = (bucket["new_pred"] == cls).sum() / len(bucket)
        old_r = (bucket[baseline_col] == cls).sum() / len(bucket)
        print(f"    {cls:10s} (n={len(bucket):3d}):  new={new_r:.0%}    baseline={old_r:.0%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model id (gpt-4o, gpt-4o-mini, claude-sonnet-4-6, ...)")
    parser.add_argument("--label-file", help="Specific qa/labels_*.csv; if omitted, runs all")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    secrets = _load_secrets()
    needed_key = "OPENAI_API_KEY" if args.model.startswith(("gpt-", "o")) else "ANTHROPIC_API_KEY"
    if needed_key not in secrets or not secrets[needed_key]:
        print(f"ERROR: {needed_key} not found in {SECRETS_PATH}", file=sys.stderr)
        sys.exit(1)

    files = [Path(args.label_file)] if args.label_file else sorted(QA_DIR.glob("labels_*.csv"))
    if not files:
        print(f"No label files found in {QA_DIR}", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for f in files:
        print(f"\n=== {f.name}  (model: {args.model}) ===")
        valid, elapsed = evaluate(f, args.model, secrets, batch_size=args.batch_size)
        print(f"  Evaluated {len(valid)} comments in {elapsed:.1f}s")
        print_metrics(valid, args.model)

        out_path = QA_DIR / f"preds_{args.model.replace('/', '_')}_{f.name}"
        valid.to_csv(out_path, index=False)
        print(f"  Predictions saved to {out_path.name}")
        all_results.append(valid)

    if len(all_results) > 1:
        combined = pd.concat(all_results, ignore_index=True)
        print(f"\n{'='*60}")
        print(f"=== COMBINED across {len(all_results)} files (model: {args.model}) ===")
        print(f"{'='*60}")
        print(f"  Total comments: {len(combined)}")
        print_metrics(combined, args.model)


if __name__ == "__main__":
    main()
