"""SocialPulse QA labeler.

Internal tool for measuring how well the GPT-4o-mini classifier matches a
human native-speaker's judgment on real comments. Not deployed publicly.

Workflow:
    1. Run the main SocialPulse app on a YouTube video, download the CSV,
       save it to ../qa/ (next to this file's parent).
    2. From the socialpulse/ directory, run:
           streamlit run app/qa_labeler.py
    3. Pick the CSV, label comments one at a time, see metrics at the end.

Labels are saved to qa/labels_<source>.csv after every click so nothing
is ever lost if the browser closes mid-session.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
QA_DIR = PROJECT_DIR / "qa"
QA_DIR.mkdir(exist_ok=True)


st.set_page_config(
    page_title="SocialPulse QA labeler",
    layout="centered",
    page_icon=":pencil2:",
)

st.title("QA labeler")
st.caption(
    "Label comments as a native speaker would; measure agreement with GPT. "
    "Internal tool - not for public deployment."
)


def _list_source_csvs():
    return sorted(
        [p for p in QA_DIR.glob("*.csv") if not p.name.startswith("labels_")]
    )


sources = _list_source_csvs()
if not sources:
    st.warning(
        f"No CSVs found in {QA_DIR}. Run the main app on a video, download "
        "the results CSV, and drop it in the qa/ folder before reloading."
    )
    st.stop()

source_names = [p.name for p in sources]
choice = st.selectbox("Source CSV", source_names, key="source_choice")
source_path = QA_DIR / choice
labels_path = QA_DIR / f"labels_{choice}"


@st.cache_data
def _load_source(path_str, mtime):
    df = pd.read_csv(path_str)
    required = {"text", "sentiment_label", "sentiment_score", "language"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source CSV missing columns: {missing}")
    return df


try:
    source_df = _load_source(str(source_path), source_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load CSV: {exc}")
    st.stop()


def _stratified_sample(df, n_total=100, seed=42):
    classes = ["positive", "neutral", "negative"]
    per_class = n_total // len(classes)
    picks = []
    for c in classes:
        bucket = df[df["sentiment_label"] == c]
        if len(bucket) == 0:
            continue
        picks.append(bucket.sample(min(per_class, len(bucket)), random_state=seed))
    sampled = pd.concat(picks) if picks else df.sample(min(n_total, len(df)), random_state=seed)
    if len(sampled) < n_total:
        remaining = df.drop(sampled.index)
        if len(remaining) > 0:
            extra = remaining.sample(min(n_total - len(sampled), len(remaining)), random_state=seed)
            sampled = pd.concat([sampled, extra])
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


sample_size = st.sidebar.slider(
    "Sample size", 20, min(200, len(source_df)),
    value=min(100, len(source_df)), step=10
)
sample_df = _stratified_sample(source_df, n_total=sample_size)


if labels_path.exists():
    existing = pd.read_csv(labels_path)
    label_map = dict(zip(existing["text"], existing["human_label"]))
else:
    label_map = {}


# Reset current_idx when the user switches to a different CSV from the dropdown,
# otherwise the stale index lets the code fall through to metrics with no labels.
if st.session_state.get("active_source") != choice:
    st.session_state.active_source = choice
    st.session_state.pop("current_idx", None)

if "current_idx" not in st.session_state:
    unlabeled_indices = [i for i in range(len(sample_df)) if sample_df.iloc[i]["text"] not in label_map]
    st.session_state.current_idx = unlabeled_indices[0] if unlabeled_indices else len(sample_df)


def _save_labels():
    rows = []
    for _, row in sample_df.iterrows():
        if row["text"] in label_map:
            rows.append({
                "text": row["text"],
                "language": row.get("language", ""),
                "gpt_label": row.get("sentiment_label", ""),
                "gpt_score": row.get("sentiment_score", 0.0),
                "human_label": label_map[row["text"]],
            })
    pd.DataFrame(rows).to_csv(labels_path, index=False)


def _record(label):
    text = sample_df.iloc[st.session_state.current_idx]["text"]
    label_map[text] = label
    _save_labels()
    nxt = st.session_state.current_idx + 1
    while nxt < len(sample_df) and sample_df.iloc[nxt]["text"] in label_map:
        nxt += 1
    st.session_state.current_idx = nxt


done = sum(1 for _, row in sample_df.iterrows() if row["text"] in label_map)
total = len(sample_df)
st.progress(done / total if total else 1.0, text=f"Labeled {done} of {total}")


if st.session_state.current_idx < len(sample_df) and done < len(sample_df):
    row = sample_df.iloc[st.session_state.current_idx]

    st.markdown("---")
    st.caption(
        f"Comment {st.session_state.current_idx + 1} of {len(sample_df)}  "
        f"|  Language hint: {row.get('language', '?')}"
    )
    st.markdown(f"### {row['text']}")

    col_p, col_n, col_neg, col_skip = st.columns(4)
    if col_p.button("Positive", use_container_width=True, type="primary"):
        _record("positive")
        st.rerun()
    if col_n.button("Neutral", use_container_width=True):
        _record("neutral")
        st.rerun()
    if col_neg.button("Negative", use_container_width=True):
        _record("negative")
        st.rerun()
    if col_skip.button("Skip", use_container_width=True):
        _record("skip")
        st.rerun()

    with st.expander("Reveal GPT's prediction (for debugging only)"):
        st.write(
            f"GPT label: **{row.get('sentiment_label', '?')}**  |  "
            f"score: {row.get('sentiment_score', 0):+.2f}"
        )

else:
    if done == 0:
        st.info("Nothing labeled yet for this CSV. Pick a comment to start.")
        st.stop()

    st.success("All comments labeled. Here's the accuracy report.")

    labeled_rows = [
        {
            "text": row["text"],
            "language": row.get("language", ""),
            "gpt_label": row.get("sentiment_label", ""),
            "human_label": label_map.get(row["text"], ""),
        }
        for _, row in sample_df.iterrows()
        if row["text"] in label_map
    ]
    labeled = pd.DataFrame(labeled_rows, columns=["text", "language", "gpt_label", "human_label"])

    valid = labeled[labeled["human_label"] != "skip"].copy()
    n_valid = len(valid)
    skipped = (labeled["human_label"] == "skip").sum()

    if n_valid == 0:
        st.warning("All comments were skipped. Nothing to evaluate.")
        st.stop()

    agree = (valid["gpt_label"] == valid["human_label"]).sum()
    overall_acc = agree / n_valid

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Labeled", n_valid)
    m2.metric("Skipped", int(skipped))
    m3.metric("Agreement", f"{overall_acc:.1%}")
    target = 0.85
    m4.metric("Vs. 85% target", f"{(overall_acc - target):+.1%}")

    st.subheader("Per-class recall")
    recall_rows = []
    for cls in ["positive", "neutral", "negative"]:
        bucket = valid[valid["human_label"] == cls]
        if len(bucket) == 0:
            continue
        correct = (bucket["gpt_label"] == cls).sum()
        recall_rows.append({
            "class": cls,
            "n_human_labels": len(bucket),
            "gpt_correct": int(correct),
            "recall": f"{correct / len(bucket):.0%}",
        })
    st.dataframe(pd.DataFrame(recall_rows), use_container_width=True, hide_index=True)

    st.subheader("Confusion matrix (rows = human, cols = GPT)")
    labels_order = ["positive", "neutral", "negative"]
    matrix = pd.crosstab(
        pd.Categorical(valid["human_label"], categories=labels_order),
        pd.Categorical(valid["gpt_label"], categories=labels_order),
        dropna=False,
    )
    st.dataframe(matrix, use_container_width=True)

    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(valid["human_label"], valid["gpt_label"], labels=labels_order)
        kappa_str = f"{kappa:.3f}"
    except ImportError:
        kappa_str = "n/a (install scikit-learn)"
    st.markdown(
        f"**Cohen's kappa:** {kappa_str}  "
        "_(>0.6 = substantial agreement, >0.8 = almost perfect)_"
    )

    st.subheader("Disagreements (where to focus prompt tuning)")
    disagreements = valid[valid["gpt_label"] != valid["human_label"]].reset_index(drop=True)
    if len(disagreements) == 0:
        st.success("No disagreements. Either accuracy is perfect or sample is too small.")
    else:
        cols_to_show = ["text", "language", "human_label", "gpt_label"]
        st.dataframe(disagreements[cols_to_show], use_container_width=True, hide_index=True)
        st.download_button(
            "Download disagreements CSV",
            data=disagreements.to_csv(index=False).encode("utf-8"),
            file_name=f"disagreements_{choice}",
            mime="text/csv",
        )

    st.markdown("---")
    if st.button("Reset and start over"):
        if labels_path.exists():
            labels_path.unlink()
        st.session_state.pop("current_idx", None)
        st.cache_data.clear()
        st.rerun()
