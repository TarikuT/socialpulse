"""SocialPulse - YouTube comment analyzer (Streamlit UI).

Uses Claude Sonnet 4.6 by default (validated 78%+ agreement on Ethiopian
content). Falls back gracefully if Anthropic key missing — will fail loudly
with a clear message rather than silently using a stale config.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from socialpulse_core.youtube import get_youtube_comments
from socialpulse_core.analyzer import (
    analyze_comments,
    summarize_overall,
    DEFAULT_MODEL,
)
from socialpulse_core.viz import (
    comment_galaxy,
    sentiment_donut,
    theme_bar,
    generate_wordcloud_image,
)


# ---------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------
def _get_secret(key):
    value = None
    try:
        value = st.secrets.get(key)
    except Exception:
        value = None
    return value or os.environ.get(key, "")


YOUTUBE_API_KEY = _get_secret("YOUTUBE_API_KEY")
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")


def _api_key_for_model(model: str) -> str:
    """Return the right API key for the chosen model."""
    if model.startswith("claude-"):
        return ANTHROPIC_API_KEY
    return OPENAI_API_KEY


# ---------------------------------------------------------------
# Page
# ---------------------------------------------------------------
st.set_page_config(
    page_title="SocialPulse - YouTube comment analyzer",
    layout="wide",
    page_icon=":dart:",
)

st.title("SocialPulse")
st.caption("AI-powered audience reaction analysis for East African content.")

with st.sidebar:
    st.markdown("### How to use")
    st.markdown(
        "1. Paste a public YouTube video URL.\n"
        "2. Choose how many comments to analyze.\n"
        "3. Click **Analyze**."
    )
    st.markdown("---")
    st.markdown("**Languages supported:** English, Amharic, Afaan Oromo, Tigrinya.")
    st.markdown(f"**Model:** `{DEFAULT_MODEL}`")
    st.markdown("Sentiment is judged by an LLM in the comment's native language.")
    if not YOUTUBE_API_KEY:
        st.warning("YouTube API key not configured.")
    if DEFAULT_MODEL.startswith("claude-") and not ANTHROPIC_API_KEY:
        st.warning("Anthropic API key not configured.")
    elif not DEFAULT_MODEL.startswith("claude-") and not OPENAI_API_KEY:
        st.warning("OpenAI API key not configured.")


# ---------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------
col_in1, col_in2 = st.columns([3, 1])
with col_in1:
    video_url = st.text_input("YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")
with col_in2:
    max_comments = st.number_input("Max comments", min_value=20, max_value=500, value=150, step=10)

run = st.button("Analyze", type="primary", use_container_width=False)


# ---------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _fetch(url, api_key, n):
    return get_youtube_comments(url, api_key=api_key, max_comments=n)


@st.cache_data(show_spinner=False, ttl=3600)
def _analyze(df, llm_key, model, _progress_cb=None):
    return analyze_comments(df, api_key=llm_key, model=model, progress_callback=_progress_cb)


@st.cache_data(show_spinner=False, ttl=3600)
def _summarize(df, llm_key, model):
    return summarize_overall(df, api_key=llm_key, model=model)


if run and video_url:
    if not YOUTUBE_API_KEY:
        st.error("Missing YOUTUBE_API_KEY. Add it to .streamlit/secrets.toml.")
        st.stop()

    llm_key = _api_key_for_model(DEFAULT_MODEL)
    if not llm_key:
        need = "ANTHROPIC_API_KEY" if DEFAULT_MODEL.startswith("claude-") else "OPENAI_API_KEY"
        st.error(f"Missing {need}. Add it to .streamlit/secrets.toml.")
        st.stop()

    with st.spinner("Fetching comments from YouTube..."):
        try:
            comments_df = _fetch(video_url, YOUTUBE_API_KEY, int(max_comments))
        except Exception as exc:
            st.error(f"Failed to fetch comments: {exc}")
            st.stop()

    if comments_df.empty:
        st.warning("No comments returned. The video may have comments disabled or be unavailable.")
        st.stop()

    st.success(f"Fetched {len(comments_df)} comments.")

    progress_slot = st.empty()
    progress_bar = progress_slot.progress(
        0.0, text=f"Analyzing {len(comments_df)} comments with {DEFAULT_MODEL}..."
    )

    def _update_progress(done, total):
        frac = (done / total) if total else 1.0
        progress_bar.progress(frac, text=f"Analyzing... {done}/{total} batches complete")

    try:
        enriched = _analyze(comments_df, llm_key, DEFAULT_MODEL, _progress_cb=_update_progress)
    except Exception as exc:
        progress_slot.empty()
        st.error(f"Analysis failed: {exc}")
        st.stop()
    progress_slot.empty()

    total = len(enriched)
    pos = int((enriched["sentiment_label"] == "positive").sum())
    neg = int((enriched["sentiment_label"] == "negative").sum())
    avg_score = float(enriched["sentiment_score"].mean()) if total else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Comments analyzed", total)
    m2.metric("Positive", f"{pos}  ({pos / total:.0%})" if total else "-")
    m3.metric("Negative", f"{neg}  ({neg / total:.0%})" if total else "-")
    m4.metric("Avg. sentiment", f"{avg_score:+.2f}")

    st.subheader("Overall reaction")
    with st.spinner("Summarizing reaction..."):
        try:
            summary = _summarize(enriched, llm_key, DEFAULT_MODEL)
        except Exception as exc:
            summary = f"_Could not generate summary: {exc}_"
    st.markdown(summary)

    st.subheader("Comment galaxy")
    st.caption(
        "Each dot is one comment. Position shows when and how positive/negative. "
        "Size shows likes. Hover for the text."
    )
    st.plotly_chart(comment_galaxy(enriched), use_container_width=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.plotly_chart(sentiment_donut(enriched), use_container_width=True)
    with col_b:
        st.plotly_chart(theme_bar(enriched), use_container_width=True)

    if "language" in enriched.columns:
        lang_counts = enriched["language"].value_counts()
        st.subheader("Language mix")
        st.bar_chart(lang_counts)

    with st.expander("Word cloud (original language)"):
        non_en = enriched[enriched["language"].isin(["am", "om", "ti"])]
        if not non_en.empty:
            dom_lang = non_en["language"].value_counts().idxmax()
            wc_texts = non_en["text"].astype(str).tolist()
        else:
            dom_lang = "en"
            wc_texts = enriched["text"].astype(str).tolist()
        fig = generate_wordcloud_image(wc_texts, language=dom_lang)
        if fig is None:
            st.info("Word cloud unavailable (missing wordcloud package or fonts).")
        else:
            st.pyplot(fig)

    with st.expander("Raw analyzed data"):
        display_cols = [
            "text", "language", "sentiment_label", "sentiment_score",
            "themes", "translation_en", "like_count", "published_at", "author",
        ]
        cols_present = [c for c in display_cols if c in enriched.columns]
        st.dataframe(enriched[cols_present], use_container_width=True)
        st.download_button(
            "Download CSV",
            data=enriched[cols_present].to_csv(index=False).encode("utf-8"),
            file_name="socialpulse_results.csv",
            mime="text/csv",
        )
