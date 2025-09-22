import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from socialpulse_core.youtube import get_youtube_comments
from socialpulse_core.analyzer import summarize_comments
from socialpulse_core.viz import plot_sentiment_histogram, generate_wordcloud

# 🔐 Load API key
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

st.set_page_config(page_title="SocialPulse AI", layout="wide")
st.title("📊 SocialPulse AI — YouTube Comment Insight")

video_url = st.text_input("🔗 Paste YouTube Video URL:")
max_comments = st.slider("How many comments to fetch?", min_value=10, max_value=500, step=10, value=100)

if st.button("Fetch & Analyze"):
    if "v=" in video_url:
        video_id = video_url.split("v=")[-1].split("&")[0]
        df = get_youtube_comments(video_id, YOUTUBE_API_KEY, max_comments=max_comments)

        if not df.empty:
            df = summarize_comments(df)

            st.subheader("📋 Raw Comments")
            st.dataframe(df[['author', 'text', 'language', 'translated_text', 'polarity']])

            plot_sentiment_histogram(df)

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                generate_wordcloud(df['text'].tolist(), title="Original Language WordCloud")
            with col2:
                generate_wordcloud(df['translated_text'].tolist(), title="Translated (English) WordCloud")

        else:
            st.warning("No comments retrieved. Try another video link.")
    else:
        st.error("Invalid YouTube URL")
