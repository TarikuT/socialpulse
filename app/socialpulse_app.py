import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from socialpulse_core.youtube import get_youtube_comments
from socialpulse_core.analyzer import analyze_comments
from socialpulse_core.viz import generate_wordcloud, plot_sentiment_histogram


with st.sidebar:
    st.markdown("### 🛠 How to Use")
    st.markdown("1. Paste a public YouTube video link")
    st.markdown("2. The app will fetch & analyze top 100 comments")
    st.markdown("3. Visual insights & language support included!")

video_url = st.text_input("🎥 Enter YouTube Video URL")





st.set_page_config(page_title="SocialPulse AI", layout="wide")
st.title("📊 SocialPulse AI — YouTube Comment Analyzer")

st.markdown("""
Analyze public YouTube comments in multiple languages and generate sentiment & keyword visualizations.
""")

video_url = st.text_input("🔗 Enter a YouTube video URL:")

if video_url:
    st.info("Fetching comments...")
    df = get_youtube_comments(video_url, os.environ.get("YOUTUBE_API_KEY"), max_comments=150)

    if df.empty:
        st.warning("🚫 No comments found or failed to fetch comments.")
        st.stop()

    st.success(f"✅ Retrieved {len(df)} comments.")

    st.info("🔍 Analyzing comment language and sentiment...")
    result = analyze_comments(df['text'].tolist())

    if not result['original']:
        st.warning("⚠️ No translatable comments were found. Try another video.")
        st.stop()

    st.markdown("---")
    st.subheader("🌍 Language WordCloud")
    generate_wordcloud(result['original'], title="Original Language WordCloud", language=result['language'][0])

    st.subheader("🗣️ English Translated WordCloud")
    generate_wordcloud(result['translated'], title="Translated to English")

    st.subheader("📈 Sentiment Analysis")
    plot_sentiment_histogram(result['sentiment'])

    st.markdown("---")
    st.subheader("📄 Raw Data")
    st.dataframe(pd.DataFrame(result))

