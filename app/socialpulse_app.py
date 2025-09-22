import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import streamlit as st
import pandas as pd

from socialpulse_core.youtube import get_youtube_comments
from socialpulse_core.analyzer import analyze_comments
from socialpulse_core.viz import generate_wordcloud, plot_sentiment_histogram

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="SocialPulse AI", layout="wide")
st.title("📊 SocialPulse AI — YouTube Comment Analyzer")

# -----------------------------
# Sidebar instructions
# -----------------------------
with st.sidebar:
    st.markdown("### 🛠 How to Use")
    st.markdown("1. Paste a public YouTube video link")
    st.markdown("2. Click 'Run Analysis'")
    st.markdown("3. View language & sentiment visualizations")

# -----------------------------
# Input section
# -----------------------------
video_url = st.text_input("🔗 Enter a YouTube video URL:")
run_button = st.button("🚀 Run Analysis")

# -----------------------------
# Fetch + Analyze
# -----------------------------
if run_button and video_url:

    @st.cache_data(show_spinner="📥 Fetching comments from YouTube...")
    def fetch_comments(url):
        return get_youtube_comments(url, os.environ.get("YOUTUBE_API_KEY"), max_comments=150)

    df = fetch_comments(video_url)

    if df.empty:
        st.warning("🚫 No comments found or failed to fetch comments.")
        st.stop()

    st.success(f"✅ Retrieved {len(df)} comments.")
    st.divider()

    st.info("🔍 Analyzing comment language and sentiment...")
    result = analyze_comments(df['text'].tolist())

    if not result['original']:
        st.warning("⚠️ No translatable comments were found. Try another video.")
        st.stop()

    # -----------------------------
    # Visualizations
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌍 Language WordCloud")
        generate_wordcloud(result['original'], title="Original Language WordCloud", language=result['language'][0])

    with col2:
        st.subheader("🗣️ English Translated WordCloud")
        generate_wordcloud(result['translated'], title="Translated to English")

    st.subheader("📈 Sentiment Distribution")
    plot_sentiment_histogram(result['sentiment'])

    # -----------------------------
    # Raw Data
    # -----------------------------
    st.markdown("---")
    with st.expander("📄 View Raw Data Table"):
        st.dataframe(pd.DataFrame(result), use_container_width=True)
