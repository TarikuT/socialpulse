import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import pathlib

def generate_wordcloud(texts, title="WordCloud", language="en"):
    """
    Generate and display a word cloud for the provided texts.

    Args:
        texts (list[str]): List of input strings.
        title (str): Title of the WordCloud.
        language (str): Detected language code (used for font selection).
    """
    if not texts:
        st.warning("⚠️ No text available for word cloud.")
        return

    # Set font path for Amharic or Afaan Oromo
    font_path = None
    if language in ['am', 'om']:
        font_path = pathlib.Path(__file__).parent.parent / "static" / "NotoSansEthiopic-Regular.ttf"

        if not font_path.exists():
            st.error("❌ Ethiopic font not found. Please place 'NotoSansEthiopic-Regular.ttf' in the 'static/' folder.")
            return

    try:
        cloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            font_path=font_path,
            colormap="viridis"
        ).generate(" ".join(texts))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Failed to generate word cloud: {str(e)}")


def plot_sentiment_histogram(sentiments):
    """
    Plot a histogram of sentiment polarity scores.

    Args:
        sentiments (list[float]): List of sentiment polarity values.
    """
    if not sentiments:
        st.warning("⚠️ No sentiment data to plot.")
        return

    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(sentiments, bins=20, color="skyblue", edgecolor="black")
        ax.set_xlabel("Sentiment Polarity")
        ax.set_ylabel("Frequency")
        ax.set_title("Sentiment Distribution")
        ax.grid(True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Failed to plot sentiment histogram: {str(e)}")
