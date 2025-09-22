import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import List
import streamlit as st


def plot_sentiment_histogram(df):
    """Visualizes sentiment polarity distribution."""
    st.subheader("Sentiment Polarity Distribution")
    fig, ax = plt.subplots()
    df['polarity'].hist(bins=20, ax=ax)
    ax.set_title("Polarity Histogram")
    ax.set_xlabel("Polarity")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


def generate_wordcloud(texts: List[str], font_path: str = None):
    """Generates and displays a word cloud from comment texts."""
    text_blob = " ".join(texts)
    wc = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate(text_blob)

    st.subheader("Word Cloud")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
