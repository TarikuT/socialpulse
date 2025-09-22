import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import os


def generate_wordcloud(texts, title="WordCloud", language="en"):
    if not texts:
        print("No text to generate wordcloud.")
        return

    font_path = None
    if language in ['am', 'om']:
        font_path = os.path.join("static", "NotoSansEthiopic-Regular.ttf")

    cloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        font_path=font_path,
        colormap="viridis"
    ).generate(" ".join(texts))

    plt.figure(figsize=(10, 5))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_sentiment_histogram(sentiments):
    plt.figure(figsize=(8, 4))
    plt.hist(sentiments, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Sentiment Polarity")
    plt.ylabel("Frequency")
    plt.title("Sentiment Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()