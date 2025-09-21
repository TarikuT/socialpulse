import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googleapiclient.discovery import build
import openai
import re
from langdetect import detect
from googletrans import Translator

# ------------------------
# CONFIGURATION & SECRETS
# ------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]
youtube_api_key = st.secrets["YOUTUBE_API_KEY"]
translator = Translator()

def get_youtube_comments(video_id, max_comments=100):
    comments = []
    try:
        youtube = build("youtube", "v3", developerKey=youtube_api_key)
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        ).execute()

        while response and len(comments) < max_comments:
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
                if len(comments) >= max_comments:
                    break
            if "nextPageToken" in response:
                response = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response["nextPageToken"],
                    maxResults=100,
                    textFormat="plainText"
                ).execute()
            else:
                break
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
    return comments

def translate_to_english(comment):
    try:
        lang = detect(comment)
        if lang in ["am", "om"]:
            translated = translator.translate(comment, src=lang, dest="en")
            return translated.text
        else:
            return comment
    except Exception as e:
        return comment

def analyze_sentiment(comments):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for comment in comments:
        translated = translate_to_english(comment)
        score = analyzer.polarity_scores(translated)
        compound = score['compound']
        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        results.append({"original": comment, "translated": translated, "sentiment": sentiment})
    return pd.DataFrame(results)

def generate_wordcloud(comments):
    text = " ".join(comments)
    wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    font_path='fonts/NotoSansEthiopic-Regular.ttf').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def generate_summary(text):
    prompt = (
        "Summarize the main themes and emotional tone in the following YouTube comments:\n" + text[:3500] + "\nSummary:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        return f"Error generating summary: {e}"

# ------------------------
# STREAMLIT UI
# ------------------------
st.set_page_config(page_title="SocialPulse AI MVP", layout="centered")
st.title("📊 SocialPulse AI - Public Comment Analyzer")

video_url = st.text_input("Enter YouTube Video URL or Video ID:")

def extract_video_id(url_or_id):
    match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", url_or_id)
    return match.group(1) if match else url_or_id.strip()

video_id = extract_video_id(video_url)

max_comments = st.slider("Number of comments to analyze:", 10, 500, 100)
generate_summary_flag = st.checkbox("Generate AI Summary (uses OpenAI GPT)")

if st.button("🔍 Analyze") and video_id:
    with st.spinner("Fetching and analyzing comments..."):
        comments = get_youtube_comments(video_id, max_comments)
        if comments:
            df = analyze_sentiment(comments)
            st.subheader("📈 Sentiment Breakdown")
            st.bar_chart(df['sentiment'].value_counts())

            st.subheader("☁️ Word Cloud (Original Language)")
            generate_wordcloud(df['original'].tolist())

            st.subheader("📋 Comments Table")
            st.dataframe(df[['original', 'translated', 'sentiment']])

            if generate_summary_flag:
                st.subheader("🧠 AI Summary")
                summary = generate_summary(" ".join(df['translated'].tolist()))
                st.write(summary)
        else:
            st.warning("No comments found or unable to fetch.")
