import pandas as pd
from textblob import TextBlob
from langdetect import detect
from deep_translator import GoogleTranslator


def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text  # return original if translation fails


def analyze_sentiment(text: str) -> dict:
    blob = TextBlob(text)
    return {
        'polarity': blob.polarity,
        'subjectivity': blob.subjectivity
    }


def summarize_comments(df: pd.DataFrame) -> pd.DataFrame:
    df['language'] = df['text'].apply(detect_language)
    df['translated_text'] = df['text'].apply(translate_to_english)
    df['polarity'] = df['translated_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['translated_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return df

