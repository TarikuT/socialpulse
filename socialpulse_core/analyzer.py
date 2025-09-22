import pandas as pd
from textblob import TextBlob


def analyze_sentiment(text: str) -> dict:
    """Analyzes sentiment using TextBlob."""
    blob = TextBlob(text)
    return {
        'polarity': blob.polarity,
        'subjectivity': blob.subjectivity
    }


def summarize_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Adds sentiment analysis columns to comment DataFrame."""
    df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['summary'] = df['text'].apply(lambda x: TextBlob(x).noun_phrases[:3])
    return df

