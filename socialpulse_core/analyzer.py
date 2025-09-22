import warnings
from langdetect import detect
from textblob import TextBlob
from deep_translator import GoogleTranslator


def analyze_comments(texts):
    """
    Analyze a list of YouTube comments:
    - Detect language
    - Translate to English
    - Analyze sentiment (polarity)

    Parameters:
        texts (list[str]): List of original YouTube comment texts

    Returns:
        dict: {
            "original": list[str],
            "language": list[str],
            "translated": list[str],
            "sentiment": list[float]
        }
    """
    if not texts:
        return {"original": [], "language": [], "translated": [], "sentiment": []}

    translated_texts = []
    detected_langs = []
    original_texts = []

    for text in texts:
        text = text.strip()
        if not text:
            continue

        # Detect language
        try:
            lang = detect(text)
        except Exception:
            # Fallback detection for Ethiopic script (Amharic)
            if any('\u1200' <= char <= '\u137F' for char in text):
                lang = 'am'
            elif any('\u2D80' <= char <= '\u2DDF' for char in text):  # Ge'ez Extended for Afaan Oromo in Ethiopic
                lang = 'om'
            else:
                lang = 'unknown'

        if lang == 'unknown':
            continue

        # Translate to English
        try:
            translated = GoogleTranslator(source=lang, target='en').translate(text)
        except Exception as e:
            warnings.warn(f"Translation failed for: {text[:30]}... ({lang}) → {e}")
            continue

        if translated and translated.strip():
            original_texts.append(text)
            translated_texts.append(translated.strip())
            detected_langs.append(lang)

    # Sentiment Analysis
    sentiments = []
    for t in translated_texts:
        try:
            sentiments.append(TextBlob(t).sentiment.polarity)
        except Exception:
            sentiments.append(0.0)

    return {
        "original": original_texts,
        "language": detected_langs,
        "translated": translated_texts,
        "sentiment": sentiments,
    }
