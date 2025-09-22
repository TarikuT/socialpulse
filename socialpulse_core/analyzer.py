from langdetect import detect
from textblob import TextBlob
from deep_translator import GoogleTranslator

def analyze_comments(texts):
    translated_texts = []
    detected_langs = []
    original_texts = []

    for text in texts:
        try:
            lang = detect(text)
        except:
            lang = 'unknown'

        if not lang or lang == 'unknown':
            if any(char >= '\u1200' and char <= '\u137F' for char in text):
                lang = 'am'  # Ethiopic unicode block fallback
            else:
                continue

        try:
            translated = GoogleTranslator(source=lang, target='en').translate(text)
        except Exception as e:
            print(f"Translation failed for: {text[:30]}... ({lang}) → {e}")
            translated = None

        if translated and translated.strip():
            translated_texts.append(translated.strip())
            detected_langs.append(lang)
            original_texts.append(text.strip())

    sentiments = []
    for text in translated_texts:
        try:
            blob = TextBlob(text)
            sentiments.append(blob.sentiment.polarity)
        except:
            sentiments.append(0.0)

    return {
        "original": original_texts,
        "language": detected_langs,
        "translated": translated_texts,
        "sentiment": sentiments,
    }

