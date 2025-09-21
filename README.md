
# SocialPulse AI - Production Version

This is the production-ready setup of the **SocialPulse AI** MVP, which extracts public comments (e.g., from YouTube), analyzes sentiment and language, and generates visual summaries.

## Project Structure

```
socialpulse/
├── app/                        # Main Streamlit app
├── socialpulse_core/          # Modular logic (NLP, visualization, APIs)
├── fonts/                     # Fonts for multilingual word clouds
├── .streamlit/                # API keys and secrets
├── requirements.txt
└── README.md
```

## Deployment (Streamlit Cloud)

1. Add your API keys to `.streamlit/secrets.toml`
2. Run locally using:
   streamlit run app/socialpulse_app.py

## Contact

For private deployment or extension (e.g., Facebook), contact the developer.
