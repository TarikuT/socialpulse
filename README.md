# SocialPulse

AI-powered audience-reaction analysis for East African video content.

SocialPulse pulls public YouTube comments and uses GPT-4o-mini to detect language, score sentiment, and extract recurring themes in their native language — preserving Amharic idioms, Afaan Oromo expressions, and Tigrinya nuance that English-first sentiment tools miss. Results are presented as an interactive "comment galaxy" plus theme and sentiment breakdowns, and a plain-language summary suitable for sharing with brand and creator teams.

## What it analyzes

- **Languages:** English, Amharic, Afaan Oromo, Tigrinya, mixed script.
- **Per comment:** language, sentiment label (positive/neutral/negative), sentiment score (-1 to +1), 1–3 theme tags, English translation.
- **Per video:** sentiment mix, top themes with average sentiment per theme, language distribution, AI-written overall reaction summary.

## Architecture

```
socialpulse/
├── app/
│   └── socialpulse_app.py     Streamlit UI (presentation only)
├── socialpulse_core/
│   ├── youtube.py             YouTube Data API v3 fetcher
│   ├── analyzer.py            GPT-4o-mini batched classifier + summarizer
│   └── viz.py                 Plotly comment galaxy, donut, theme bar
├── static/                    Fonts for Ethiopic word clouds
├── requirements.txt
└── .streamlit/secrets.toml    API keys (NOT committed; see Setup)
```

UI calls core modules, which are pure Python — no Streamlit imports — so the pipeline is testable and re-usable from notebooks, schedulers, or future non-Streamlit surfaces.

## Setup (local)

1. Clone the repo and `cd socialpulse/`.
2. Create and activate a Python 3.10+ environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate            # Windows PowerShell
   # or: source .venv/bin/activate   # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your API keys to `.streamlit/secrets.toml` (this file is gitignored — never commit it):
   ```toml
   YOUTUBE_API_KEY = "your-youtube-data-api-v3-key"
   OPENAI_API_KEY  = "your-openai-api-key"
   ```
   - YouTube key: enable YouTube Data API v3 in Google Cloud Console, create an API key, restrict it to that API.
   - OpenAI key: create one at https://platform.openai.com/api-keys.
5. Run:
   ```bash
   streamlit run app/socialpulse_app.py
   ```

## Deployment (Streamlit Community Cloud)

1. Push to GitHub.
2. At https://share.streamlit.io, click **New app**, choose this repo, set the main file to `socialpulse/app/socialpulse_app.py`.
3. In **Advanced settings → Secrets**, paste the same TOML keys you used locally.
4. Deploy. Subsequent pushes to `main` redeploy automatically.

## Cost

With `gpt-4o-mini`, analyzing ~150 comments per video costs roughly $0.002–0.005. A few hundred analyses per month sit well under $5.

## Status

Early MVP. Sentiment accuracy on Amharic is being validated against a manually-labeled sample (see `roadmap.md`). Not yet recommended for high-stakes decisions without a human reading the raw comment table.

## License

Private project. Not yet released.
