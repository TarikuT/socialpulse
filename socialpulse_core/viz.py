"""Visualizations for SocialPulse.

Primary: comment_galaxy — an interactive Plotly scatter where every comment
is a dot positioned by publish time and sentiment intensity, colored by
sentiment, sized by likes, with hover revealing the comment text.

Secondary: sentiment_donut, theme_bar, and the original wordcloud (kept
for the original-language cloud, which still has marketing value).
"""

from __future__ import annotations

import pathlib
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral": "#94a3b8",
    "negative": "#ef4444",
}


def _wrap_for_hover(text: str, width: int = 60) -> str:
    """Insert <br> tags so long comments don't overflow the hover tooltip."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        if current_len + len(word) + 1 > width and current:
            lines.append(" ".join(current))
            current, current_len = [word], len(word)
        else:
            current.append(word)
            current_len += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)


def comment_galaxy(df: pd.DataFrame, title: str = "Comment galaxy") -> go.Figure:
    """Primary visualization. Each dot is one comment.

    Required columns: text, published_at, like_count, sentiment_score,
    sentiment_label, language, themes, translation_en.
    """
    if df.empty:
        return _empty_figure("No comments to plot.")

    plot_df = df.copy()
    plot_df["published_at"] = pd.to_datetime(plot_df["published_at"], utc=True, errors="coerce")
    plot_df = plot_df.dropna(subset=["published_at"])
    if plot_df.empty:
        return _empty_figure("No timestamped comments to plot.")

    # Make zero-like-count points still visible; map likes -> marker size.
    plot_df["marker_size"] = plot_df["like_count"].clip(lower=0).pow(0.5) * 4 + 6

    plot_df["hover_text"] = plot_df["translation_en"].fillna(plot_df["text"]).map(_wrap_for_hover)
    plot_df["theme_str"] = plot_df["themes"].apply(
        lambda ts: ", ".join(ts) if isinstance(ts, list) else ""
    )

    fig = px.scatter(
        plot_df,
        x="published_at",
        y="sentiment_score",
        color="sentiment_label",
        color_discrete_map=SENTIMENT_COLORS,
        size="marker_size",
        size_max=40,
        hover_data={
            "hover_text": True,
            "language": True,
            "theme_str": True,
            "like_count": True,
            "published_at": False,
            "sentiment_score": ":.2f",
            "marker_size": False,
            "sentiment_label": False,
        },
        labels={
            "published_at": "Comment date",
            "sentiment_score": "Sentiment (-1 negative ↔ +1 positive)",
            "hover_text": "Comment",
            "language": "Lang",
            "theme_str": "Themes",
            "like_count": "Likes",
        },
        title=title,
    )

    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="rgba(0,0,0,0.3)"), opacity=0.85),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Lang: %{customdata[1]} &nbsp;|&nbsp; Likes: %{customdata[3]}<br>"
            "Themes: %{customdata[2]}<br>"
            "Score: %{customdata[4]:.2f}<extra></extra>"
        ),
    )

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.25)")
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", range=[-1.1, 1.1]),
        legend_title_text="Sentiment",
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def sentiment_donut(df: pd.DataFrame, title: str = "Sentiment mix") -> go.Figure:
    """Clean donut chart replacing the previous histogram. Good for at-a-glance
    sentiment proportions in reports."""
    if df.empty or "sentiment_label" not in df.columns:
        return _empty_figure("No sentiment data.")
    counts = df["sentiment_label"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [SENTIMENT_COLORS.get(lbl, "#94a3b8") for lbl in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="white", width=2)),
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        title=title,
        showlegend=False,
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
    )
    return fig


def theme_bar(df: pd.DataFrame, top_n: int = 10, title: str = "Top themes") -> go.Figure:
    """Horizontal bar of the most-discussed themes, colored by average sentiment."""
    if df.empty or "themes" not in df.columns:
        return _empty_figure("No theme data.")

    exploded = df.explode("themes").dropna(subset=["themes"])
    exploded = exploded[exploded["themes"].astype(str).str.strip().astype(bool)]
    if exploded.empty:
        return _empty_figure("No themes extracted.")

    agg = (
        exploded.groupby("themes")
        .agg(count=("text", "size"), avg_sentiment=("sentiment_score", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(top_n)
        .sort_values("count", ascending=True)  # for horizontal bar order
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=agg["count"],
                y=agg["themes"],
                orientation="h",
                marker=dict(
                    color=agg["avg_sentiment"],
                    colorscale=[[0, "#ef4444"], [0.5, "#94a3b8"], [1, "#22c55e"]],
                    cmin=-1,
                    cmax=1,
                    colorbar=dict(title="Avg sentiment", thickness=10),
                ),
                hovertemplate="<b>%{y}</b><br>Comments: %{x}<br>Avg sentiment: %{marker.color:.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=80, r=20, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Mentions",
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        xref="paper", yref="paper", x=0.5, y=0.5,
        font=dict(size=14, color="#64748b"),
    )
    fig.update_layout(
        height=300,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# --- Kept from previous version for the original-language word cloud --------

def generate_wordcloud_image(
    texts: list[str],
    language: str = "en",
    font_dir: Optional[pathlib.Path] = None,
):
    """Returns a matplotlib Figure for a word cloud. UI layer renders it.
    Returns None if the wordcloud package isn't available or input is empty."""
    if not texts:
        return None
    try:
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
    except ImportError:
        return None

    font_path = None
    if language in ("am", "ti", "om"):
        if font_dir is None:
            font_dir = pathlib.Path(__file__).parent.parent / "static"
        candidate = font_dir / "NotoSansEthiopic-Regular.ttf"
        if candidate.exists():
            font_path = str(candidate)

    cloud = WordCloud(
        width=900,
        height=420,
        background_color="white",
        font_path=font_path,
        colormap="viridis",
    ).generate(" ".join(texts))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    return fig
