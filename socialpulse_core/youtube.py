"""YouTube comment extraction.

Fetches top-level comments from a public YouTube video and returns a DataFrame
enriched with metadata needed for downstream sentiment analysis and the
comment-galaxy visualization (timestamps, like counts, author names).
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from googleapiclient.discovery import build


_YT_URL_PATTERNS = [
    re.compile(r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([0-9A-Za-z_-]{11})"),
]


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract an 11-character video ID from a URL or return the input if it
    already looks like a bare ID. Returns None if nothing matches."""
    if not url_or_id:
        return None

    candidate = url_or_id.strip()

    # Bare ID
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", candidate):
        return candidate

    for pattern in _YT_URL_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return match.group(1)

    return None


def get_youtube_comments(
    video_url: str,
    api_key: str,
    max_comments: int = 150,
) -> pd.DataFrame:
    """Fetch top-level comments from a public YouTube video.

    Returns a DataFrame with columns:
        - text         : str   (comment body, plain text)
        - published_at : datetime (UTC)
        - like_count   : int
        - author       : str   (display name)

    Returns an empty DataFrame on failure. Errors are raised to the caller so
    the UI layer can decide how to surface them.
    """
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError(f"Could not extract a video ID from: {video_url!r}")

    youtube = build("youtube", "v3", developerKey=api_key)

    rows: list[dict] = []
    next_page_token: Optional[str] = None

    while len(rows) < max_comments:
        page_size = min(100, max_comments - len(rows))
        response = (
            youtube.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                maxResults=page_size,
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance",
            )
            .execute()
        )

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            rows.append(
                {
                    "text": snippet.get("textDisplay", ""),
                    "published_at": snippet.get("publishedAt"),
                    "like_count": int(snippet.get("likeCount", 0) or 0),
                    "author": snippet.get("authorDisplayName", ""),
                }
            )

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        # Drop comments with no text after extraction
        df = df[df["text"].astype(str).str.strip().astype(bool)].reset_index(drop=True)

    return df
