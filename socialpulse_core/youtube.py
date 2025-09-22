from googleapiclient.discovery import build
import pandas as pd
import streamlit as st

# Load API key from Streamlit secrets
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

def get_youtube_comments(video_url: str, api_key: str = YOUTUBE_API_KEY, max_comments: int = 100) -> pd.DataFrame:
    """Fetches top-level comments from a public YouTube video."""

    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL. Please check and try again.")
            return pd.DataFrame()

        youtube = build("youtube", "v3", developerKey=api_key)
        comments = []
        next_page_token = None

        while len(comments) < max_comments:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText",
            ).execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return pd.DataFrame({"text": comments})

    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return pd.DataFrame()

def extract_video_id(url: str) -> str:
    """Extracts the video ID from a YouTube URL."""
    try:
        if "v=" in url:
            return url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[-1].split("?")[0]
        return ""
    except:
        return ""
