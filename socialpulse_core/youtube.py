from urllib import response
from googleapiclient.discovery import build
from langdetect import detect
import pandas as pd
import streamlit as st

# Load API key from secrets.toml
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def get_youtube_comments(video_url: str, api_key: str, max_comments: int = 100):
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
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
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        return pd.DataFrame({"text": comments})
    except Exception as e:
        print(f"Error fetching comments: {e}")
        return pd.DataFrame()
    

    