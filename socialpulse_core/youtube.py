import requests
import pandas as pd

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def get_youtube_comments(video_id: str, api_key: str, max_comments: int = 100) -> pd.DataFrame:
    """Fetches comments from a YouTube video using the YouTube Data API."""
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None

    try:
        while len(comments) < max_comments:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                maxResults=100,
                pageToken=next_page_token
            ).execute()

            for item in response['items']:
                top_comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'author': top_comment['authorDisplayName'],
                    'text': top_comment['textDisplay'],
                    'likeCount': top_comment['likeCount'],
                    'publishedAt': top_comment['publishedAt']
                })

                if len(comments) >= max_comments:
                    break

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

    except HttpError as e:
        print(f"An HTTP error occurred: {e}")

    return pd.DataFrame(comments)
