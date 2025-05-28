from youtube_transcript_api import YouTubeTranscriptApi
import re

def get_video_id(youtube_url: str) -> str:
    match = re.search(r"(?:v=|youtu\\.be/)([a-zA-Z0-9_-]{11})", youtube_url)
    return match.group(1) if match else None

def fetch_transcript(youtube_url: str) -> str:
    video_id = get_video_id(youtube_url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([entry['text'] for entry in transcript])