import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from music_mapper import MusicMapper

from app.config import CSV_PATH


class PlaylistService:
    def __init__(self):
        self.mapper = MusicMapper(str(CSV_PATH))

    def get_playlist(self, emotion_label: str, n: int = 5) -> dict:
        raw = self.mapper.recommend_songs(emotion_label, n_results=n)
        songs = []
        for i, song in enumerate(raw["songs"]):
            songs.append({
                "song_id": f"spotify_{i + 1:03d}",
                "title": song.get("title", ""),
                "artist": song.get("artist", ""),
                "genre": song.get("genre", ""),
                "mood_tag": song.get("mood_category", raw.get("mood_category", "")),
                "spotify_url": song.get("spotify_url", ""),
                "cover_image": "",
            })
        return {
            "mood_category": raw["mood_category"],
            "songs": songs,
            "total_songs": len(songs),
        }
