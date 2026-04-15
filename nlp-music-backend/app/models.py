"""
models.py
---------
MongoDB collection schemas and helper functions.
Collections:
  - chat_history   : stores every chat session message
  - songs          : mood-tagged song pool (seed data / Spotify cache)
"""

from datetime import datetime, timezone
from bson import ObjectId


# ── Helpers ──────────────────────────────────────────────────────────────────

def serialize(doc: dict) -> dict:
    """Convert MongoDB ObjectId to string so the document is JSON-serialisable."""
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


# ── Chat History Schema ───────────────────────────────────────────────────────

def chat_history_schema(
    user_id: str,
    raw_message: str,
    emotion_result: dict,
    support_response: str,
    playlist: dict,
) -> dict:
    """
    Build a chat history document to insert into MongoDB.

    Parameters
    ----------
    user_id         : unique identifier for the user
    raw_message     : the original rant text sent by the user
    emotion_result  : dict returned by the NLP emotion detection service
    support_response: empathetic reply text
    playlist        : dict containing mood_category + list of songs
    """
    return {
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_message": raw_message,
        "emotion": {
            "label": emotion_result.get("dominant_emotion"),
            "confidence": emotion_result.get("confidence", 0.0),
            "secondary_emotion": emotion_result.get("secondary_emotion"),
            "valence": emotion_result.get("valence"),
            "arousal": emotion_result.get("arousal"),
        },
        "support_response": support_response,
        "playlist": playlist,
    }


# ── Song Schema ───────────────────────────────────────────────────────────────

def song_schema(
    title: str,
    artist: str,
    genre: str,
    mood_tag: str,
    spotify_url: str,
    cover_image: str = "",
    valence: float = 0.0,
    arousal: float = 0.0,
) -> dict:
    """
    Build a song document to insert into the songs collection.
    Used when seeding the database or caching Spotify results.
    """
    return {
        "title": title,
        "artist": artist,
        "genre": genre,
        "mood_tag": mood_tag,       # e.g. "melancholic", "euphoric", "intense"
        "spotify_url": spotify_url,
        "cover_image": cover_image,
        "valence": valence,         # numeric score -1.0 → 1.0
        "arousal": arousal,         # numeric score -1.0 → 1.0
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Mood Mapping (Rule-Based Config) ─────────────────────────────────────────

MOOD_MAPPING = [
    {
        "emotion_label": "sadness",
        "valence_range": [-1.0, -0.5],
        "arousal_range": [-1.0,  0.0],
        "mood_category": "melancholic",
        "recommended_genres": ["indie", "folk", "acoustic"],
        "tempo_range": "slow",
    },
    {
        "emotion_label": "anger",
        "valence_range": [-1.0, -0.3],
        "arousal_range": [ 0.5,  1.0],
        "mood_category": "intense",
        "recommended_genres": ["rock", "metal", "hip-hop"],
        "tempo_range": "fast",
    },
    {
        "emotion_label": "joy",
        "valence_range": [ 0.5,  1.0],
        "arousal_range": [ 0.3,  1.0],
        "mood_category": "euphoric",
        "recommended_genres": ["pop", "dance", "funk"],
        "tempo_range": "fast",
    },
    {
        "emotion_label": "anxiety",
        "valence_range": [-0.8, -0.2],
        "arousal_range": [ 0.3,  0.8],
        "mood_category": "tense",
        "recommended_genres": ["ambient", "lo-fi", "classical"],
        "tempo_range": "slow",
    },
    {
        "emotion_label": "calm",
        "valence_range": [ 0.2,  0.7],
        "arousal_range": [-1.0,  0.2],
        "mood_category": "peaceful",
        "recommended_genres": ["ambient", "jazz", "classical"],
        "tempo_range": "slow",
    },
]