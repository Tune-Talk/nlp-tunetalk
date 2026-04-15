"""
services.py
-----------
Business logic layer. Contains:
  - EmotionService   : stub that the NLP engineer (Person 1) will fill in
  - ResponseService  : retrieval-based empathetic response selection
  - PlaylistService  : Valence-Arousal mood mapping + Spotify API call
  - ChatService      : orchestrates the full pipeline & persists to MongoDB
"""

import os
import random
import logging
from datetime import datetime, timezone

import requests
from pymongo.database import Database

from models import MOOD_MAPPING, chat_history_schema, serialize

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMOTION SERVICE  (stub — Person 1 replaces this with the real NLP model)
# ─────────────────────────────────────────────────────────────────────────────

class EmotionService:
    """
    Detects the dominant emotion from a raw text string.

    STUB BEHAVIOUR
    --------------
    Returns a mock result so the API works end-to-end even before
    the NLP model is integrated. Person 1 should replace `detect()`
    with the real RoBERTa / IndoBERT inference call.
    """

    # Placeholder mapping used only in stub mode
    _STUB_EMOTIONS = [
        {"label": "sadness",  "score": 0.87, "valence": -0.75, "arousal": -0.40},
        {"label": "anger",    "score": 0.80, "valence": -0.70, "arousal":  0.70},
        {"label": "joy",      "score": 0.90, "valence":  0.80, "arousal":  0.60},
        {"label": "anxiety",  "score": 0.75, "valence": -0.50, "arousal":  0.55},
        {"label": "calm",     "score": 0.70, "valence":  0.45, "arousal": -0.30},
    ]

    def detect(self, text: str) -> dict:
        """
        Parameters
        ----------
        text : preprocessed or raw user message

        Returns
        -------
        dict with keys:
            dominant_emotion, confidence, secondary_emotion,
            valence, arousal, all_emotions
        """
        # ── TODO (Person 1): replace stub with real model inference ──────────
        # Example:
        #   from transformers import pipeline
        #   classifier = pipeline("text-classification", model="your-model-path")
        #   results = classifier(text, top_k=3)
        #   ...map results to the return format below...
        # ─────────────────────────────────────────────────────────────────────

        # Stub: pick a random emotion to simulate model output
        primary   = random.choice(self._STUB_EMOTIONS)
        secondary = random.choice([e for e in self._STUB_EMOTIONS if e["label"] != primary["label"]])

        return {
            "dominant_emotion":  primary["label"],
            "confidence":        primary["score"],
            "secondary_emotion": secondary["label"],
            "valence":           primary["valence"],
            "arousal":           primary["arousal"],
            "all_emotions": [
                {"label": primary["label"],   "score": primary["score"]},
                {"label": secondary["label"], "score": secondary["score"]},
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. RESPONSE SERVICE  (retrieval-based empathetic response)
# ─────────────────────────────────────────────────────────────────────────────

class ResponseService:
    """
    Returns an empathetic support response based on the detected emotion.

    STUB BEHAVIOUR
    --------------
    Uses a small hand-crafted response pool per emotion. Person 1 should
    replace this with a proper TF-IDF / BM25 retrieval over EmpatheticDialogues.
    """

    _RESPONSES = {
        "sadness": [
            "It sounds like you're carrying a lot right now. It's okay to feel sad — your feelings are completely valid.",
            "I'm really sorry you're going through this. Sometimes life can feel heavy, and it's okay to take it one step at a time.",
            "Feeling sad is a natural response to difficult situations. Be kind to yourself — you don't have to have it all together right now.",
        ],
        "anger": [
            "It makes sense that you're feeling frustrated. Your emotions are telling you something important.",
            "That sounds really infuriating. It's okay to feel angry — what matters is how you process it.",
            "I hear how upset you are. Sometimes things just aren't fair, and it's valid to feel angry about that.",
        ],
        "joy": [
            "It's wonderful to hear that you're feeling great! Hold onto that positive energy.",
            "That's amazing! Happiness looks good on you — enjoy every moment of it.",
            "So glad things are going well! Celebrate yourself — you deserve it.",
        ],
        "anxiety": [
            "I can hear that you're feeling overwhelmed. Let's take a breath together — things will be okay.",
            "Anxiety can feel so consuming. Try to focus on what you can control right now, one thing at a time.",
            "It's understandable to feel anxious. You're not alone in this — many people feel this way, and it does get better.",
        ],
        "calm": [
            "It's lovely that you're in a peaceful state of mind. Enjoy this moment of stillness.",
            "A calm mind is a wonderful thing. Use this energy to reflect and recharge.",
            "Great that you're feeling settled. This is a good time to focus on things that matter to you.",
        ],
    }

    _DEFAULT = [
        "I'm here for you. Whatever you're feeling, know that it's okay.",
        "Thank you for sharing. Your feelings matter, and I'm listening.",
    ]

    def get_response(self, emotion_label: str) -> str:
        """
        Parameters
        ----------
        emotion_label : dominant emotion string, e.g. "sadness"

        Returns
        -------
        A single empathetic response string.
        """
        # ── TODO (Person 1): replace with TF-IDF / BM25 retrieval ───────────
        pool = self._RESPONSES.get(emotion_label, self._DEFAULT)
        return random.choice(pool)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PLAYLIST SERVICE  (Valence-Arousal mapping + Spotify)
# ─────────────────────────────────────────────────────────────────────────────

class PlaylistService:
    """
    Maps a detected emotion (via Valence-Arousal) to a mood category,
    then retrieves a playlist from Spotify API.
    """

    # Spotify credentials — loaded from environment variables
    SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    SPOTIFY_TOKEN_URL     = "https://accounts.spotify.com/api/token"
    SPOTIFY_SEARCH_URL    = "https://api.spotify.com/v1/search"

    def _get_access_token(self) -> str | None:
        """Fetch a short-lived Spotify Bearer token using Client Credentials flow."""
        if not self.SPOTIFY_CLIENT_ID or not self.SPOTIFY_CLIENT_SECRET:
            logger.warning("Spotify credentials not set. Returning mock playlist.")
            return None

        try:
            resp = requests.post(
                self.SPOTIFY_TOKEN_URL,
                data={"grant_type": "client_credentials"},
                auth=(self.SPOTIFY_CLIENT_ID, self.SPOTIFY_CLIENT_SECRET),
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("access_token")
        except requests.RequestException as e:
            logger.error("Spotify token error: %s", e)
            return None

    def _map_to_mood(self, emotion_label: str, valence: float, arousal: float) -> dict:
        """
        Use Valence-Arousal scores to find the best matching mood config.
        Falls back to emotion_label matching if VA scores are out of all ranges.
        """
        # Try exact VA range match first
        for mapping in MOOD_MAPPING:
            v_min, v_max = mapping["valence_range"]
            a_min, a_max = mapping["arousal_range"]
            if v_min <= valence <= v_max and a_min <= arousal <= a_max:
                return mapping

        # Fallback: match by emotion label
        for mapping in MOOD_MAPPING:
            if mapping["emotion_label"] == emotion_label:
                return mapping

        # Final fallback: peaceful
        return MOOD_MAPPING[-1]

    def _fetch_from_spotify(self, mood_config: dict, limit: int = 5) -> list[dict]:
        """Query Spotify for tracks matching the mood's recommended genres."""
        token = self._get_access_token()
        if not token:
            return self._mock_songs(mood_config)

        genre  = random.choice(mood_config["recommended_genres"])
        query  = f"genre:{genre}"
        params = {"q": query, "type": "track", "limit": limit, "market": "ID"}
        headers = {"Authorization": f"Bearer {token}"}

        try:
            resp = requests.get(self.SPOTIFY_SEARCH_URL, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            items = resp.json().get("tracks", {}).get("items", [])

            songs = []
            for item in items:
                album  = item.get("album", {})
                images = album.get("images", [{}])
                songs.append({
                    "song_id":    item.get("id", ""),
                    "title":      item.get("name", ""),
                    "artist":     ", ".join(a["name"] for a in item.get("artists", [])),
                    "genre":      genre,
                    "mood_tag":   mood_config["mood_category"],
                    "spotify_url": item.get("external_urls", {}).get("spotify", ""),
                    "cover_image": images[0].get("url", "") if images else "",
                })
            return songs

        except requests.RequestException as e:
            logger.error("Spotify search error: %s", e)
            return self._mock_songs(mood_config)

    def _mock_songs(self, mood_config: dict) -> list[dict]:
        """Return placeholder songs when Spotify is unavailable (dev/testing)."""
        genre = random.choice(mood_config["recommended_genres"])
        mood  = mood_config["mood_category"]
        return [
            {
                "song_id":    f"mock_{i}",
                "title":      f"Mock Song {i}",
                "artist":     "Mock Artist",
                "genre":      genre,
                "mood_tag":   mood,
                "spotify_url": "",
                "cover_image": "",
            }
            for i in range(1, 6)
        ]

    def get_playlist(self, emotion_label: str, valence: float, arousal: float) -> dict:
        """
        Main entry point.

        Returns
        -------
        dict with mood_category, songs list, and total_songs count.
        """
        mood_config = self._map_to_mood(emotion_label, valence, arousal)
        songs       = self._fetch_from_spotify(mood_config)

        return {
            "mood_category": mood_config["mood_category"],
            "songs":         songs,
            "total_songs":   len(songs),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHAT SERVICE  (orchestrator — ties everything together)
# ─────────────────────────────────────────────────────────────────────────────

class ChatService:
    """
    Orchestrates the full pipeline:
      1. Detect emotion
      2. Get empathetic response
      3. Get playlist
      4. Save to MongoDB
      5. Return packaged response dict
    """

    def __init__(self, db: Database):
        self.db               = db
        self.emotion_service  = EmotionService()
        self.response_service = ResponseService()
        self.playlist_service = PlaylistService()

    def process(self, user_id: str, message: str) -> dict:
        """
        Parameters
        ----------
        user_id : unique user identifier from request
        message : raw rant text from the user

        Returns
        -------
        Full response dict matching the API response schema.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Step 1 — Emotion detection
        emotion_result = self.emotion_service.detect(message)

        # Step 2 — Empathetic response
        support_text = self.response_service.get_response(
            emotion_result["dominant_emotion"]
        )

        # Step 3 — Playlist
        playlist = self.playlist_service.get_playlist(
            emotion_label=emotion_result["dominant_emotion"],
            valence=emotion_result["valence"],
            arousal=emotion_result["arousal"],
        )

        # Step 4 — Persist to MongoDB
        doc = chat_history_schema(
            user_id=user_id,
            raw_message=message,
            emotion_result=emotion_result,
            support_response=support_text,
            playlist=playlist,
        )
        try:
            self.db.chat_history.insert_one(doc)
        except Exception as e:
            logger.error("MongoDB insert error: %s", e)

        # Step 5 — Build API response
        return {
            "user_id": user_id,
            "timestamp": timestamp,
            "emotion": {
                "label":            emotion_result["dominant_emotion"],
                "confidence":       emotion_result["confidence"],
                "secondary_emotion": emotion_result.get("secondary_emotion"),
            },
            "support_response": {
                "text": support_text
            },
            "playlist": playlist,
        }