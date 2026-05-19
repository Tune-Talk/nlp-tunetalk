from datetime import datetime, timezone

from app.services.emotion_service import EmotionService
from app.services.response_service import ResponseService
from app.services.playlist_service import PlaylistService
from app.utils.preprocessing import clean_text


class ChatService:
    MIN_WORDS = 5

    def __init__(self):
        self.emotion_service = EmotionService()
        self.response_service = ResponseService()
        self.playlist_service = PlaylistService()

    def process(self, user_id: str, message: str) -> dict:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message cannot be empty.")

        word_count = len(message.strip().split())
        if word_count < self.MIN_WORDS:
            raise ValueError(
                f"Input text is too short. Please write at least {self.MIN_WORDS} words."
            )

        cleaned = clean_text(message)
        emotion_result = self.emotion_service.detect(cleaned)

        support_text = self.response_service.get_response(
            message, emotion_result["dominant_emotion"]
        )

        playlist = self.playlist_service.get_playlist(
            emotion_result["dominant_emotion"], n=5
        )

        return {
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotion": {
                "label": emotion_result["dominant_emotion"],
                "confidence": emotion_result["confidence"],
                "secondary_emotion": emotion_result.get("secondary_emotion"),
            },
            "support_response": {
                "text": support_text,
            },
            "playlist": playlist,
        }
