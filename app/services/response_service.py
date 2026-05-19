import pickle
import random

import numpy as np
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import (
    SBERT_ENCODER_ID,
    SBERT_MODEL_ID,
    HF_TOKEN,
    EMOTION_BUCKET_MAP,
    FOLLOW_UP_PHRASES,
    MUSIC_MENTION_PHRASES,
    SIMILARITY_THRESHOLD,
)
from app.utils.preprocessing import clean_text


class ResponseService:
    _FALLBACK = {
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

    def __init__(self):
        self.model = None
        self.response_bank = {}
        self.assistant_replies = []
        self.reply_emotions = []
        self.embeddings = None

        self._load_model()
        if self.model is not None:
            self._load_data()
        self._build_response_bank()

    def _load_model(self):
        try:
            self.model = SentenceTransformer(
                SBERT_ENCODER_ID, token=HF_TOKEN or None, device="cpu"
            )
        except Exception:
            self.model = None

    def _load_data(self):
        try:
            texts_path = hf_hub_download(
                repo_id=SBERT_MODEL_ID,
                filename="response_texts.pkl",
                token=HF_TOKEN or None,
                repo_type="model",
            )
            embeddings_path = hf_hub_download(
                repo_id=SBERT_MODEL_ID,
                filename="response_embeddings.npy",
                token=HF_TOKEN or None,
                repo_type="model",
            )
            labels_path = hf_hub_download(
                repo_id=SBERT_MODEL_ID,
                filename="emotion_labels.pkl",
                token=HF_TOKEN or None,
                repo_type="model",
            )

            with open(texts_path, "rb") as f:
                self.assistant_replies = pickle.load(f)
            self.embeddings = np.load(embeddings_path)
            with open(labels_path, "rb") as f:
                self.reply_emotions = pickle.load(f)
        except Exception:
            pass

    def _build_response_bank(self):
        for label, responses in self._FALLBACK.items():
            self.response_bank[label] = responses

    def get_response(self, text: str, emotion_label: str) -> str:
        emotion_label = str(emotion_label).strip().lower()
        bucket = EMOTION_BUCKET_MAP.get(emotion_label, [emotion_label])

        retrieved = self._retrieve(text, bucket)
        music_phrase = MUSIC_MENTION_PHRASES.get(emotion_label, MUSIC_MENTION_PHRASES["calm"])
        follow_up = FOLLOW_UP_PHRASES.get(emotion_label, FOLLOW_UP_PHRASES["calm"])

        if retrieved:
            return f"{retrieved} {follow_up} {music_phrase}"

        return self._fallback_response(emotion_label, follow_up, music_phrase)

    def _retrieve(self, text: str, bucket: list) -> str | None:
        if self.model is None or self.embeddings is None or self.embeddings.shape[0] == 0:
            return None

        query_vec = self.model.encode([text], convert_to_numpy=True)
        scores = cosine_similarity(query_vec, self.embeddings)[0]

        bucket_indices = [
            i for i, e in enumerate(self.reply_emotions)
            if e and any(b in e for b in bucket)
        ]

        if not bucket_indices:
            bucket_indices = list(range(len(scores)))

        if not bucket_indices:
            return None

        best_idx = max(bucket_indices, key=lambda i: scores[i])
        best_score = scores[best_idx]

        if best_score < SIMILARITY_THRESHOLD:
            return None

        return self.assistant_replies[best_idx]

    def _fallback_response(self, emotion_label: str, follow_up: str, music_phrase: str) -> str:
        bank_responses = self.response_bank.get(emotion_label, [])
        if not bank_responses:
            bank_responses = self.response_bank.get(emotion_label.lower(), [])
        if not bank_responses:
            bank_responses = self._FALLBACK.get(emotion_label, self._DEFAULT)

        base = random.choice(bank_responses)
        return f"{base} {follow_up} {music_phrase}"
