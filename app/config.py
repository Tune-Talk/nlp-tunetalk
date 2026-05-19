import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")

EMOTION_MODEL_ID = "melisaolivia18/emotion-detection-roberta"
SBERT_MODEL_ID = "melisaolivia18/empathetic-retrieval-sbert"
SBERT_ENCODER_ID = "all-MiniLM-L6-v2"

# Must match the exact order in the model's config.json id2label.
# model: melisaolivia18/emotion-detection-roberta
EMOTION_LABELS = ["angry", "anxious", "content", "joyful", "sad"]

LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}

# Map model labels to canonical coarse labels used by EMOTION_TO_VA,
# EMOTION_BUCKET_MAP, and MUSIC_MENTION_PHRASES.
MODEL_LABEL_TO_CANONICAL = {
    "angry": "anger",
    "anxious": "anxiety",
    "content": "calm",
    "joyful": "joy",
    "sad": "sadness",
}

# Canonical labels for external API validation (playlist endpoint, etc.)
CANONICAL_LABELS = list(MODEL_LABEL_TO_CANONICAL.values())

EMOTION_TO_VA = {
    "sadness": {"valence": -0.75, "arousal": -0.40},
    "anger": {"valence": -0.80, "arousal": 0.75},
    "joy": {"valence": 0.85, "arousal": 0.70},
    "anxiety": {"valence": -0.65, "arousal": 0.85},
    "calm": {"valence": 0.60, "arousal": -0.55},
}

EMOTION_BUCKET_MAP = {
    "sadness": ["sad", "lonely", "disappointed", "devastated", "guilty", "embarrassed", "nostalgic"],
    "anger": ["angry", "furious", "annoyed", "disgusted", "jealous"],
    "joy": ["joyful", "excited", "proud", "grateful", "hopeful", "impressed", "caring", "trusting", "faithful", "enthusiastic", "surprised", "confident", "happy"],
    "anxiety": ["anxious", "afraid", "terrified", "apprehensive", "nervous"],
    "calm": ["content", "peaceful", "prepared", "sentimental"],
}

MUSIC_MENTION_PHRASES = {
    "sadness": "Here's some music that might help you process what you're feeling.",
    "anger": "Here are some tracks that might match your energy right now.",
    "joy": "Here's some music to keep the good vibes going!",
    "anxiety": "Here are some soothing tracks that might help calm your mind.",
    "calm": "Here's some music to complement your peaceful state of mind.",
}

FOLLOW_UP_PHRASES = {
    "sadness": "I hope things get a little easier for you soon.",
    "anger": "Your feelings are valid, and it is okay to feel upset.",
    "joy": "I hope this happy moment brings you even more beautiful experiences ahead.",
    "anxiety": "I hope you can take things one step at a time and be gentle with yourself.",
    "calm": "It is really nice to hear that you are feeling peaceful right now.",
}

SIMILARITY_THRESHOLD = 0.3

CSV_PATH = os.path.join(
    Path(__file__).resolve().parent.parent,
    "nlp preprocessing", "dataset-cleaning", "cleaned_spotify_lyrics.csv"
)
