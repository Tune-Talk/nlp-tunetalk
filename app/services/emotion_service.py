import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import EMOTION_MODEL_ID, EMOTION_LABELS, EMOTION_TO_VA, ID2LABEL, HF_TOKEN, MODEL_LABEL_TO_CANONICAL


class EmotionService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            EMOTION_MODEL_ID, token=HF_TOKEN or None
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            EMOTION_MODEL_ID, token=HF_TOKEN or None
        )
        self.model.eval()
        self.model.to("cpu")

    def detect(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits.squeeze(0)
        probs = torch.softmax(logits, dim=0).numpy()

        ranked = sorted(
            [(ID2LABEL[i], float(probs[i])) for i in range(len(EMOTION_LABELS))],
            key=lambda x: x[1],
            reverse=True,
        )

        dominant_label, dominant_conf = ranked[0]
        secondary_label, secondary_conf = ranked[1] if len(ranked) > 1 else (None, 0.0)

        canonical_dominant = MODEL_LABEL_TO_CANONICAL.get(dominant_label, dominant_label)
        canonical_secondary = (
            MODEL_LABEL_TO_CANONICAL.get(secondary_label, secondary_label)
            if secondary_label else None
        )

        va = EMOTION_TO_VA.get(canonical_dominant, {"valence": 0.0, "arousal": 0.0})

        return {
            "dominant_emotion": canonical_dominant,
            "confidence": dominant_conf,
            "secondary_emotion": canonical_secondary,
            "valence": va["valence"],
            "arousal": va["arousal"],
            "all_emotions": [
                {"label": MODEL_LABEL_TO_CANONICAL.get(label, label), "score": score}
                for label, score in ranked
            ],
        }
