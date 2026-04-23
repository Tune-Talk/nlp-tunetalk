import ast
import random
from typing import Dict, List, Optional

import pandas as pd


class MusicMapper:
    """Load the cleaned music dataset once and serve lightweight recommendations."""

    EMOTION_TO_VA = {
        "sadness": {"valence": -0.75, "arousal": -0.40},
        "anger": {"valence": -0.80, "arousal": 0.75},
        "joy": {"valence": 0.85, "arousal": 0.70},
        "anxiety": {"valence": -0.65, "arousal": 0.85},
        "calm": {"valence": 0.60, "arousal": -0.55},
    }

    EMOTION_TO_SONG_LABELS = {
        "sadness": ["sad", "lonely", "guilty", "disappointed", "devastated", "heartbroken"],
        "anger": ["angry", "furious", "annoyed", "disgusted"],
        "joy": ["joyful", "excited", "proud", "grateful", "hopeful", "happy"],
        "anxiety": ["anxious", "afraid", "terrified", "apprehensive", "nervous"],
        "calm": ["content", "peaceful", "calm", "confident", "prepared", "sentimental"],
    }

    REQUIRED_COLUMNS = [
        "title",
        "artists",
        "genres_clean",
        "mood_category",
        "emotion_label",
        "spotify_url",
    ]

    def __init__(self, csv_path: str, random_seed: int = 42) -> None:
        self.csv_path = csv_path
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)
        self.df = self._load_dataset(csv_path)

    def _load_dataset(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        work_df = df[self.REQUIRED_COLUMNS].copy()
        for col in self.REQUIRED_COLUMNS:
            work_df[col] = work_df[col].fillna("").astype(str).str.strip()

        work_df["mood_category"] = work_df["mood_category"].str.lower()
        work_df["emotion_label"] = work_df["emotion_label"].str.lower()

        work_df = work_df[
            (work_df["title"] != "")
            & (work_df["artists"] != "")
            & (work_df["spotify_url"] != "")
        ]
        work_df = work_df.drop_duplicates(
            subset=["title", "artists", "spotify_url"]
        ).reset_index(drop=True)

        if work_df.empty:
            raise ValueError("Dataset is empty after cleaning.")

        return work_df

    def _parse_genres(self, value: str) -> List[str]:
        value = str(value).strip()
        if not value:
            return []

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item).strip().lower() for item in parsed if str(item).strip()]
        except Exception:
            pass

        value = value.replace("|", ",").replace(";", ",")
        return [part.strip().lower() for part in value.split(",") if part.strip()]

    def emotion_to_mood(self, emotion_label: str) -> Dict[str, float]:
        emotion_label = str(emotion_label).strip().lower()
        if emotion_label not in self.EMOTION_TO_VA:
            raise ValueError(f"Unsupported emotion label: {emotion_label}")

        va = self.EMOTION_TO_VA[emotion_label]
        return {
            "emotion": emotion_label,
            "valence": va["valence"],
            "arousal": va["arousal"],
            "mood_category": self._valence_arousal_to_mood(va["valence"], va["arousal"]),
        }

    def _valence_arousal_to_mood(self, valence: float, arousal: float) -> str:
        if valence < 0 and arousal < 0:
            return "melancholic"
        if valence < 0 and arousal >= 0:
            if arousal >= 0.7:
                return "tense"
            return "intense"
        if valence >= 0 and arousal >= 0:
            return "euphoric"
        return "peaceful"

    def _allowed_song_emotions(self, emotion_label: str) -> List[str]:
        emotion_label = str(emotion_label).strip().lower()
        return self.EMOTION_TO_SONG_LABELS.get(emotion_label, [emotion_label])

    def recommend_songs(
        self,
        emotion_label: str,
        genre: Optional[str] = None,
        n_results: int = 5,
    ) -> Dict[str, object]:
        mood_info = self.emotion_to_mood(emotion_label)
        target_mood = mood_info["mood_category"]
        target_emotions = self._allowed_song_emotions(emotion_label)

        candidates = self.df[self.df["mood_category"] == target_mood].copy()

        if not candidates.empty:
            emotion_filtered = candidates[candidates["emotion_label"].isin(target_emotions)]
            if not emotion_filtered.empty:
                candidates = emotion_filtered

        if genre:
            genre = str(genre).strip().lower()
            genre_mask = candidates["genres_clean"].apply(
                lambda x: genre in self._parse_genres(x) or genre in str(x).lower()
            )
            genre_filtered = candidates[genre_mask].copy()
            if not genre_filtered.empty:
                candidates = genre_filtered

        if candidates.empty:
            candidates = self.df[self.df["mood_category"] == target_mood].copy()

        if candidates.empty:
            raise ValueError(f"No songs found for mood category: {target_mood}")

        sample_size = min(max(int(n_results), 1), len(candidates))
        random_state = self._rng.randint(0, 10_000)
        sampled = candidates.sample(sample_size, random_state=random_state)

        songs = []
        for _, row in sampled.iterrows():
            songs.append(
                {
                    "title": row["title"],
                    "artist": row["artists"],
                    "genre": row["genres_clean"],
                    "mood_category": row["mood_category"],
                    "spotify_url": row["spotify_url"],
                }
            )

        return {
            "input_emotion": str(emotion_label).strip().lower(),
            "valence": mood_info["valence"],
            "arousal": mood_info["arousal"],
            "mood_category": target_mood,
            "songs": songs,
        }

