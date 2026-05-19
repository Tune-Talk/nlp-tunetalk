#!/usr/bin/env python3
"""Import a trending Spotify dataset into TuneTalk's cleaned playlist CSV.

The current backend only needs track metadata, mood labels, and Spotify URLs.
This script accepts a downloaded Kaggle CSV/ZIP, normalizes common Spotify
dataset column names, converts rows into the existing cleaned schema, then
merges them into dataset-cleaning/cleaned_spotify_lyrics.csv.
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "dataset-raw"
CLEANED_PATH = BASE_DIR / "dataset-cleaning" / "cleaned_spotify_lyrics.csv"
DEFAULT_OUTPUT_PATH = CLEANED_PATH
DEFAULT_WESTERN_COUNTRIES = ["US", "CA", "AU", "NZ", "IE"]

OUTPUT_COLUMNS = [
    "song_id_clean",
    "title",
    "artists",
    "genres_clean",
    "mood_category",
    "emotion_label",
    "clean_text",
    "final_text",
    "spotify_url",
]

GENRE_MOOD_MAP = {
    "indie": "melancholic",
    "folk": "melancholic",
    "acoustic": "melancholic",
    "singer-songwriter": "melancholic",
    "emo": "melancholic",
    "blues": "melancholic",
    "country": "melancholic",
    "soul": "melancholic",
    "sad": "melancholic",
    "melancholic": "melancholic",
    "rock": "intense",
    "metal": "intense",
    "punk": "intense",
    "hard rock": "intense",
    "hardcore": "intense",
    "hip hop": "intense",
    "hip-hop": "intense",
    "rap": "intense",
    "trap": "intense",
    "drill": "intense",
    "gangsta": "intense",
    "conscious": "intense",
    "pop": "euphoric",
    "dance": "euphoric",
    "funk": "euphoric",
    "edm": "euphoric",
    "electronic": "euphoric",
    "disco": "euphoric",
    "house": "euphoric",
    "k-pop": "euphoric",
    "tropical": "euphoric",
    "upbeat": "euphoric",
    "ambient": "tense",
    "lo-fi": "tense",
    "lofi": "tense",
    "dark": "tense",
    "gothic": "tense",
    "post-punk": "tense",
    "alternative": "tense",
    "experimental": "tense",
    "industrial": "tense",
    "jazz": "peaceful",
    "classical": "peaceful",
    "bossa nova": "peaceful",
    "new age": "peaceful",
    "meditation": "peaceful",
    "chill": "peaceful",
    "r&b": "peaceful",
    "neo soul": "peaceful",
    "easy listening": "peaceful",
    "gospel": "peaceful",
    "soft rock": "peaceful",
}

MOOD_TO_EMOTION = {
    "melancholic": "sadness",
    "intense": "anger",
    "euphoric": "joy",
    "tense": "anxiety",
    "peaceful": "calm",
}

NON_ENGLISH_ARTIST_KEYWORDS = [
    "aleman",
    "anitta",
    "bad bunny",
    "fuerza regida",
    "jennie",
    "jimin",
    "jin",
    "karan aujla",
    "karol g",
    "myke towers",
    "neton vega",
    "oscar maydon",
    "peso pluma",
    "rauw alejandro",
    "rose",
    "tito double p",
    "yuki chiba",
]

NON_ENGLISH_TITLE_KEYWORDS = [
    "adivino",
    "baile inolvidable",
    "bokete",
    "cafe con ron",
    "dtmf",
    "el club",
    "eoo",
    "ketu tecre",
    "kloufrens",
    "l'amour",
    "la mudanza",
    "la patrulla",
    "lo que le paso",
    "me prometi",
    "nuevayol",
    "ojos tristes",
    "perfumito nuevo",
    "pitorro de coco",
    "que pasaria",
    "si antes te hubiera conocido",
    "te queria ver",
    "tu boda",
    "tu sancho",
    "turista",
    "una velita",
    "velda",
    "voy a llevarte",
    "weltita",
]

CJK_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")

COLUMN_ALIASES = {
    "track_id": [
        "spotify_id",
        "track_id",
        "song_id",
        "id",
        "track_uri",
        "uri",
        "spotify_uri",
        "spotify_url",
        "url",
    ],
    "title": ["name", "track_name", "track", "title", "song"],
    "artists": [
        "artists",
        "artist",
        "artist_name",
        "artist_names",
        "artist_s_name",
        "artist(s)_name",
        "artistnames",
    ],
    "genres": ["genres", "genre", "track_genre", "artist_genres", "playlist_genre"],
    "lyrics": ["lyrics", "lyric", "text"],
    "country": ["country", "region"],
    "snapshot_date": ["snapshot_date", "date", "chart_date"],
    "rank": ["daily_rank", "rank", "position"],
    "popularity": ["popularity", "track_popularity"],
    "release_date": ["album_release_date", "release_date", "track_release_date"],
    "valence": ["valence"],
    "energy": ["energy", "danceability"],
    "acousticness": ["acousticness"],
}


def normalize_column_name(name: object) -> str:
    normalized = str(name).strip().lower()
    normalized = normalized.replace("%", "percent")
    normalized = re.sub(r"[\s\-/]+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_()]+", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def normalize_typography(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("®", "")
    )


def ascii_fold(value: object) -> str:
    text = normalize_typography(value)
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    folded = re.sub(r"\s+", " ", folded).strip().lower()
    return folded


def is_likely_english_western(title: object, artists: object) -> bool:
    raw_title = "" if pd.isna(title) else str(title)
    raw_artists = "" if pd.isna(artists) else str(artists)
    if CJK_PATTERN.search(raw_title) or CJK_PATTERN.search(raw_artists):
        return False

    normalized_title = normalize_typography(raw_title)
    if normalized_title.encode("ascii", "ignore").decode("ascii") != normalized_title:
        return False

    title_text = ascii_fold(raw_title)
    artist_text = ascii_fold(raw_artists)

    if any(keyword in artist_text for keyword in NON_ENGLISH_ARTIST_KEYWORDS):
        return False
    if any(keyword in title_text for keyword in NON_ENGLISH_TITLE_KEYWORDS):
        return False

    return True


def normalize_track_id(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return ""
    match = re.search(r"(?:spotify:track:|open\.spotify\.com/track/)([A-Za-z0-9]+)", text)
    if match:
        return match.group(1)
    text = re.sub(r"\?.*$", "", text)
    return text.replace("spotify:track:", "").strip()


def clean_artist(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return ""
    text = re.sub(r"[{}\[\]'\"]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,")


def clean_genres(value: object, country: object = "") -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text and text.lower() not in {"nan", "none", "null"}:
        parts = re.split(r"[;,|]", text)
        genres = [re.sub(r"\s+", " ", part).strip().lower() for part in parts]
        genres = [genre for genre in genres if genre]
        if genres:
            return "; ".join(dict.fromkeys(genres[:3]))

    country_text = "" if pd.isna(country) else str(country).strip()
    if country_text:
        return f"spotify chart; {country_text.lower()} top 50"
    return "spotify chart"


def clean_lyrics(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"Translations\s+[A-Za-z\s]+", " ", text)
    text = re.sub(r"You might also like", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\d*Embed$", " ", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def assign_mood_from_genres(genres: str) -> str:
    genre_list = [item.strip().lower() for item in str(genres).split(";") if item.strip()]
    scores: Counter[str] = Counter()
    for genre in genre_list:
        for keyword, mood in GENRE_MOOD_MAP.items():
            if keyword in genre:
                scores[mood] += 1
    if not scores:
        return "unknown"
    return scores.most_common(1)[0][0]


def normalize_feature(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    numeric = float(numeric)
    if numeric > 1:
        numeric = numeric / 100
    return max(0.0, min(1.0, numeric))


def assign_mood(
    row: pd.Series,
    genres_col: str | None,
    valence_col: str | None,
    energy_col: str | None,
    acousticness_col: str | None,
) -> str:
    valence = normalize_feature(row[valence_col]) if valence_col else None
    energy = normalize_feature(row[energy_col]) if energy_col else None
    acousticness = normalize_feature(row[acousticness_col]) if acousticness_col else None

    if valence is not None and energy is not None:
        if valence < 0.5 and energy < 0.5:
            return "melancholic"
        if valence < 0.5 and energy >= 0.7:
            return "tense"
        if valence < 0.5:
            return "intense"
        if energy >= 0.5:
            return "euphoric"
        return "peaceful"

    if valence is not None and acousticness is not None:
        if valence < 0.5:
            return "melancholic" if acousticness >= 0.5 else "intense"
        return "peaceful" if acousticness >= 0.5 else "euphoric"

    if genres_col:
        return assign_mood_from_genres(row[genres_col])
    return "unknown"


def find_column(columns: Iterable[str], logical_name: str) -> str | None:
    available = set(columns)
    for alias in COLUMN_ALIASES[logical_name]:
        normalized_alias = normalize_column_name(alias)
        if normalized_alias in available:
            return normalized_alias
    return None


def extract_source_file(path: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path

    temp_dir = Path(tempfile.mkdtemp(prefix="tunetalk_spotify_"))
    with zipfile.ZipFile(path) as archive:
        csv_members = [member for member in archive.namelist() if member.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError(f"No CSV file found inside {path}")

        member_infos = [archive.getinfo(member) for member in csv_members]
        largest_csv = max(member_infos, key=lambda info: info.file_size)
        archive.extract(largest_csv, temp_dir)
        return temp_dir / largest_csv.filename


def read_source(path: Path) -> pd.DataFrame:
    csv_path = extract_source_file(path)
    return pd.read_csv(csv_path, low_memory=False)


def pick_latest_snapshots(df: pd.DataFrame, snapshot_col: str | None, latest_days: int) -> pd.DataFrame:
    if not snapshot_col:
        return df
    parsed = pd.to_datetime(df[snapshot_col], errors="coerce")
    if parsed.notna().sum() == 0:
        return df
    if latest_days <= 0:
        return df
    latest_dates = sorted(parsed.dropna().dt.normalize().unique())[-latest_days:]
    return df[parsed.dt.normalize().isin(latest_dates)].copy()


def sort_trending(df: pd.DataFrame, rank_col: str | None, popularity_col: str | None) -> pd.DataFrame:
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if rank_col:
        df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce")
        sort_cols.append(rank_col)
        ascending.append(True)
    if popularity_col:
        df[popularity_col] = pd.to_numeric(df[popularity_col], errors="coerce")
        sort_cols.append(popularity_col)
        ascending.append(False)
    if sort_cols:
        return df.sort_values(sort_cols, ascending=ascending, na_position="last")
    return df


def filter_release_year(df: pd.DataFrame, release_col: str | None, min_release_year: int | None) -> pd.DataFrame:
    if not release_col or not min_release_year:
        return df

    release_dates = pd.to_datetime(df[release_col], errors="coerce")
    return df[release_dates.dt.year >= min_release_year].copy()


def filter_countries(df: pd.DataFrame, country_col: str | None, countries: list[str]) -> pd.DataFrame:
    if not countries or not country_col:
        return df

    allowed = {country.strip().upper() for country in countries if country.strip()}
    if not allowed:
        return df

    country_values = df[country_col].fillna("GLOBAL").astype(str).str.strip().str.upper()
    return df[country_values.isin(allowed)].copy()


def convert_to_cleaned_schema(
    source: pd.DataFrame,
    max_tracks: int,
    countries: list[str],
    min_release_year: int | None,
    latest_days: int,
    english_only: bool,
) -> pd.DataFrame:
    df = source.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]

    track_col = find_column(df.columns, "track_id")
    title_col = find_column(df.columns, "title")
    artists_col = find_column(df.columns, "artists")
    genres_col = find_column(df.columns, "genres")
    lyrics_col = find_column(df.columns, "lyrics")
    country_col = find_column(df.columns, "country")
    snapshot_col = find_column(df.columns, "snapshot_date")
    rank_col = find_column(df.columns, "rank")
    popularity_col = find_column(df.columns, "popularity")
    release_col = find_column(df.columns, "release_date")
    valence_col = find_column(df.columns, "valence")
    energy_col = find_column(df.columns, "energy")
    acousticness_col = find_column(df.columns, "acousticness")

    required = {"track id/url": track_col, "title": title_col, "artists": artists_col}
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(
            f"Missing required source columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = pick_latest_snapshots(df, snapshot_col, latest_days)
    df = filter_countries(df, country_col, countries)
    df = filter_release_year(df, release_col, min_release_year)
    if english_only:
        df = df[
            df.apply(lambda row: is_likely_english_western(row[title_col], row[artists_col]), axis=1)
        ].copy()
    df = sort_trending(df, rank_col, popularity_col)

    converted = pd.DataFrame()
    converted["song_id_clean"] = df[track_col].apply(normalize_track_id)
    converted["title"] = df[title_col].fillna("").astype(str).str.strip()
    converted["artists"] = df[artists_col].apply(clean_artist)

    countries = df[country_col] if country_col else pd.Series([""] * len(df), index=df.index)
    raw_genres = df[genres_col] if genres_col else pd.Series([""] * len(df), index=df.index)
    converted["genres_clean"] = [
        clean_genres(genre, country)
        for genre, country in zip(raw_genres, countries)
    ]

    converted["mood_category"] = df.apply(
        lambda row: assign_mood(row, genres_col, valence_col, energy_col, acousticness_col),
        axis=1,
    )
    converted["emotion_label"] = converted["mood_category"].map(MOOD_TO_EMOTION)

    if lyrics_col:
        converted["clean_text"] = df[lyrics_col].apply(clean_lyrics)
        converted["final_text"] = converted["clean_text"]
    else:
        converted["clean_text"] = ""
        converted["final_text"] = ""

    converted["spotify_url"] = "https://open.spotify.com/track/" + converted["song_id_clean"]
    converted = converted[OUTPUT_COLUMNS]

    converted = converted[
        (converted["song_id_clean"] != "")
        & (converted["title"] != "")
        & (converted["artists"] != "")
        & (converted["mood_category"] != "unknown")
        & converted["emotion_label"].notna()
    ].copy()
    converted = converted.drop_duplicates(subset=["song_id_clean"]).reset_index(drop=True)

    if max_tracks > 0:
        converted = converted.head(max_tracks).copy()

    return converted


def merge_with_existing(converted: pd.DataFrame, output_path: Path, replace_existing: bool) -> pd.DataFrame:
    if CLEANED_PATH.exists() and not replace_existing:
        existing = pd.read_csv(CLEANED_PATH, low_memory=False)
        for col in OUTPUT_COLUMNS:
            if col not in existing.columns:
                existing[col] = ""
        existing = existing[OUTPUT_COLUMNS]
        merged = pd.concat([existing, converted], ignore_index=True)
    else:
        merged = converted.copy()

    merged = merged.fillna("")
    merged = merged.drop_duplicates(subset=["song_id_clean"], keep="first")
    merged = merged.drop_duplicates(subset=["title", "artists", "spotify_url"], keep="first")
    merged = merged.reset_index(drop=True)

    backup_path = output_path.with_suffix(".backup.csv")
    if output_path.exists() and output_path == CLEANED_PATH:
        shutil.copy2(output_path, backup_path)

    merged.to_csv(output_path, index=False)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the downloaded trending Spotify CSV or Kaggle ZIP.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV output path. Defaults to the backend cleaned Spotify dataset.",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=5000,
        help="Maximum converted trending tracks to merge. Use 0 for no limit.",
    )
    parser.add_argument(
        "--countries",
        default="US,CA,AU,NZ,IE",
        help=(
            "Comma-separated country codes to keep after selecting the latest snapshot. "
            "Use GLOBAL for rows with empty country. Defaults to English-speaking Western charts."
        ),
    )
    parser.add_argument(
        "--min-release-year",
        type=int,
        default=2020,
        help="Drop tracks released before this year. Use 0 to allow older charting tracks.",
    )
    parser.add_argument(
        "--latest-days",
        type=int,
        default=365,
        help="Keep this many latest chart dates before deduping tracks. Use 1 for only today's chart.",
    )
    parser.add_argument(
        "--allow-non-english",
        action="store_true",
        help="Allow non-English tracks that chart in Western countries.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace the cleaned dataset instead of merging with the existing rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.source.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    source_df = read_source(source_path)
    countries = [country.strip() for country in args.countries.split(",") if country.strip()]
    min_release_year = args.min_release_year if args.min_release_year > 0 else None
    converted = convert_to_cleaned_schema(
        source_df,
        args.max_tracks,
        countries,
        min_release_year,
        args.latest_days,
        not args.allow_non_english,
    )
    if converted.empty:
        raise ValueError("No usable rows after conversion.")

    before_count = len(pd.read_csv(CLEANED_PATH, low_memory=False)) if CLEANED_PATH.exists() else 0
    merged = merge_with_existing(converted, output_path, args.replace_existing)
    added_count = len(merged) - before_count if output_path == CLEANED_PATH and not args.replace_existing else len(converted)

    print(f"Source rows      : {len(source_df):,}")
    print(f"Country filter   : {', '.join(countries) if countries else 'none'}")
    print(f"Min release year : {min_release_year or 'none'}")
    print(f"Latest days      : {args.latest_days if args.latest_days > 0 else 'all'}")
    print(f"English only     : {not args.allow_non_english}")
    print(f"Converted rows   : {len(converted):,}")
    print(f"Output rows      : {len(merged):,}")
    print(f"Estimated added  : {max(added_count, 0):,}")
    print("Mood distribution:")
    print(merged["mood_category"].value_counts().to_string())
    print(f"Saved to         : {output_path}")


if __name__ == "__main__":
    main()
