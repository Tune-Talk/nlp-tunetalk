import re


def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
