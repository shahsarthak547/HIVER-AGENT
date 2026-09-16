import html
import re

def normalize_text(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\b\d{4,}\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()