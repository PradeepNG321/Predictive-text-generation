# data/preprocess.py
# Usage: run this script from project root: python data/preprocess.py
from pathlib import Path
import re
import html
import random

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
OUT_PATH = ROOT.parent / "data" / "corpus.txt"   # top-level data/corpus.txt
TRAIN_PATH = ROOT.parent / "data" / "train.txt"
TEST_PATH = ROOT.parent / "data" / "test.txt"

# basic cleaning steps for lines
def clean_line(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    # unescape html entities
    s = html.unescape(s)
    # remove URLs
    s = re.sub(r'https?://\S+', ' ', s)
    # remove email addresses
    s = re.sub(r'\S+@\S+', ' ', s)
    # remove weird characters, keep basic punctuation and letters/numbers
    s = re.sub(r'[^A-Za-z0-9\s\.\,\?\!\:\;\-\']', ' ', s)
    # normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # keep short lines out
    if len(s) < 3:
        return ""
    return s

def gather_raw_lines():
    lines = []
    # gather all .txt files in data/raw
    if not RAW_DIR.exists():
        print("No raw folder found at", RAW_DIR)
        return lines
    for p in RAW_DIR.glob("*.txt"):
        with p.open("r", encoding="utf8", errors="ignore") as f:
            raw = f.read().splitlines()
        for l in raw:
            cl = clean_line(l)
            if cl:
                lines.append(cl)
    return lines

def main():
    # If there is a top-level data/corpus_raw.txt, include it too
    corpus_raw = ROOT.parent / "data" / "corpus_raw.txt"
    all_lines = []

    # lines from data/raw/*.txt
    raw_lines = gather_raw_lines()
    all_lines.extend(raw_lines)

    # optional: include data/corpus_raw.txt if present
    if corpus_raw.exists():
        with corpus_raw.open("r", encoding="utf8", errors="ignore") as f:
            for l in f:
                cl = clean_line(l)
                if cl:
                    all_lines.append(cl)

    # fallback: if nothing found, keep the default small sample
    if not all_lines:
        print("No raw data found. Creating small sample corpus.")
        all_lines = [
            "hello how are you",
            "i am fine thank you",
            "this project is about predictive text",
            "predictive text suggestions help typing faster",
            "what is your name",
        ]

    # deduplicate and shuffle
    unique = list(dict.fromkeys(all_lines))  # preserve order then unique
    random.shuffle(unique)

    # Write full corpus
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf8") as fout:
        for l in unique:
            fout.write(l + "\n")
    print(f"Wrote {len(unique)} lines to {OUT_PATH}")

    # split into train/test (90/10)
    n = len(unique)
    cut = max(1, int(n * 0.9))
    train = unique[:cut]
    test = unique[cut:]
    with TRAIN_PATH.open("w", encoding="utf8") as f:
        for l in train:
            f.write(l + "\n")
    with TEST_PATH.open("w", encoding="utf8") as f:
        for l in test:
            f.write(l + "\n")
    print(f"Wrote train: {len(train)} lines to {TRAIN_PATH}")
    print(f"Wrote test: {len(test)} lines to {TEST_PATH}")

if __name__ == "__main__":
    main()
