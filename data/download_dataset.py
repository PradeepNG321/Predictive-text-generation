# data/download_dataset.py
from datasets import load_dataset
from pathlib import Path
import random
import re

OUT_PATH = Path("data/corpus.txt")
TRAIN_PATH = Path("data/train.txt")
TEST_PATH = Path("data/test.txt")

def clean_text(s):
    s = s.strip()
    # basic cleanup: keep letters, numbers, punctuation
    s = re.sub(r'[^A-Za-z0-9\s\.\,\?\!\:\;\-\']', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def main():
    print("📥 Downloading WikiText-2 (small English corpus)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    # combine train/validation/test splits
    texts = []
    for split in ["train", "validation", "test"]:
        for t in ds[split]["text"]:
            line = clean_text(t)
            if len(line.split()) > 3:
                texts.append(line)

    print("✅ Downloaded", len(texts), "lines")

    # shuffle + deduplicate
    texts = list(dict.fromkeys(texts))
    random.shuffle(texts)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf8") as f:
        for l in texts:
            f.write(l + "\n")
    print("💾 Saved full corpus to", OUT_PATH)

    # split 90/10 train/test
    n = len(texts)
    cut = int(0.9 * n)
    with TRAIN_PATH.open("w", encoding="utf8") as f:
        f.write("\n".join(texts[:cut]))
    with TEST_PATH.open("w", encoding="utf8") as f:
        f.write("\n".join(texts[cut:]))
    print(f"✂️  Split into train:{cut} / test:{n-cut}")

if __name__ == "__main__":
    main()
