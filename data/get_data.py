"""Download the UCI SECOM dataset into data/raw/."""
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom"
FILES = ["secom.data", "secom_labels.data"]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = RAW / name
        if dst.exists():
            print(f"[skip] {name} already exists")
            continue
        url = f"{BASE}/{name}"
        print(f"[get ] {url}")
        urllib.request.urlretrieve(url, dst)
    print(f"[done] files in {RAW}")


if __name__ == "__main__":
    main()
