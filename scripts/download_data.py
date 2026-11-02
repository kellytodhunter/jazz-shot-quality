"""Fetch the raw shot-chart CSVs (idempotent)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shotquality import data  # noqa: E402

if __name__ == "__main__":
    for p in data.download():
        print("ready:", p)
