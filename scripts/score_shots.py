"""Score a CSV of shots (shot-chart schema) with the saved model.

    python3 scripts/score_shots.py data/raw/shotdetail_2024.csv reports/scored_2024.csv [--model gbm_full]
Adds `xfg` (expected make probability) and `xpts` (expected points) columns.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from shotquality import config, data, features, model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"), ap.add_argument("output")
    ap.add_argument("--model", default="gbm_full", choices=["gbm_full", "gbm_geometry"])
    args = ap.parse_args()

    raw = pd.read_csv(args.input)
    data.validate(raw)  # refuse to score malformed data rather than emit silent garbage
    feats = features.add_features(raw)
    m = model.load(config.MODEL_DIR / f"{args.model}.joblib")
    feats["xfg"] = m.predict(feats)
    feats["xpts"] = feats["xfg"] * feats["pt_value"]
    feats.to_csv(args.output, index=False)
    print(f"scored {len(feats):,} shots -> {args.output}; mean xFG {feats.xfg.mean():.3f} vs actual {feats.SHOT_MADE_FLAG.mean():.3f}")


if __name__ == "__main__":
    main()
