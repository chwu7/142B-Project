"""
src/data/labels.py
OWNER: Person 1 (Data pipeline)

Converts continuous abnormal returns into 3-class labels and
produces the master index file used by the Dataset class.

Classes:
    0 = down  (abnormal_return < -FLAT_BAND)
    1 = flat  (-FLAT_BAND <= abnormal_return <= FLAT_BAND)
    2 = up    (abnormal_return > FLAT_BAND)

Usage:
    python src/data/labels.py
"""
import os
import pandas as pd
from src.utils.config import DATA_RAW_DIR, DATA_PROC_DIR, FLAT_BAND

RETURNS_PATH = os.path.join(DATA_RAW_DIR, "returns.parquet")
OUT_PATH     = os.path.join(DATA_PROC_DIR, "master_index.parquet")


def assign_label(abnormal_return: float) -> int:
    if abnormal_return < -FLAT_BAND:
        return 0
    elif abnormal_return > FLAT_BAND:
        return 2
    else:
        return 1


def main():
    os.makedirs(DATA_PROC_DIR, exist_ok=True)
    df = pd.read_parquet(RETURNS_PATH)

    df["label"] = df["abnormal_return"].apply(assign_label)

    # Print class distribution for sanity check
    print("Label distribution:")
    print(df["label"].value_counts().sort_index().rename({0: "down", 1: "flat", 2: "up"}))
    print(f"\nTotal calls: {len(df)}")

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nMaster index saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
