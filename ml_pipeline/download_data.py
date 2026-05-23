"""
OptiMPa — Dataset Downloader
Phase 1 Utility: Downloads the UCI Concrete Compressive Strength dataset
and saves it as a clean CSV, ready for train.py.

Why not ship the data in the repo?
  - Keeps the repo lightweight.
  - Ensures we always use the canonical UCI source.
  - Reproducible by anyone who clones the project.
"""

import io
import os
import urllib.request

import pandas as pd

# ── Dataset Source ────────────────────────────────────────────────────────────
# UCI ML Repository — Concrete Compressive Strength Data Set
# I-Cheng Yeh, 1998. 1030 samples, 9 attributes.
# We fetch the raw CSV mirror (no XLS dependency needed).
DATA_URL = (
    "https://raw.githubusercontent.com/dsrscientist/"
    "dataset1/master/cement_strength.csv"
)

OUTPUT_DIR  = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "concrete.csv")

COLUMN_MAP = {
    # Original verbose names → compact internal names used throughout the project
    "cement (component 1)(kg in a m^3 mixture)"                 : "cement",
    "blast furnace slag (component 2)(kg in a m^3 mixture)"     : "slag",
    "fly ash (component 3)(kg in a m^3 mixture)"                : "fly_ash",
    "water  (component 4)(kg in a m^3 mixture)"                 : "water",
    "superplasticizer (component 5)(kg in a m^3 mixture)"       : "superplasticizer",
    "coarse aggregate  (component 6)(kg in a m^3 mixture)"      : "coarse_agg",
    "fine aggregate (component 7)(kg in a m^3 mixture)"         : "fine_agg",
    "age (day)"                                                  : "age",
    "concrete compressive strength(MPa, megapascals) "          : "strength",
    # Fallback for the mirror CSV which uses positional/short names
    "Cement (component 1)(kg in a m^3 mixture)"                 : "cement",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)"     : "slag",
    "Fly Ash (component 3)(kg in a m^3 mixture)"                : "fly_ash",
    "Water  (component 4)(kg in a m^3 mixture)"                 : "water",
    "Superplasticizer (component 5)(kg in a m^3 mixture)"       : "superplasticizer",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)"      : "coarse_agg",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)"         : "fine_agg",
    "Age (day)"                                                  : "age",
    "Concrete compressive strength(MPa, megapascals) "          : "strength",
}


def download_dataset() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[>>] Fetching dataset from:\n    {DATA_URL}\n")

    try:
        with urllib.request.urlopen(DATA_URL, timeout=15) as response:
            raw = response.read().decode("utf-8")

        df = pd.read_csv(io.StringIO(raw))

        # Rename columns if they match the verbose UCI headers
        df.rename(columns=COLUMN_MAP, inplace=True)

        # If the mirror already has short names, the rename is a no-op.
        # Verify we have all required columns.
        required = ["cement","slag","fly_ash","water","superplasticizer",
                    "coarse_agg","fine_agg","age","strength"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            # Fallback: treat columns positionally (UCI XLS order is fixed)
            df.columns = required
            print("  [!] Used positional column assignment (mirror had unnamed headers)")

        # Drop any duplicates — UCI dataset has none, but defensive practice
        before = len(df)
        df.drop_duplicates(inplace=True)
        after = len(df)
        if before != after:
            print(f"  [!] Removed {before - after} duplicate rows")

        df.to_csv(OUTPUT_FILE, index=False)
        print(f"[✓] Dataset saved → {OUTPUT_FILE}")
        print(f"    Rows: {len(df)}  |  Columns: {list(df.columns)}\n")

    except Exception as e:
        print(f"\n[✗] Download failed: {e}")
        print("    Please manually download the dataset and save it as data/concrete.csv")
        print("    Source: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength\n")
        raise


if __name__ == "__main__":
    download_dataset()
