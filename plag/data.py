from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA = Path(__file__).resolve().parents[1] / "dataset" / "train_snli.txt"
SEED = 42


def load():
    """Raw file has no header (the notebook lost row 1 by reading header=0)."""
    df = pd.read_csv(DATA, sep="\t", header=None, names=["source_text", "plagiarized_text", "label"])
    df = df.dropna().drop_duplicates()
    # same pair with both labels is noise, drop it
    conflict = df.duplicated(["source_text", "plagiarized_text"], keep=False)
    return df[~conflict].reset_index(drop=True)


def split(df):
    """60/20/20, grouped by source sentence: SNLI reuses each premise ~2.4x, so a
    random split leaks the same source into train and test."""
    g = df["source_text"]
    tr, rest = next(GroupShuffleSplit(1, test_size=0.4, random_state=SEED).split(df, groups=g))
    va, te = next(GroupShuffleSplit(1, test_size=0.5, random_state=SEED).split(df.iloc[rest], groups=g.iloc[rest]))
    return df.iloc[tr], df.iloc[rest[va]], df.iloc[rest[te]]
