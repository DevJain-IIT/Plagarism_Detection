from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_FILE = Path(__file__).resolve().parents[1] / "dataset" / "train_snli.txt"
SEED = 42


def load_data():
    # the file has no header row
    df = pd.read_csv(DATA_FILE, sep="\t", header=None,
                     names=["source_text", "plagiarized_text", "label"])
    df = df.dropna().drop_duplicates()

    # the same pair with two different labels is noise, drop both copies
    same_pair = df.duplicated(["source_text", "plagiarized_text"], keep=False)
    df = df[~same_pair]
    return df.reset_index(drop=True)


def split_data(df):
    # Split by source sentence. SNLI uses each source about 2.4 times, so a
    # random split would put the same source in both train and test.
    first = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)
    train_idx, rest_idx = next(first.split(df, groups=df["source_text"]))
    rest = df.iloc[rest_idx]

    second = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx, test_idx = next(second.split(rest, groups=rest["source_text"]))

    return df.iloc[train_idx], rest.iloc[val_idx], rest.iloc[test_idx]
