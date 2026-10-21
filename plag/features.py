import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

NAMES = ["tfidf_cosine", "jaccard", "len_ratio_chars", "len_ratio_tokens",
         "shared_1gram", "shared_2gram", "shared_3gram", "cand_covered", "src_covered"]
TOK = re.compile(r"\w+")


def _grams(toks, n):
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


class PairFeatures:
    """Similarity features of a (source, candidate) pair. Fit on train only."""

    def fit(self, df):
        self.tfidf = TfidfVectorizer().fit(df["source_text"].tolist() + df["plagiarized_text"].tolist())
        return self

    def transform(self, df):
        a, b = df["source_text"].tolist(), df["plagiarized_text"].tolist()
        cos = np.asarray(self.tfidf.transform(a).multiply(self.tfidf.transform(b)).sum(axis=1)).ravel()
        rows = []
        for s, c in zip(a, b):
            ts, tc = TOK.findall(s.lower()), TOK.findall(c.lower())
            ss, sc = set(ts), set(tc)
            inter = len(ss & sc)
            rows.append([
                inter / max(len(ss | sc), 1),
                min(len(s), len(c)) / max(len(s), len(c), 1),
                min(len(ts), len(tc)) / max(len(ts), len(tc), 1),
                inter,
                len(_grams(ts, 2) & _grams(tc, 2)),
                len(_grams(ts, 3) & _grams(tc, 3)),
                inter / max(len(sc), 1),
                inter / max(len(ss), 1),
            ])
        return np.column_stack([cos, np.array(rows)])
