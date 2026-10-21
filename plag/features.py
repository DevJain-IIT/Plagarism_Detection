import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURE_NAMES = [
    "tfidf_cosine", "jaccard", "len_ratio_chars", "len_ratio_tokens",
    "shared_1gram", "shared_2gram", "shared_3gram", "cand_covered", "src_covered",
]
WORD = re.compile(r"\w+")


def get_ngrams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


class PairFeatures:
    """Turns a (source, candidate) pair into 9 similarity numbers."""

    def fit(self, df):
        # learn the vocabulary from the training sentences only
        all_text = df["source_text"].tolist() + df["plagiarized_text"].tolist()
        self.tfidf = TfidfVectorizer().fit(all_text)
        return self

    def transform(self, df):
        sources = df["source_text"].tolist()
        candidates = df["plagiarized_text"].tolist()

        # cosine of the two tfidf vectors (they are already normalised)
        src_vecs = self.tfidf.transform(sources)
        cand_vecs = self.tfidf.transform(candidates)
        cosine = np.asarray(src_vecs.multiply(cand_vecs).sum(axis=1)).ravel()

        rows = []
        for src, cand in zip(sources, candidates):
            src_words = WORD.findall(src.lower())
            cand_words = WORD.findall(cand.lower())
            src_set = set(src_words)
            cand_set = set(cand_words)
            shared = len(src_set & cand_set)

            jaccard = shared / max(len(src_set | cand_set), 1)
            len_ratio_chars = min(len(src), len(cand)) / max(len(src), len(cand), 1)
            len_ratio_tokens = min(len(src_words), len(cand_words)) / max(len(src_words), len(cand_words), 1)
            shared_2 = len(get_ngrams(src_words, 2) & get_ngrams(cand_words, 2))
            shared_3 = len(get_ngrams(src_words, 3) & get_ngrams(cand_words, 3))
            cand_covered = shared / max(len(cand_set), 1)
            src_covered = shared / max(len(src_set), 1)

            rows.append([jaccard, len_ratio_chars, len_ratio_tokens,
                         shared, shared_2, shared_3, cand_covered, src_covered])

        return np.column_stack([cosine, np.array(rows)])
