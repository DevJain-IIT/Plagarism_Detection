"""Phase 2: engineered pair-similarity features. Writes results/features.json."""
import json
from pathlib import Path

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from plag.data import SEED, load, split
from plag.features import NAMES, PairFeatures
from train_baseline import score, text

OUT = Path(__file__).parent / "results"

if __name__ == "__main__":
    train, val, test = split(load())
    pf = PairFeatures().fit(train)
    Ftr, Fva, Fte = (pf.transform(d) for d in (train, val, test))
    ytr, yva, yte = train["label"], val["label"], test["label"]

    vec = TfidfVectorizer()
    Ttr = vec.fit_transform(text(train))
    Tva, Tte = vec.transform(text(val)), vec.transform(text(test))

    xgb = lambda: XGBClassifier(n_estimators=300, max_depth=6, eval_metric="logloss", random_state=SEED)
    sets = {
        "pair_features_xgb": (xgb(), Ftr, Fva, Fte),
        "pair_features+tfidf_xgb": (xgb(), sp.hstack([Ftr, Ttr]).tocsr(), sp.hstack([Fva, Tva]).tocsr(), sp.hstack([Fte, Tte]).tocsr()),
    }
    res = {"features": NAMES, "val": {}}
    for name, (m, a, b, _) in sets.items():
        m.fit(a, ytr)
        res["val"][name] = score(m, b, yva)
        print(name, {k: round(v, 4) for k, v in res["val"][name].items()}, flush=True)
    best = max(res["val"], key=lambda k: res["val"][k]["acc"])
    res["best"], res["test"] = best, score(sets[best][0], sets[best][3], yte)
    print("BEST", best, res["test"])
    (OUT / "features.json").write_text(json.dumps(res, indent=2))
