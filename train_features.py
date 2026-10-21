"""Step 2: add pair-similarity features. Saves results/features.json."""
import json
from pathlib import Path

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from plag.data import SEED, load_data, split_data
from plag.features import FEATURE_NAMES, PairFeatures
from train_baseline import evaluate, make_text

RESULTS_DIR = Path(__file__).parent / "results"


def new_xgb():
    return XGBClassifier(n_estimators=300, max_depth=6, eval_metric="logloss", random_state=SEED)


if __name__ == "__main__":
    train, val, test = split_data(load_data())
    y_train, y_val, y_test = train["label"], val["label"], test["label"]

    # the 9 pair features
    pair_features = PairFeatures().fit(train)
    F_train = pair_features.transform(train)
    F_val = pair_features.transform(val)
    F_test = pair_features.transform(test)

    # the tf-idf text features from step 1
    vectorizer = TfidfVectorizer()
    T_train = vectorizer.fit_transform(make_text(train))
    T_val = vectorizer.transform(make_text(val))
    T_test = vectorizer.transform(make_text(test))

    # (model, train data, val data, test data) for each experiment
    experiments = {
        "pair_features_xgb": (new_xgb(), F_train, F_val, F_test),
        "pair_features+tfidf_xgb": (
            new_xgb(),
            sp.hstack([F_train, T_train]).tocsr(),
            sp.hstack([F_val, T_val]).tocsr(),
            sp.hstack([F_test, T_test]).tocsr(),
        ),
    }

    results = {"features": FEATURE_NAMES, "val": {}}
    for name, (model, X_train, X_val, X_test) in experiments.items():
        model.fit(X_train, y_train)
        results["val"][name] = evaluate(model, X_val, y_val)
        print(name, results["val"][name], flush=True)

    # pick on validation, test once
    best = max(results["val"], key=lambda name: results["val"][name]["acc"])
    best_model, _, _, best_X_test = experiments[best]
    results["best"] = best
    results["test"] = evaluate(best_model, best_X_test, y_test)
    print("best:", best, results["test"])

    (RESULTS_DIR / "features.json").write_text(json.dumps(results, indent=2))
