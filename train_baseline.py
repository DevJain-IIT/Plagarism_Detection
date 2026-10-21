"""Phase 1: the five TF-IDF classifiers from the notebook, on a clean pipeline
(dedupe, grouped split, vectoriser fit on train only). Writes results/baseline.json."""
import json
import string
import time
from pathlib import Path

import joblib
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from plag.data import SEED, load, split

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
STOP = set(stopwords.words("english"))
PUNCT = str.maketrans("", "", string.punctuation)


def clean(s):
    return " ".join(w for w in s.translate(PUNCT).lower().split() if w not in STOP)


def text(df):
    return (df["source_text"].map(clean) + " " + df["plagiarized_text"].map(clean)).tolist()


def score(model, X, y):
    pred = model.predict(X)
    s = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    return {"acc": accuracy_score(y, pred), "recall_1": recall_score(y, pred), "auc": roc_auc_score(y, s)}


if __name__ == "__main__":
    train, val, test = split(load())
    print(f"train {len(train)}  val {len(val)}  test {len(test)}")
    vec = TfidfVectorizer()
    Xtr = vec.fit_transform(text(train))
    Xva, Xte = vec.transform(text(val)), vec.transform(text(test))
    ytr, yva, yte = train["label"], val["label"], test["label"]

    models = {
        "xgboost": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=SEED),
        "linear_svm": LinearSVC(random_state=SEED),
        "logistic": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=10, random_state=SEED, n_jobs=-1),
        "naive_bayes": MultinomialNB(),
    }
    res = {"n": {"train": len(train), "val": len(val), "test": len(test)}, "val": {}}
    for name, m in models.items():
        t = time.time()
        m.fit(Xtr, ytr)
        res["val"][name] = score(m, Xva, yva)
        print(name, {k: round(v, 4) for k, v in res["val"][name].items()}, f"{time.time() - t:.0f}s", flush=True)

    best = max(res["val"], key=lambda k: res["val"][k]["acc"])  # select on val, touch test once
    res["best"] = best
    res["test"] = score(models[best], Xte, yte)
    pred = models[best].predict(Xte)
    res["test_report"] = classification_report(yte, pred, output_dict=True)
    res["test_confusion"] = confusion_matrix(yte, pred).tolist()
    print("BEST", best, res["test"])
    (OUT / "baseline.json").write_text(json.dumps(res, indent=2))
    joblib.dump((vec, models[best]), OUT / "baseline_tfidf.joblib")
