"""Step 1: five TF-IDF classifiers. Saves results/baseline.json."""
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

from plag.data import SEED, load_data, split_data

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STOP_WORDS = set(stopwords.words("english"))
NO_PUNCT = str.maketrans("", "", string.punctuation)


def clean_text(sentence):
    words = sentence.translate(NO_PUNCT).lower().split()
    return " ".join(w for w in words if w not in STOP_WORDS)


def make_text(df):
    # the model sees both sentences joined into one string
    return (df["source_text"].map(clean_text) + " " + df["plagiarized_text"].map(clean_text)).tolist()


def evaluate(model, X, y):
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        prob = model.decision_function(X)
    return {
        "acc": accuracy_score(y, pred),
        "recall_1": recall_score(y, pred),
        "auc": roc_auc_score(y, prob),
    }


if __name__ == "__main__":
    train, val, test = split_data(load_data())
    print("train", len(train), "val", len(val), "test", len(test))

    # fit the vectoriser on the training data only
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(make_text(train))
    X_val = vectorizer.transform(make_text(val))
    X_test = vectorizer.transform(make_text(test))
    y_train, y_val, y_test = train["label"], val["label"], test["label"]

    models = {
        "xgboost": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=SEED),
        "linear_svm": LinearSVC(random_state=SEED),
        "logistic": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=10, random_state=SEED, n_jobs=-1),
        "naive_bayes": MultinomialNB(),
    }

    results = {"n": {"train": len(train), "val": len(val), "test": len(test)}, "val": {}}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        results["val"][name] = evaluate(model, X_val, y_val)
        print(name, results["val"][name], round(time.time() - start), "s", flush=True)

    # pick the best model on validation, then look at the test set once
    best = max(results["val"], key=lambda name: results["val"][name]["acc"])
    results["best"] = best
    results["test"] = evaluate(models[best], X_test, y_test)

    test_pred = models[best].predict(X_test)
    results["test_report"] = classification_report(y_test, test_pred, output_dict=True)
    results["test_confusion"] = confusion_matrix(y_test, test_pred).tolist()
    print("best:", best, results["test"])

    (RESULTS_DIR / "baseline.json").write_text(json.dumps(results, indent=2))
    joblib.dump((vectorizer, models[best]), RESULTS_DIR / "baseline_tfidf.joblib")
