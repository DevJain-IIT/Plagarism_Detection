"""Step 3: sentence embeddings + a small classifier on top.

Each unique sentence is encoded once and saved, so a rerun only retrains the
classifier. Encoding takes about an hour on CPU, a few minutes on a GPU.
Saves results/embeddings.json and results/head.pt.
"""
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

from plag.data import SEED, load_data, split_data
from plag.model import MODEL_NAME, make_head, pair_vector

RESULTS_DIR = Path(__file__).parent / "results"
EPOCHS = 12
BATCH_SIZE = 512


def encode_sentences(sentences):
    cache_file = RESULTS_DIR / "sent_emb.npy"
    if cache_file.exists():
        return np.load(cache_file)

    encoder = SentenceTransformer(MODEL_NAME)
    embeddings = encoder.encode(sentences, batch_size=256, show_progress_bar=True,
                                normalize_embeddings=True)
    np.save(cache_file, embeddings.astype(np.float16))
    return embeddings


def build_pairs(df, index, embeddings):
    # look up the embedding of every source and candidate sentence
    u = embeddings[df["source_text"].map(index).to_numpy()].astype(np.float32)
    v = embeddings[df["plagiarized_text"].map(index).to_numpy()].astype(np.float32)
    return pair_vector(u, v)


def predict(head, X):
    with torch.no_grad():
        return torch.sigmoid(head(torch.from_numpy(X)).squeeze(1)).numpy()


def evaluate(prob, y):
    return {
        "acc": accuracy_score(y, prob > 0.5),
        "recall_1": recall_score(y, prob > 0.5),
        "auc": roc_auc_score(y, prob),
    }


if __name__ == "__main__":
    torch.manual_seed(SEED)
    train, val, test = split_data(load_data())

    # every distinct sentence in the data, in a fixed order
    all_sentences = set()
    for df in (train, val, test):
        all_sentences |= set(df["source_text"]) | set(df["plagiarized_text"])
    sentences = sorted(all_sentences)
    index = {sentence: i for i, sentence in enumerate(sentences)}
    embeddings = encode_sentences(sentences)

    X_train = build_pairs(train, index, embeddings)
    X_val = build_pairs(val, index, embeddings)
    X_test = build_pairs(test, index, embeddings)
    y_train = torch.tensor(train["label"].to_numpy(), dtype=torch.float32)

    head = make_head(X_train.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    X_train_t = torch.from_numpy(X_train)

    results = {"val_by_epoch": []}
    best_acc = -1
    best_weights = None

    for epoch in range(EPOCHS):
        head.train()
        order = torch.randperm(len(X_train_t))
        for i in range(0, len(order), BATCH_SIZE):
            batch = order[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            logits = head(X_train_t[batch]).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train[batch])
            loss.backward()
            optimizer.step()

        head.eval()
        val_scores = evaluate(predict(head, X_val), val["label"].to_numpy())
        results["val_by_epoch"].append(val_scores)
        print(epoch, val_scores, flush=True)

        # keep the epoch that did best on validation
        if val_scores["acc"] > best_acc:
            best_acc = val_scores["acc"]
            best_weights = {k: v.clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_weights)
    results["model"] = MODEL_NAME
    results["test"] = evaluate(predict(head, X_test), test["label"].to_numpy())
    print("test:", results["test"])

    (RESULTS_DIR / "embeddings.json").write_text(json.dumps(results, indent=2))
    torch.save(head.state_dict(), RESULTS_DIR / "head.pt")
