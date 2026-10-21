"""Phase 3: frozen sentence-transformer embeddings + a small MLP head on
[u, v, |u-v|, u*v]. Encodes each unique sentence once and caches it, so it is
resumable. Runs on CPU (~10-20 min) or a Colab GPU. Writes results/embeddings.json."""
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

from plag.data import SEED, load, split
from plag.model import MODEL, make_head, pair_vector

OUT = Path(__file__).parent / "results"


def encode_all(sents):
    cache = OUT / "sent_emb.npy"
    if cache.exists():
        return np.load(cache)
    model = SentenceTransformer(MODEL)
    emb = model.encode(sents, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    np.save(cache, emb.astype(np.float16))
    return emb


def pair_matrix(df, index, emb):
    u = emb[df["source_text"].map(index).to_numpy()].astype(np.float32)
    v = emb[df["plagiarized_text"].map(index).to_numpy()].astype(np.float32)
    return pair_vector(u, v)


def predict(net, X):
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(X)).squeeze(1)).numpy()


def metrics(p, y):
    return {"acc": accuracy_score(y, p > 0.5), "recall_1": recall_score(y, p > 0.5), "auc": roc_auc_score(y, p)}


if __name__ == "__main__":
    torch.manual_seed(SEED)
    train, val, test = split(load())
    sents = sorted(set(train.source_text) | set(train.plagiarized_text) | set(val.source_text) | set(val.plagiarized_text)
                   | set(test.source_text) | set(test.plagiarized_text))
    index = {s: i for i, s in enumerate(sents)}
    emb = encode_all(sents)
    (Xtr, Xva, Xte) = (pair_matrix(d, index, emb) for d in (train, val, test))
    ytr = torch.tensor(train["label"].to_numpy(), dtype=torch.float32)

    net = make_head(Xtr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(Xtr)
    best, res = (-1, None), {"val_by_epoch": []}
    for ep in range(12):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b]).squeeze(1), ytr[b]).backward()
            opt.step()
        net.eval()
        m = metrics(predict(net, Xva), val["label"].to_numpy())
        res["val_by_epoch"].append(m)
        print(ep, {k: round(v, 4) for k, v in m.items()}, flush=True)
        if m["acc"] > best[0]:  # select epoch on val only
            best = (m["acc"], {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    res["model"], res["test"] = MODEL, metrics(predict(net, Xte), test["label"].to_numpy())
    print("TEST", res["test"])
    (OUT / "embeddings.json").write_text(json.dumps(res, indent=2))
    torch.save(net.state_dict(), OUT / "head.pt")
