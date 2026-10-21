# Semantic Plagiarism Detection

Given a source sentence and a candidate sentence, decide whether the candidate
restates the source. Three generations of the model are compared on the same
held-out test split, then the best one is served as a Dockerized FastAPI
service with a Streamlit front end.

## Results

Test split: 73,352 pairs, never used for model selection.

| Model | Accuracy | Recall (plagiarised) | ROC-AUC |
| --- | --- | --- | --- |
| TF-IDF + XGBoost (best of 5 baselines) | 74.0% | 0.82 | 0.818 |
| + engineered pair features (cosine, Jaccard, length ratio, shared n-grams) | 81.2% | 0.81 | 0.898 |
| **MiniLM sentence embeddings + MLP head** | **92.2%** | **0.93** | **0.975** |

Baselines on validation: XGBoost 74.3%, Linear SVM 71.6%, Random Forest 71.0%,
Logistic Regression 70.2%, Naive Bayes 67.2%.

Pair features alone reach only 70.5%: overlap measures topic, not meaning, so a
contradiction with the same words looks identical to a paraphrase.

## Dataset and its limits

`train_snli.txt`, a tab-separated derivative of SNLI (367,373 pairs, balanced
labels). The SNLI entailment label is a **proxy** for "the candidate restates
the source". It is not a corpus of real academic plagiarism, and the model
judges sentence pairs, not documents.

Cleaning applied before splitting: dropped 4 null rows, 454 duplicate rows and
18 pairs carrying both labels. The split is 60/20/20 **grouped by source
sentence**, because SNLI reuses each premise about 2.4 times and a random split
would put the same source in train and test.

## Layout

```
plag/data.py          load, clean, grouped split
plag/features.py      9 pair-similarity features
plag/model.py         Detector: frozen MiniLM encoder + MLP head
train_baseline.py     5 TF-IDF classifiers      -> results/baseline.json
train_features.py     pair features + TF-IDF    -> results/features.json
train_embeddings.py   embeddings + head         -> results/embeddings.json, head.pt
api.py                FastAPI: POST /predict, GET /health
app.py                Streamlit front end
tests/                pytest (features, API contract, real-model check)
Dockerfile            CPU-only image, encoder baked in
.github/workflows/    pytest + docker build + container smoke test
```

## Run

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"

python train_baseline.py      # ~5 min
python train_features.py      # ~2 min
python train_embeddings.py    # encodes ~500K sentences; CPU ~1 h, Colab GPU minutes

python -m pytest tests -q
streamlit run app.py
```

API:

```bash
docker build -t plag-api .
docker run -p 8000:8000 plag-api
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"source":"A man rides a horse.","candidate":"A person is on an animal."}'
# {"plagiarised": true, "score": 0.98, "latency_ms": 29.2}
```

Measured CPU latency in the container: median 27.5 ms, p95 66 ms (single
thread, Docker Desktop on a laptop CPU).

## Known limitations

1. Entailment proxy, not real plagiarism (see above).
2. SNLI hypotheses carry annotation artefacts (negation words, shorter length)
   that any model can exploit; the 92% figure is on this distribution only.
3. The encoder is frozen. Fine-tuning a cross-encoder would likely score higher
   but needs a GPU.
4. Sentence pairs only. Document-level detection (chunk, then score every
   chunk pair) is not implemented.
5. Pickles from the original notebook (`*.pkl`) are superseded by `results/` and
   were trained with a leaky vectoriser.
