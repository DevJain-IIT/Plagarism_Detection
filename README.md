# Semantic Plagiarism Detection

Given a source sentence and a candidate sentence, this checks whether the
candidate is a reworded version of the source. I compared TF-IDF models,
hand-made similarity features and sentence embeddings, and served the best one
with FastAPI and a small Streamlit page.

## Dataset

`train_snli.txt`, a tab-separated file derived from SNLI (367,373 sentence
pairs, about 50/50 labels). The entailment label is used as "plagiarised". That
is a stand-in, not real plagiarism data, and the model only compares sentence
pairs, not whole documents.

I removed null rows, duplicates and pairs with conflicting labels, then split
60/20/20 by source sentence so the same source never appears in both train and
test.

## Results

Test split of 73,352 pairs:

| Model | Accuracy | Recall (plagiarised) | ROC-AUC |
| --- | --- | --- | --- |
| TF-IDF + XGBoost (best of 5) | 74.0% | 0.82 | 0.818 |
| + similarity features (cosine, Jaccard, length ratio, shared n-grams) | 81.2% | 0.81 | 0.898 |
| MiniLM sentence embeddings + small neural net | 92.2% | 0.93 | 0.975 |

The similarity features alone only reach 70.5%, because word overlap shows
topic but not meaning.

## Run

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"

python train_baseline.py      # 5 TF-IDF models
python train_features.py      # adds similarity features
python train_embeddings.py    # slow on CPU (about 1 hour to encode)

python -m pytest tests -q
streamlit run app.py
```

API with Docker:

```bash
docker build -t plag-api .
docker run -p 8000:8000 plag-api
curl -X POST localhost:8000/predict -H "content-type: application/json" \
  -d '{"source":"A man rides a horse.","candidate":"A person is on an animal."}'
```

On my laptop CPU the container answers in about 27 ms (median).

`code/` and the `.pkl` files are the first notebook version. The app does not
use them.
