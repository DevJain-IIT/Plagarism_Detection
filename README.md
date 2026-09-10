# Plagiarism Detection Using NLP and Machine Learning

A supervised machine learning pipeline that classifies a pair of sentences as
plagiarised or not plagiarised. The project covers the full workflow: data
validation, statistical analysis of the corpus, text preprocessing, TF-IDF
feature extraction, training and comparison of five classifiers, and
persistence of the best performing model for reuse.

## Problem Statement

Given a source sentence and a candidate sentence, decide whether the candidate
is a reworded version of the source (label 1) or an unrelated statement
(label 0). The task is treated as binary text-pair classification rather than
exact string matching, so that paraphrasing and synonym substitution are still
detected.

## Dataset

The corpus is `train_snli.txt`, a tab separated derivative of the Stanford
Natural Language Inference corpus, shipped in this repository as
`dataset/plag_project_dataset.txt.zip` (38 MB uncompressed).

| Property | Value |
| --- | --- |
| Rows | 367,372 |
| Columns | `source_text`, `plagiarized_text`, `label` |
| Label 0 (not plagiarised) | 183,965 |
| Label 1 (plagiarised) | 183,407 |
| Null values | 4, all in `plagiarized_text` |
| Duplicate rows | 454 |

The class balance is close to 50/50, so plain accuracy is a reasonable headline
metric and no resampling was required.

Note on provenance: SNLI is a natural language inference dataset. The
entailment label is used here as a proxy for "the candidate sentence restates
the source". This is a reasonable stand in for semantic plagiarism, but it is
not a corpus of real academic plagiarism cases, and results should be read with
that in mind.

## Statistical Analysis

Before modelling, the corpus was tested for a measurable difference between
source and candidate sentence lengths.

95 percent confidence intervals for mean character length:

| Field | Lower bound | Upper bound |
| --- | --- | --- |
| `source_text` | 47.27 | 47.41 |
| `plagiarized_text` | 23.75 | 23.82 |

Independent two sample t-test on the two length distributions:

* Null hypothesis: the mean lengths are equal.
* T-statistic: 605.83
* P-value: 0.0
* Conclusion: reject the null hypothesis. Candidate sentences are
  significantly shorter than source sentences.

This is a useful diagnostic. It shows the corpus carries a length signal that a
model could exploit as a shortcut, which is one reason to keep raw length out of
the feature set and rely on lexical overlap instead.

## Pipeline

1. **Load and rename.** The raw file has no header row, so the first record is
   promoted by pandas and the three columns are renamed to `source_text`,
   `plagiarized_text` and `label`.
2. **Validate.** Null and duplicate counts are reported.
3. **Preprocess.** For both text columns: strip punctuation, lowercase, and
   remove English stopwords using the NLTK stopword list.
4. **Vectorise.** The two cleaned columns are concatenated into a single string
   per row and passed through `TfidfVectorizer`.
5. **Split.** 60 percent train, 20 percent validation, 20 percent test, using
   two successive calls to `train_test_split` with `random_state=42`.
6. **Train and compare.** Five classifiers are fitted on the training split and
   scored on the validation split.
7. **Select and persist.** The strongest model on validation is pickled
   together with the vectoriser, reloaded, and scored once on the held out test
   split.

## Models and Results

All figures below are accuracy on the validation split (73,475 rows).

| Model | Configuration | Validation accuracy |
| --- | --- | --- |
| XGBoost | 100 estimators | 0.7384 |
| Linear SVM | `LinearSVC`, default C | 0.6952 |
| Logistic Regression | `max_iter=1000` | 0.6850 |
| Random Forest | 10 estimators | 0.6640 |
| Multinomial Naive Bayes | default alpha | 0.6446 |

XGBoost was selected as the final model. Its performance on the held out test
split (73,475 rows, scored exactly once):

```
Test Accuracy: 0.7380

              precision    recall  f1-score   support

           0       0.78      0.66      0.72     36957
           1       0.70      0.82      0.76     36518

    accuracy                           0.74     73475
   macro avg       0.74      0.74      0.74     73475
weighted avg       0.74      0.74      0.74     73475

Confusion matrix:
[[24409 12548]
 [ 6706 29812]]
```

Validation and test accuracy agree to within 0.0005, which indicates the model
is not overfitted to the validation split.

Reading the confusion matrix: the model recovers 82 percent of true plagiarism
cases (recall on class 1) at the cost of flagging 12,548 clean pairs as
plagiarised. For a screening tool that bias is usually the correct one, since a
missed case is more costly than a false alarm that a human reviewer can
dismiss.

## Repository Structure

```
Plagarism_Detection/
├── code/
│   └── Plagiarism_Detector (2).ipynb   Notebook: EDA, statistics, training, evaluation
├── dataset/
│   └── plag_project_dataset.txt.zip    Zipped train_snli.txt, 367,372 rows
├── logistic_model.pkl                  Trained logistic regression
├── naive_bayes_model.pkl               Trained multinomial naive Bayes
├── svm_model.pkl                       Trained linear SVM
├── xgboost_model (1).pkl               Trained XGBoost, the selected model
├── tfidf_vectorizer (1).pkl            Fitted TF-IDF vectoriser
└── README.md
```

## Tech Stack

| Layer | Tools |
| --- | --- |
| Language | Python 3.11 |
| Data handling | pandas, NumPy |
| NLP | NLTK (stopwords), scikit-learn `TfidfVectorizer` |
| Modelling | scikit-learn, XGBoost |
| Statistics | SciPy, statsmodels |
| Visualisation | Matplotlib, Seaborn |
| Persistence | pickle |

## How to Run

Clone the repository and unpack the dataset:

```bash
git clone https://github.com/DevJain-IIT/Plagarism_Detection.git
cd Plagarism_Detection
unzip dataset/plag_project_dataset.txt.zip -d dataset/
```

Install the dependencies:

```bash
pip install pandas numpy scikit-learn xgboost nltk scipy statsmodels matplotlib seaborn
python -c "import nltk; nltk.download('stopwords')"
```

Open `code/Plagiarism_Detector (2).ipynb` in Jupyter or Google Colab and run the
cells in order. The notebook currently reads the dataset from a Google Drive
path, so update the path in the second cell to `dataset/train_snli.txt` before
running locally.

To use the saved model directly:

```python
import pickle

model = pickle.load(open("xgboost_model (1).pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer (1).pkl", "rb"))

def detect(source_text, candidate_text):
    features = vectorizer.transform([source_text + " " + candidate_text])
    return "Plagiarism Detected" if model.predict(features)[0] == 1 else "No Plagiarism"
```

Both arguments are required. The model was trained on the concatenation of a
source and a candidate sentence, so it cannot judge a single passage in
isolation.

## Known Limitations

These are documented deliberately, and each one is a concrete target for the
next iteration.

1. **The saved vectoriser and the saved model are out of sync.** The Random
   Forest cell refits `tfidf_vectorizer` with `max_features=10000` without
   re-splitting the data. Every later model still trains on the original full
   vocabulary matrix, but the vectoriser written to `tfidf_vectorizer.pkl` is
   the reduced one. Loading both artefacts together will therefore not
   reproduce the reported test accuracy. The fix is to fit the vectoriser once,
   before the split, and never refit it afterwards.
2. **The notebook `detect()` helper takes a single string.** Since the model
   expects a source and candidate pair, single string calls compare the input
   against nothing and the output is not meaningful. The corrected two argument
   version is given above.
3. **No leakage guard on the vectoriser.** `TfidfVectorizer` is fitted on the
   full dataset before splitting, so test set vocabulary and document
   frequencies influence training. It should be fitted on the training split
   only and applied to the other splits with `transform`.
4. **Nulls and duplicates are counted but not removed.** The 4 null rows and
   454 duplicate rows remain in the data used for training.
5. **Bag of words features ignore word order and synonymy.** TF-IDF over a
   concatenated pair cannot represent the relationship between the two
   sentences, only the union of their tokens. This is the main reason accuracy
   sits near 74 percent rather than the low 90s that this corpus allows with
   sentence embeddings.
6. **No hyperparameter search.** Random Forest uses 10 estimators and the other
   models use library defaults, so the comparison measures default behaviour
   rather than tuned potential.

## Roadmap

* Refactor the notebook into an importable package with a single `train.py`
  entry point and one fixed random seed.
* Add pair aware features: TF-IDF cosine similarity, Jaccard overlap, length
  ratio, and shared n-gram counts.
* Replace bag of words with Sentence-BERT embeddings and compare against the
  TF-IDF baseline on the same test split.
* Add cross validation and a proper hyperparameter search.
* Expose the model through a FastAPI endpoint with a small web interface.
* Extend from sentence pairs to document level detection by chunking documents
  and scoring every chunk pair.

## License

No license file is currently present. Add one before reusing this code.
