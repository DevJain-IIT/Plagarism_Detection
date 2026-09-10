# Plagiarism Detection Using NLP and Machine Learning

This project focuses on detecting plagiarism between pairs of text using a robust Natural Language Processing (NLP) and machine learning pipeline. The core idea is to determine whether a given text is plagiarized based on its similarity to a source sentence using statistical and predictive modeling.

---

## Dataset

The dataset contains **367,372 rows** with:
- `source_text`: Original sentence
- `plagiarized_text`: Possibly plagiarized sentence
- `label`: `1` = Plagiarized, `0` = Not Plagiarized

---

## Statistical Analysis

- **Confidence Intervals**: Used to compare average text lengths between source and plagiarized texts
- **Hypothesis Testing (t-test)**: Validated statistical significance in differences between source and plagiarized text lengths

---

## Models Trained

Five classification models were trained and evaluated:

1. **Logistic Regression**
2. **Naive Bayes**
3. **Random Forest**
4. **XGBoost**
5. **Support Vector Machine (SVM)**

All models were evaluated using standard classification metrics including precision, recall, F1-score, and accuracy.

---

## Features

- Text cleaning: punctuation removal, lowercasing, stopword filtering
- TF-IDF vectorization to convert text into machine-readable form
- Trained on five popular classification algorithms
- Evaluated model performance with confusion matrix and metrics
- Confidence intervals & hypothesis testing used for statistical insight
- Saved trained model using `pickle` for deployment or reuse

---

## Tech Stack

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- XGBoost
- NLTK (for stopwords)
- Matplotlib, Seaborn (for visualizations)

---

## Results

The models performed well on the binary classification task. SVM and XGBoost delivered particularly strong performance. Statistical analysis also showed that plagiarized text is generally shorter than source text with a significant difference.

---

##  File Structure

```
 Plagiarism_Detector/
│
├── Plagiarism_Detector.ipynb     # Main notebook with code
├── dataset.csv.gz                # Compressed dataset (~367k rows)
├── svm_model.pkl                 # Saved SVM model
├── tfidf_vectorizer.pkl          # Saved vectorizer (optional)
└── README.md                     # Project documentation
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Plagiarism_Detector.git
   cd Plagiarism_Detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   - Launch `Plagiarism_Detector.ipynb` in Jupyter or Google Colab

4. Use trained model:
   ```python
   import pickle
   model = pickle.load(open("svm_model.pkl", "rb"))
   ```

---

## Future Improvements

- Add semantic embeddings (BERT, Sentence-BERT) for better similarity detection
- Build an API using Flask or FastAPI
- Add a web UI for sentence comparison
- Deploy as a cloud-based microservice

---

## Acknowledgments

Special thanks to the open-source libraries and the NLP research community.
