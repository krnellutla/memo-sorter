"""TF-IDF + logistic regression baseline. Trains in about a second."""
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
train = pd.read_csv(ROOT / "data" / "train.csv")
test = pd.read_csv(ROOT / "data" / "test.csv")

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
    LogisticRegression(max_iter=1000, C=5),
)
t0 = time.time()
model.fit(train.text, train.label)
print(f"trained in {time.time() - t0:.2f}s")

pred = model.predict(test.text)
print(classification_report(test.label, pred, digits=3))
labels = sorted(train.label.unique())
print(pd.DataFrame(confusion_matrix(test.label, pred, labels=labels), index=labels, columns=labels))

wrong = test[pred != test.label].assign(predicted=pred[pred != test.label])
print("\nmisses:\n", wrong.to_string(index=False))

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "models" / "baseline.joblib")
