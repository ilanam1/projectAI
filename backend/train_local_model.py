# train_local_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "data" / "combined_reviews_with_labels.csv"

df = pd.read_csv(data_path, encoding="utf-8")


df = df.dropna(subset=["review_text", "label"])
df = df[df["label"].isin(["REAL", "FAKE"])]
texts = df["review_text"].astype(str)
labels = df["label"].map({"REAL": 0, "FAKE": 1})


df = df.drop_duplicates(subset=["review_text"])

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=80000,
        min_df=3   # להוריד רעש של תווים שמופיעים מעט מאוד
    )),
    ("clf", LogisticRegression(
        max_iter=300,
        class_weight={0: 1.0, 1: 1.8},  # לתת משקל גדול יותר ל־FAKE (label=1)
        C=0.7,                           # קצת יותר רגולריזציה
        solver="liblinear"
    ))
])


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_val)
print(classification_report(y_val, y_pred))

models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)
joblib.dump(pipeline, models_dir / "hebrew_fake_review_tfidf.joblib")
print("✅ Model saved to", models_dir / "hebrew_fake_review_tfidf.joblib")
