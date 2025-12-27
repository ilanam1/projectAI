import pandas as pd
import re
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

# --- 1. פונקציית ניקוי טקסט משופרת לעברית ---
def clean_hebrew_text(text):
    if not isinstance(text, str):
        return ""
    # הסרת ניקוד (אם קיים)
    text = re.sub(r'[\u0591-\u05C7]', '', text)
    # צמצום אותיות חוזרות (למשל: ממממש -> ממש)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # הסרת תווים מיוחדים שאינם עברית או סימני פיסוק בסיסיים
    text = re.sub(r'[^\u0590-\u05FFa-zA-Z0-9\s.,?!]', ' ', text)
    # רווחים כפולים
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- טעינת דאטה ---
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "data" / "combined_reviews_with_labels.csv"

print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path, encoding="utf-8")

# ניקוי בסיסי
df = df.dropna(subset=["review_text", "label"])
df = df[df["label"].isin(["REAL", "FAKE"])]

# הפעלת הניקוי המתקדם על הטקסט
df["review_text"] = df["review_text"].apply(clean_hebrew_text)

# הסרת כפילויות *אחרי* הניקוי (יותר יעיל)
df = df.drop_duplicates(subset=["review_text"])

texts = df["review_text"].astype(str)
labels = df["label"].map({"REAL": 0, "FAKE": 1})

print(f"Dataset size after cleaning: {len(df)}")

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- 2. בניית Pipeline משולב (מילים + תווים) ---

# TF-IDF ברמת תווים (לוכד מורפולוגיה וסגנון כתיבה)
char_tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=50000,
    min_df=3
)

# TF-IDF ברמת מילים (לוכד ביטויים וסלנג: "לא ממליץ", "חבל על הזמן")
word_tfidf = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2), # מילים בודדות וצמדים (Bigrams)
    max_features=30000,
    min_df=3
)

# איחוד הפיצ'רים
combined_features = FeatureUnion([
    ("chars", char_tfidf),
    ("words", word_tfidf)
])

pipeline = Pipeline([
    ("features", combined_features),
    ("clf", LogisticRegression(solver="liblinear", max_iter=500))
])

# --- 3. Grid Search למציאת הפרמטרים הטובים ביותר ---
# נגדיר פרמטרים לבדיקה. המודל יריץ את כל הקומבינציות ויבחר את הטובה ביותר.
param_grid = {
    # משקלים שונים ל-FAKE (לפעמים 2.0 או 3.0 עדיף אם יש מעט זיופים)
    'clf__class_weight': [{0: 1.0, 1: 1.5}, {0: 1.0, 1: 2.0}, 'balanced'],
    # חוזק הרגולריזציה (C קטן = פחות Overfitting)
    'clf__C': [0.5, 1.0, 3.0], 
}

print("Starting Grid Search (this might take a minute)...")
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5,                 # 5-Fold Cross Validation
    scoring='f1_macro',   # אופטימיזציה לממוצע בין FAKE ל-REAL
    n_jobs=-1             # שימוש בכל הליבות של המעבד
)

grid_search.fit(X_train, y_train)

print(f"Best params found: {grid_search.best_params_}")

# שמירת המודל הטוב ביותר
best_model = grid_search.best_estimator_

# --- בדיקת ביצועים ---
y_pred = best_model.predict(X_val)
print("\nValidation Report:")
print(classification_report(y_val, y_pred))

# שמירה
models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)
model_path = models_dir / "hebrew_fake_review_tfidf.joblib"
joblib.dump(best_model, model_path)
print(f"✅ Improved model saved to {model_path}")