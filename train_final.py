"""Final training script: symptoms-only Logistic Regression -> model.joblib (used by main.py)."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from features import SymptomMultiHot

CSV_PATH = "animal_disease_dataset.csv"
SYMPTOM_COLS = ["symptom 1", "symptom 2", "symptom 3"]
TARGET = "disease"
RANDOM_STATE = 42

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()
X = df[SYMPTOM_COLS]
y = df[TARGET]


def make_model():
    # No class_weight on purpose: we want honest probabilities for the API.
    # C=10 = less regularisation, so clear-cut cases get confident probabilities.
    return Pipeline([
        ("sym", SymptomMultiHot()),
        ("clf", LogisticRegression(C=10, max_iter=2000)),
    ])


# ---- honest evaluation (split before fitting) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(make_model(), X_train, y_train, cv=cv, scoring="f1_macro")
print(f"CV macro-F1 (train): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

model = make_model().fit(X_train, y_train)
print("\nTEST REPORT:")
print(classification_report(y_test, model.predict(X_test), digits=3))

# ---- refit on ALL data for deployment ----
final = make_model().fit(X, y)
joblib.dump(final, "model.joblib")
print("Symptoms known to the model:", len(final.named_steps["sym"].mlb_.classes_))
print("saved model.joblib")
