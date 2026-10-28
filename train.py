import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import SymptomMultiHot

# ---------- CONFIG (edit these to match your CSV) ----------
CSV_PATH = "animal_disease_dataset.csv"
SYMPTOM_COLS = ["symptom 1", "symptom 2", "symptom 3"]
TARGET = "disease"
RANDOM_STATE = 42

# ---------- 1. LOAD + SANITY CHECKS ----------
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()
print("shape:", df.shape)
print("nulls:\n", df.isna().sum())

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"exact duplicate rows dropped: {before - len(df)}")

# How ambiguous is the dataset? (same symptom set -> more than one disease)
df["_sym_key"] = df[SYMPTOM_COLS].apply(lambda r: "|".join(sorted(r.dropna().astype(str))), axis=1)
n_diseases_per_combo = df.groupby("_sym_key")[TARGET].nunique()
print(f"unique symptom combos: {len(n_diseases_per_combo)}")
print(f"combos that map to >1 disease: {(n_diseases_per_combo > 1).sum()}")
print(df[df["_sym_key"].isin(n_diseases_per_combo[n_diseases_per_combo > 1].index)][TARGET].value_counts())
df = df.drop(columns="_sym_key")

# ---------- 2. SPLIT (before fitting anything) ----------
X = df.drop(columns=TARGET)
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)


# ---------- 3. PIPELINES ----------
def build_pre(use_animal, use_numeric):
    parts = [("sym", SymptomMultiHot(), SYMPTOM_COLS)]
    if use_animal:
        parts.append(("animal", OneHotEncoder(handle_unknown="ignore"), ["animal"]))
    if use_numeric:
        parts.append(("num", StandardScaler(), ["age", "temperature"]))
    return ColumnTransformer(parts, sparse_threshold=0)


FEATURE_SETS = {
    "symptoms_only": (False, False),
    "symptoms+animal": (True, False),
    "all_features": (True, True),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---------- 4. BASELINE ----------
base = cross_val_score(
    DummyClassifier(strategy="most_frequent"),
    np.zeros((len(X_train), 1)), y_train, cv=cv, scoring="f1_macro",
)
print(f"\nBASELINE (majority class) macro-F1: {base.mean():.3f}")

# ---------- 5. MODEL x FEATURE-SET COMPARISON (CV on train only) ----------
results = []
for fs_name, (use_animal, use_numeric) in FEATURE_SETS.items():
    lr = Pipeline([
        ("pre", build_pre(use_animal, use_numeric)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    s = cross_val_score(lr, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    results.append((fs_name, "logreg", s.mean(), s.std(), lr))

    knn = Pipeline([
        ("pre", build_pre(use_animal, use_numeric)),
        ("clf", KNeighborsClassifier()),
    ])
    gs = GridSearchCV(
        knn,
        {"clf__n_neighbors": [5, 15, 31, 51], "clf__weights": ["uniform", "distance"]},
        cv=cv, scoring="f1_macro", n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    results.append((fs_name, f"knn {gs.best_params_}", gs.best_score_,
                    gs.cv_results_["std_test_score"][gs.best_index_], gs.best_estimator_))

res_df = pd.DataFrame(results, columns=["features", "model", "cv_macro_f1", "cv_std", "estimator"])
print("\nCV RESULTS (macro-F1, train only):")
print(res_df.drop(columns="estimator").sort_values("cv_macro_f1", ascending=False).to_string(index=False))

# ---------- 6. PICK BEST, EVALUATE ONCE ON TEST ----------
best_row = res_df.sort_values("cv_macro_f1", ascending=False).iloc[0]
print(f"\nBEST: {best_row['features']} | {best_row['model']}")
final = clone(best_row["estimator"]).fit(X_train, y_train)
y_pred = final.predict(X_test)

print("\nTEST REPORT:")
print(classification_report(y_test, y_pred, digits=3))

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix (test set)")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150)

# ---------- 7. WHICH SYMPTOMS DRIVE EACH DISEASE (LogReg only) ----------
clf = final.named_steps["clf"]
if isinstance(clf, LogisticRegression):
    names = final.named_steps["pre"].get_feature_names_out()
    coefs = pd.DataFrame(clf.coef_, index=clf.classes_, columns=names)
    print("\nTOP 5 FEATURES PER DISEASE:")
    for d in coefs.index:
        print(d, "->", coefs.loc[d].nlargest(5).round(2).to_dict())

# ---------- 8. REFIT ON ALL DATA + SAVE FOR THE API ----------
deploy_model = clone(best_row["estimator"]).fit(X, y)
joblib.dump(deploy_model, "model.joblib")
print("\nsaved model.joblib (keep features.py next to it)")
