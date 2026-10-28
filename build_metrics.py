"""Builds metrics.json for the dashboard. Run after train_final.py:  python build_metrics.py"""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import SymptomMultiHot

CSV_PATH = "animal_disease_dataset.csv"
SYMPTOM_COLS = ["symptom 1", "symptom 2", "symptom 3"]
TARGET = "disease"
RANDOM_STATE = 42
AMBIGUITY_MARGIN = 0.15  # keep identical to main.py

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()
df["animal"] = df["animal"].str.strip().str.lower()
df["_key"] = df[SYMPTOM_COLS].apply(lambda r: "|".join(sorted(r.dropna().astype(str).str.lower())), axis=1)

diseases = sorted(df[TARGET].unique())
symptoms = sorted({s.strip().lower() for c in SYMPTOM_COLS for s in df[c].dropna().astype(str)})

# ---------- dataset facts ----------
per_key = df.groupby("_key")[TARGET].nunique()
amb_keys = set(per_key[per_key > 1].index)
dataset = {
    "rows": int(len(df)),
    "symptoms": len(symptoms),
    "diseases": diseases,
    "class_counts": {d: int(n) for d, n in df[TARGET].value_counts().reindex(diseases).items()},
    "unique_symptom_combos": int(len(per_key)),
    "ambiguous_combos": int(len(amb_keys)),
    "ambiguous_rows": int(df["_key"].isin(amb_keys).sum()),
    "animals": sorted(df["animal"].unique()),
    "age_range": [float(df["age"].min()), float(df["age"].max())],
    "temp_range": [float(df["temperature"].min()), float(df["temperature"].max())],
}
by_disease = df.groupby(TARGET)[["age", "temperature"]].agg(["mean", "std"])
age_temp = {
    d: {
        "age_mean": round(float(by_disease.loc[d, ("age", "mean")]), 2),
        "age_std": round(float(by_disease.loc[d, ("age", "std")]), 2),
        "temp_mean": round(float(by_disease.loc[d, ("temperature", "mean")]), 2),
        "temp_std": round(float(by_disease.loc[d, ("temperature", "std")]), 2),
    }
    for d in diseases
}

# ---------- split once, before any fitting ----------
X = df.drop(columns=[TARGET, "_key"])
y = df[TARGET]
keys = df["_key"]
X_train, X_test, y_train, y_test, k_train, k_test = train_test_split(
    X, y, keys, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_pre(use_animal, use_numeric):
    parts = [("sym", SymptomMultiHot(), SYMPTOM_COLS)]
    if use_animal:
        parts.append(("animal", OneHotEncoder(handle_unknown="ignore"), ["animal"]))
    if use_numeric:
        parts.append(("num", StandardScaler(), ["age", "temperature"]))
    return ColumnTransformer(parts, sparse_threshold=0)


# ---------- 1. model comparison (5-fold CV macro-F1, train only) ----------
base = cross_val_score(DummyClassifier(strategy="most_frequent"),
                       np.zeros((len(X_train), 1)), y_train, cv=cv, scoring="f1_macro")
comparison = [{"features": "none (majority class)", "model": "Baseline",
               "cv_macro_f1": round(float(base.mean()), 4), "cv_std": round(float(base.std()), 4)}]
FEATURE_SETS = {
    "symptoms only": (False, False),
    "symptoms + animal": (True, False),
    "symptoms + animal + age + temperature": (True, True),
}
for fs, (ua, un) in FEATURE_SETS.items():
    lr = Pipeline([("pre", build_pre(ua, un)),
                   ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    s = cross_val_score(lr, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    comparison.append({"features": fs, "model": "Logistic Regression",
                       "cv_macro_f1": round(float(s.mean()), 4), "cv_std": round(float(s.std()), 4)})
    knn = Pipeline([("pre", build_pre(ua, un)), ("clf", KNeighborsClassifier())])
    gs = GridSearchCV(knn, {"clf__n_neighbors": [5, 15, 31, 51], "clf__weights": ["uniform", "distance"]},
                      cv=cv, scoring="f1_macro", n_jobs=-1).fit(X_train, y_train)
    comparison.append({"features": fs, "model": f"KNN (k={gs.best_params_['clf__n_neighbors']})",
                       "cv_macro_f1": round(float(gs.best_score_), 4),
                       "cv_std": round(float(gs.cv_results_['std_test_score'][gs.best_index_]), 4)})

# ---------- 2. deployed model (symptoms only), trained on the train split ----------
deployed = Pipeline([("sym", SymptomMultiHot()), ("clf", LogisticRegression(C=10, max_iter=2000))])
deployed.fit(X_train[SYMPTOM_COLS], y_train)
classes = list(deployed.classes_)
proba = deployed.predict_proba(X_test[SYMPTOM_COLS])
hard = deployed.predict(X_test[SYMPTOM_COLS])


def report(y_true, y_pred):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "per_class": {d: {k: round(float(rep[d][k]), 4) for k in ("precision", "recall", "f1-score")}
                      | {"support": int(rep[d]["support"])} for d in diseases},
    }


# What the API actually does: commit to one disease, or flag a tie.
decisions = []
for p in proba:
    order = np.argsort(p)[::-1]
    decisions.append([classes[i] for i in order if p[order[0]] - p[i] < AMBIGUITY_MARGIN])
confident = np.array([len(d) == 1 for d in decisions])
contains_truth = np.array([t in d for t, d in zip(y_test, decisions)])
conf_correct = np.array([d[0] == t for t, d in zip(y_test, decisions) if len(d) == 1])
cols = diseases + ["ambiguous"]
matrix = [[0] * len(cols) for _ in diseases]
for t, d in zip(y_test, decisions):
    matrix[diseases.index(t)][cols.index(d[0] if len(d) == 1 else "ambiguous")] += 1

deployed_block = {
    "hard_label": report(y_test, hard),
    "decision": {
        "confident_share": round(float(confident.mean()), 4),
        "confident_accuracy": round(float(conf_correct.mean()), 4),
        "ambiguous_share": round(float(1 - confident.mean()), 4),
        "truth_in_decision_set": round(float(contains_truth.mean()), 4),
        "columns": cols,
        "rows": diseases,
        "matrix": matrix,
        "test_rows": int(len(y_test)),
    },
}

coefs = pd.DataFrame(deployed.named_steps["clf"].coef_, index=classes,
                     columns=deployed.named_steps["sym"].get_feature_names_out())
top_symptoms = {d: [{"symptom": s, "weight": round(float(w), 2)} for s, w in coefs.loc[d].nlargest(4).items()]
                for d in diseases}

# ---------- 3. evidence: do animal / age / temperature help? ----------
ref = Pipeline([("pre", build_pre(True, True)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]).fit(X_train, y_train)
ref_pred = ref.predict(X_test)
mask = k_test.isin(amb_keys).to_numpy()
sub_true = y_test.to_numpy()[mask]
sub_acc = float((ref_pred[mask] == sub_true).mean())
majority = float(pd.Series(sub_true).value_counts(normalize=True).max())
evidence = {
    "reference_model": "Logistic Regression, all features, balanced",
    "reference_test": report(y_test, ref_pred),
    "ambiguous_test_rows": int(mask.sum()),
    "ambiguous_subset_accuracy_all_features": round(sub_acc, 4),
    "ambiguous_subset_always_guess_majority": round(majority, 4),
    "age_temp_by_disease": age_temp,
}

out = {"dataset": dataset, "symptoms": symptoms, "comparison": comparison,
       "deployed": deployed_block, "top_symptoms": top_symptoms, "evidence": evidence}
with open("metrics.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out[k] for k in ("dataset", "deployed")}, indent=1)[:2500])
print("ambiguous subset acc (all features):", sub_acc, "| always-majority:", majority, "| n =", int(mask.sum()))
print("saved metrics.json")
