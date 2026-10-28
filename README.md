# Livestock disease screening

Predicts which of five livestock diseases (anthrax, blackleg, foot and mouth, lumpy virus, pneumonia) is most likely from 1 to 3 symptoms. When two diseases fit the symptoms equally well, it says so instead of guessing.

- **Dashboard:** `index.html` (reads `metrics.json`, calls the API)
- **API:** FastAPI app in `main.py`
  - `POST /predict` takes `{"symptoms": ["chills", "sweats", "fatigue"]}`
  - `GET /symptoms` lists valid symptoms, `GET /health` is the health check, `/docs` is interactive
- **Model:** Logistic Regression on multi-hot symptom features (`features.py`, `train_final.py`)

## Results (held-out test set, 8,756 cases)

- Anthrax, blackleg and foot and mouth: precision and recall 1.00
- Lumpy virus vs pneumonia: indistinguishable in this dataset (identical symptoms; 49.5% accuracy vs 50.5% for always guessing the majority). The API flags these as a tie.
- Animal, age and temperature carry no signal, so the model uses symptoms only. Random forest and KNN were tested and did not help.

## Run locally

```
pip install -r requirements.txt
python train_final.py        # creates model.joblib
uvicorn main:app --reload
python -m http.server 8080   # then open http://127.0.0.1:8080/index.html
```

`python build_metrics.py` regenerates `metrics.json` for the dashboard.

Screening aid only, not a diagnosis.
