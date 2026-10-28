"""Livestock disease screening API.  Run locally:  uvicorn main:app --reload"""
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from features import SymptomMultiHot  # noqa: F401  (joblib needs this import to unpickle the model)

MODEL_PATH = Path(__file__).parent / "model.joblib"
SYMPTOM_COLS = ["symptom 1", "symptom 2", "symptom 3"]
ANIMALS = ["cow", "buffalo", "sheep", "goat"]
AMBIGUITY_MARGIN = 0.15  # diseases within this probability of the top one count as "can't tell apart"

model = joblib.load(MODEL_PATH)
KNOWN_SYMPTOMS: List[str] = sorted(model.named_steps["sym"].mlb_.classes_)
KNOWN_SET = set(KNOWN_SYMPTOMS)
DISEASES: List[str] = list(model.classes_)

app = FastAPI(
    title="Livestock Disease Screening API",
    description="Predicts a likely disease from 1-3 symptoms. Screening aid only, not a vet replacement.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dashboard lives on GitHub Pages, so allow cross-origin calls
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    symptoms: List[str] = Field(
        ..., min_length=1, max_length=3,
        description="1 to 3 symptoms. See GET /symptoms for valid names.",
        examples=[["painless lumps", "depression", "loss of appetite"]],
    )
    # Optional context. Recorded and echoed back, but NOT used by the model (no signal in this dataset).
    animal: Optional[str] = Field(None, description="cow, buffalo, sheep or goat")
    age: Optional[float] = Field(None, ge=0, le=40, description="Age in years")
    temperature: Optional[float] = Field(None, ge=95, le=110, description="Body temperature in degrees F")

    @field_validator("animal")
    @classmethod
    def validate_animal(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().lower()
        if v not in ANIMALS:
            raise ValueError(f"Unknown animal '{v}'. Use one of: {ANIMALS}.")
        return v

    @field_validator("symptoms")
    @classmethod
    def validate_symptoms(cls, v: List[str]) -> List[str]:
        cleaned = [s.strip().lower() for s in v]
        unknown = [s for s in cleaned if s not in KNOWN_SET]
        if unknown:
            raise ValueError(f"Unknown symptom(s): {unknown}. Call GET /symptoms for valid names.")
        return list(dict.fromkeys(cleaned))  # drop repeats, keep order


class PredictResponse(BaseModel):
    top_prediction: str
    probabilities: Dict[str, float]
    ambiguous: bool
    possible_diseases: List[str]
    message: str
    inputs_used: List[str]
    ignored_inputs: List[str]
    note: str


@app.get("/")
def root():
    return {"service": "Livestock Disease Screening API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "diseases": DISEASES, "n_symptoms": len(KNOWN_SYMPTOMS)}


@app.get("/symptoms")
def symptoms():
    return {"symptoms": KNOWN_SYMPTOMS}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    row = req.symptoms + [None] * (3 - len(req.symptoms))
    X = pd.DataFrame([row], columns=SYMPTOM_COLS)
    proba = model.predict_proba(X)[0]

    ranked = sorted(zip(DISEASES, proba), key=lambda t: t[1], reverse=True)
    top_disease, top_p = ranked[0]
    possible = [d for d, p in ranked if top_p - p < AMBIGUITY_MARGIN]
    ambiguous = len(possible) > 1

    if ambiguous:
        message = (
            f"These symptoms cannot separate {' and '.join(possible)}. "
            "Get a vet exam or lab test to tell them apart."
        )
    else:
        message = f"Most likely {top_disease} ({top_p:.0%}). Screening aid only; confirm with a vet."

    given = {"animal": req.animal, "age": req.age, "temperature": req.temperature}
    ignored = [k for k, v in given.items() if v is not None]
    note = (
        "Animal, age and temperature were recorded but not used: in this dataset they carry no "
        "signal for telling these diseases apart. The prediction comes from symptoms only."
        if ignored else "The prediction comes from symptoms only."
    )

    return PredictResponse(
        top_prediction=top_disease,
        probabilities={d: round(float(p), 4) for d, p in ranked},
        ambiguous=ambiguous,
        possible_diseases=possible,
        message=message,
        inputs_used=["symptoms"],
        ignored_inputs=ignored,
        note=note,
    )
