"""Custom feature transformer. Keep this in its own file: the FastAPI app must import it to load model.joblib."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class SymptomMultiHot(BaseEstimator, TransformerMixin):
    """Turns several symptom columns into ONE set of 0/1 columns.
    Slot order doesn't matter: 'fever, cough' == 'cough, fever'."""

    def _rows(self, X):
        X = pd.DataFrame(X)
        return [
            [str(s).strip().lower() for s in row if pd.notna(s) and str(s).strip() != ""]
            for row in X.itertuples(index=False)
        ]

    def fit(self, X, y=None):
        self.mlb_ = MultiLabelBinarizer()
        self.mlb_.fit(self._rows(X))
        return self

    def transform(self, X):
        known = set(self.mlb_.classes_)
        rows = [[s for s in r if s in known] for r in self._rows(X)]  # unseen symptoms ignored
        return self.mlb_.transform(rows)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.mlb_.classes_, dtype=object)
