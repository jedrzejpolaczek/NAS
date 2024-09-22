import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models.template_classification_model import ClassificationModel


class CustomRandomForestClassifier(ClassificationModel):
    """
    Custom Random Forest classifier model.

    Inherits from the `ClassificationModel` base class and provides a specific implementation
    for the Random Forest algorithm.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model = RandomForestClassifier(**self.config)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def  evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        y_pred = self.model.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),  
            "f1_score": f1_score(y, y_pred),  
        }
