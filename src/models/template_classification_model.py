from abc import abstractmethod
import pandas as pd
import numpy as np

from src.models.template_base_model import BaseModel


class ClassificationModel(BaseModel):
    """
    Base class for classification models.

    Inherits from `BaseModel` and provides additional methods specific to classification tasks.
    """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts class probabilities for new data.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        raise NotImplementedError("Subclasses must implement the predict_proba method.")
