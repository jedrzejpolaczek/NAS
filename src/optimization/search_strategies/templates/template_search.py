"""
Module for defining an abstract base class for search optimization.
"""
import pandas as pd
from abc import ABC, abstractmethod

from src.models.templates.template_base_model import BaseModel


class BaseSearch(ABC):
    def __init__(
        self,
        config: dict
    ) -> None:
        """
        Initializes the BaseSearch with a configuration dictionary.

        Args:
            config (dict):
                Configuration dictionary for the search.
        """
        self.config = config

    @abstractmethod
    def optimize(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame
    ):
        """
        Perform search optimization on the model.

        Args:
            model (BaseModel):
                The model to be optimized.
            X_train (pd.DataFrame):
                Training input samples.
            y_train (pd.DataFrame):
                Target values.

        Returns:
            tuple
                The best parameters and the best score from the Grid Search.
        """
        raise NotImplementedError(
            "Subclasses must implement the fit method."
        )
