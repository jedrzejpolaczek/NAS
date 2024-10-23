"""
Module for optimizing models using Grid Search.
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.optimization.search_strategies.templates.template_search import BaseSearch
from src.models.templates.template_base_model import BaseModel


class GridSearchOptimizer(BaseSearch):
    """
    Class for optimizing models using Grid Search.

    Attributes:
        config (dict):
            Configuration for the Grid Search.
    """
    def optimize(
        self,
        model: BaseModel,
        input_features_for_train: pd.DataFrame,
        target_labels_for_train: pd.DataFrame
    ):
        """
        Perform Grid Search optimization on the model.

        Args:
            model (BaseModel):
                The model to be optimized.
            input_features_for_train (pd.DataFrame):
                Training input features. Commonly marked as X.
            target_labels_for_train (pd.DataFrame):
                Target labels values. Commonly marked as y.

        Returns:
            tuple
                The best parameters and the best score from the Grid Search.
        """
        grid_search = GridSearchCV(
            model,
            self.config["config"]["param_grid"],
            cv=self.config["config"]["cv"],
            scoring=self.config["config"]["scoring"]
        )

        grid_search.fit(
            input_features_for_train,
            target_labels_for_train
        )

        return grid_search.best_params_, grid_search.best_score_
