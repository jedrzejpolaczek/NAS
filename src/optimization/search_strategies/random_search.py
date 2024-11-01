import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from typing import (Dict, Any, Tuple)

from src.optimization.search_strategies.templates.template_search import BaseSearch


class RandomSearchOptimizer(BaseSearch):
    """
    Random search optimization strategy using sklearn's RandomizedSearchCV.

    This class implements random search for
    hyperparameter optimization of machine learning models.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the RandomSearchOptimizer.

        Args:
            config: Configuration dictionary containing:
                - param_distributions
                     Dict of parameters and their distributions
                - cv:
                    Number of cross-validation folds
                - n_iter:
                    Number of random samples

        Raises:
            ValueError:
                If required configuration parameters are missing
        """
        # Parameter validation
        required_params = [
            "param_distributions",
            "cv",
            "n_iter"
        ]
        if not all(param in config for param in required_params):
            raise ValueError(
                f"Missing required parameters. Expected: {required_params}"
            )

        self.config = config

    def optimize(
        self,
        model: Any,
        input_features_for_train: np.ndarray,
        target_labels_for_train: np.ndarray
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize model hyperparameters using random search.

        Args:
            model (Any):
                The machine learning model to optimize.
            input_features_for_train (np.ndarray):
                Training features.
            target_labels_for_train (np.ndarray):
                Training labels

        Returns:
            Tuple containing:
                - Dictionary of best parameters
                - Best score achieved

        Raises:
            ValueError:
                If input data is invalid
        """
        if len(input_features_for_train) != len(target_labels_for_train):
            raise ValueError(
                "input_features_for_train and target_labels_for_train \
                    must have the same length"
            )

        try:
            random_search = RandomizedSearchCV(
                model,
                self.config["param_distributions"],
                cv=self.config["cv"],
                n_iter=self.config["n_iter"],
                n_jobs=-1,  # Utilize all available cores
                verbose=2   # Enable progress logging
            )

            random_search.fit(input_features_for_train, target_labels_for_train)

            return random_search.best_params_, random_search.best_score_

        except Exception as e:
            raise ValueError(f"Optimization failed: {str(e)}") from e
