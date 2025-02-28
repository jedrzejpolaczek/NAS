"""
Module for optimizing models using Random Search with TensorFlow.
"""
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from abc import ABC
from typing import Tuple, Dict, Any

from src.models.templates.template_base_model import BaseModel
from src.optimization.search_strategies.templates.template_search import BaseSearch


class RandomSearchOptimizer(BaseSearch, ABC):
    """
    Class for optimizing models using Random Search with TensorFlow.

    Attributes:
        config (dict): Configuration for the Random Search, including param_distributions, n_iter, cv, and scoring.
        logger (logging.Logger): Logger instance for tracking optimization process.
    """

    def __init__(
        self,
        config: dict,
        logger: logging.Logger
    ) -> None:
        """
        Initializes the RandomSearchOptimizer with a configuration dictionary.

        Args:
            config (dict): Configuration dictionary for the search, containing:
                - param_distributions: dict of hyperparameters with their possible values (lists or ranges).
                - n_iter: int, number of random combinations to try.
                - cv: int, number of cross-validation folds.
                - scoring: str, metric to optimize (e.g., 'accuracy').
            logger (logging.Logger): Logger object shared in the project.
        """
        super().__init__(config, logger)
        self.param_distributions = self.config["config"]["param_distributions"]
        self.n_iter = self.config["config"]["n_iter"]
        self.cv = self.config["config"]["cv"]
        self.scoring = self.config["config"]["scoring"]

    def _generate_random_combinations(self) -> list:
        """
        Generate a fixed number of random parameter combinations from the param_distributions.

        Returns:
            list: List of dictionaries, each representing a random parameter combination.
        """
        combinations = []
        keys = list(self.param_distributions.keys())
        
        for _ in range(self.n_iter):
            # Randomly sample one value for each parameter
            params = {key: np.random.choice(self.param_distributions[key]) for key in keys}
            combinations.append(params)
        
        return combinations

    def _cross_validate(
        self,
        model: BaseModel,
        params: Dict[str, Any],
        X: any,
        y: any
    ) -> float:
        """
        Perform k-fold cross-validation with the given parameters.

        Args:
            model (BaseModel): The model to evaluate.
            params (dict): Hyperparameters to set on the model.
            X (pd.DataFrame or np.ndarray): Training input features.
            y (pd.DataFrame, pd.Series, or np.ndarray): Target labels.

        Returns:
            float: Mean score across all folds.
        """
        # Convert to TensorFlow tensors
        X_tf = tf.convert_to_tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y, dtype=tf.int32)
        n_samples = X_tf.shape[0]
        
        fold_size = n_samples // self.cv
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        scores = []
        for fold in range(self.cv):
            start = fold * fold_size
            end = start + fold_size if fold < self.cv - 1 else n_samples
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            X_train = tf.gather(X_tf, train_indices)
            y_train = tf.gather(y_tf, train_indices)
            X_val = tf.gather(X_tf, val_indices)
            y_val = tf.gather(y_tf, val_indices)
            
            # Clone the actual model class with its original configuration
            model_clone = type(model)(model.config, self.logger)
            model_clone.set_params(**params)
            
            # Convert tensors back to pandas for model compatibility
            X_train_df = pd.DataFrame(X_train.numpy(), columns=X.columns if isinstance(X, pd.DataFrame) else None)
            y_train_series = pd.Series(y_train.numpy())
            X_val_df = pd.DataFrame(X_val.numpy(), columns=X.columns if isinstance(X, pd.DataFrame) else None)
            
            model_clone.fit(X_train_df, y_train_series)
            y_pred = model_clone.predict(X_val_df)
            
            if self.scoring == "accuracy":
                score = np.mean(y_pred == y_val.numpy())
            else:
                raise ValueError(f"Scoring method '{self.scoring}' not supported yet.")
            
            scores.append(score)
        
        return np.mean(scores)

    def optimize(
        self,
        model: BaseModel,
        input_features_for_train: pd.DataFrame,
        target_labels_for_train: pd.DataFrame
    ) -> Tuple[Dict[str, Any], float]:
        """
        Perform Random Search optimization on the model using TensorFlow.

        Args:
            model (BaseModel): The model to be optimized.
            input_features_for_train (pd.DataFrame or np.ndarray): Training input features.
            target_labels_for_train (pd.DataFrame or np.ndarray): Target labels.

        Returns:
            tuple: The best parameters and the best score from the Random Search.
        """
        self.logger.info("Starting Random Search optimization...")
        
        param_combinations = self._generate_random_combinations()
        best_score = -float('inf')
        best_params = None
        
        for params in param_combinations:
            self.logger.debug(f"Evaluating parameters: {params}")
            score = self._cross_validate(model, params, input_features_for_train, target_labels_for_train)
            self.logger.debug(f"Score for params {params}: {score}")
            
            if score > best_score:
                best_score = score
                best_params = params
        
        self.logger.debug(f"Best parameters: {best_params}, Best score: {best_score}")
        self.logger.info("Random Search optimization completed.")
        return best_params, best_score

