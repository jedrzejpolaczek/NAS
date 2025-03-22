"""
Module for optimizing model's hyperparameters using Grid Search with TensorFlow.
"""
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from abc import ABC
from typing import Tuple, Dict, Any

from src.models.templates.template_base_model import BaseModel
from src.optimization.search_strategies.templates.template_search import BaseSearch


class GridSearchOptimizer(BaseSearch, ABC):
    """
    Class for optimizing  model's hyperparameters using Grid Search with TensorFlow.
    Attributes:
        config (dict):
            Configuration for the Grid Search, including param_grid, cv, and scoring.
        logger (logging.Logger):
            Logger instance for tracking optimization process.
    """

    def __init__(
        self,
        config: dict,
        logger: logging.Logger
    ) -> None:
        """
        Initializes the GridSearchOptimizer with a configuration dictionary.

        Args:
            config (dict):
                Configuration dictionary for the search.
            logger (logging.Logger):
                Logger object shared in the project.
        """
        super().__init__(config, logger)
        self.param_grid = self.config["config"]["param_grid"]
        self.cv = self.config["config"]["cv"]
        self.scoring = self.config["config"]["scoring"]

    def _generate_param_combinations(self) -> list:
        """
        Generate all possible combinations of parameters from the param_grid.

        Returns:
            list:
                List of dictionaries, each representing a parameter combination.
        """
        keys = list(self.param_grid.keys())
        values = [self.param_grid[key] for key in keys]
        grid = np.array(np.meshgrid(*values)).T.reshape(-1, len(keys))
        combinations = [dict(zip(keys, combo)) for combo in grid]
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
            model (BaseModel):
                The model to evaluate (may be wrapped with a .model attribute).
            params (dict):
                Hyperparameters to set on the model.
            X (pd.DataFrame or np.ndarray):
                Training input features.
            y (pd.DataFrame, pd.Series, or np.ndarray):
                Target labels.

        Returns:
            float:
                Mean score across all folds.
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
        Perform Grid Search hyperparameter optimization on the model using TensorFlow.

        Args:
            model (BaseModel):
                The model to be optimized (may be wrapped).
            input_features_for_train (pd.DataFrame or np.ndarray):
                Training input features.
            target_labels_for_train (pd.DataFrame or np.ndarray):
                Target labels.

        Returns:
            tuple:
                The best parameters and the best score from the Grid Search.
        """
        self.logger.info("Starting Grid Search optimization...")
        
        param_combinations = self._generate_param_combinations()
        best_score = -float('inf')
        best_params = None
        
        for params in param_combinations:
            score = self._cross_validate(model, params, input_features_for_train, target_labels_for_train)

            if score > best_score:
                best_score = score
                best_params = params
        
        self.logger.debug(f"Best parameters: {best_params}, Best score: {best_score}")
        self.logger.info("Grid Search optimization completed.")
        return best_params, best_score
