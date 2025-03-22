"""
Module for optimizing model's hyperparameters using Grid Search with TensorFlow.
We should have in mind that "grid" in these search strategies refers to the exhaustive search
over all possible combinations of hyperparameters.
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
            Configuration dictionary for the grid search.
            It must contain a nested 'config' key with the following sub-keys:
            - 'param_grid' (dict):
                A dictionary where keys are hyperparameter names (e.g., 'learning_rate', 'batch_size')
                and values are lists of possible settings to explore (e.g., [0.01, 0.1, 1.0]).
                This defines the grid of hyperparameter combinations to search over exhaustively.
            - 'cv' (int):
                The number of folds for k-fold cross-validation.
                For example, if cv=5, the training data will be split into 5 parts,
                with 4 used for training and 1 for validation in each iteration,
                to assess model performance robustly across different data subsets.
            - 'scoring' (str):
                The metric used to evaluate model performance during cross-validation.
                Currently supports 'accuracy' (fraction of correct predictions),
                with potential for additional metrics (e.g., 'f1', 'loss') in future implementations.
        logger (logging.Logger):
            Logger object shared in the project for tracking the optimization process,
            such as logging parameter combinations tested and their resulting scores.
        """
        super().__init__(config, logger)
        self.param_grid = self.config["config"]["param_grid"]
        self.cv = self.config["config"]["cv"]
        self.scoring = self.config["config"]["scoring"]

    def _generate_param_combinations(self) -> list[dict]:
        """
        Generate all possible combinations of parameters from the param_grid.

        Returns:
            list[dict]:
                List of dictionaries, each representing a parameter combination.
        """
        # Extract the hyperparameter names (e.g., 'learning_rate', 'batch_size') from the param_grid dictionary
        keys: list = list(self.param_grid.keys())
        
        # Create a list of possible values for each hyperparameter (e.g., [0.01, 0.1], [32, 64])
        values: list = [self.param_grid[key] for key in keys]
        
        # Use np.meshgrid to create a grid of all possible combinations of the values 
        # (structured way of representing every possible pairing of values from multiple sets of options):
        # - *values unpacks the list of value arrays into separate arguments for meshgrid
        # - This creates a multidimensional array where each dimension corresponds to one hyperparameter's values
        # Example, for leatning rate [0.01, 0.1] and batch size [32, 64], we get a 2x2 grid of combinations:
        # {'learning_rate': 0.01, 'batch_size': 32}
        # {'learning_rate': 0.01, 'batch_size': 64}
        # {'learning_rate': 0.1, 'batch_size': 32}
        # {'learning_rate': 0.1, 'batch_size': 64}
        grid: np.array = np.array(np.meshgrid(*values))
        
        # Reshape the grid into a 2D array where:
        # - Each row is a unique combination of parameters
        # - Each column corresponds to a hyperparameter
        # - T transposes the array, and reshape(-1, len(keys)) flattens it into the desired shape
        # Reshaping the grid into a 2D array is necessary to transform the multidimensional
        # output of np.meshgrid into a format where each row represents a single combination of hyperparameter values,
        # making it easy to pair with hyperparameter names for the grid search.
        # This ensures the data is structured as (n_combinations, n_params)
        # for straightforward conversion into a list of dictionaries.
        grid: list = grid.T.reshape(-1, len(keys))
        
        # Convert each row of the grid into a dictionary mapping hyperparameter names to their values
        # - zip(keys, combo) pairs each key with its corresponding value in the combination
        # - dict() creates a dictionary for each combination
        combinations: list = [dict(zip(keys, combination)) for combination in grid]
        
        # Return the list of all parameter combinations to be tested in the grid search
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
        # Convert input features and labels to TensorFlow tensors for efficient processing
        # - Handles pandas DataFrame by extracting .values, otherwise assumes numpy array
        # - Specifies float32 for features and int32 for labels to match typical ML data types
        X_tf = tf.convert_to_tensor(
            X.values if isinstance(X, pd.DataFrame) else X, dtype=tf.float32
        )
        y_tf = tf.convert_to_tensor(
            y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y, dtype=tf.int32
        )
        
        # Get the total number of samples in the dataset to calculate fold sizes
        n_samples = X_tf.shape[0]
        
        # Calculate the size of each fold for k-fold cross-validation
        # - Divides total samples by the number of folds (self.cv), using integer division
        fold_size = n_samples // self.cv
        
        # Create an array of indices for all samples and shuffle them randomly
        # - This ensures folds are random subsets of the data, reducing bias
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Initialize a list to store the score for each fold
        scores: list = []
        
        # Iterate over each fold in the k-fold cross-validation
        for fold in range(self.cv):
            # Calculate the start and end indices for the validation set of the current fold
            # - Start is fold number times fold size
            # - End is either the next fold's start or the dataset end for the last fold
            start = fold * fold_size
            end = start + fold_size if fold < self.cv - 1 else n_samples
            
            # Split indices into validation and training sets
            # - val_indices: samples used for validation in this fold
            # - train_indices: all other samples, used for training
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            # Use TensorFlow's gather to split the data based on the shuffled indices
            # - Extracts training and validation subsets from the tensors
            X_train = tf.gather(X_tf, train_indices)
            y_train = tf.gather(y_tf, train_indices)
            X_val = tf.gather(X_tf, val_indices)
            y_val = tf.gather(y_tf, val_indices)
            
            # Create a fresh copy of the model with its original configuration
            # - Uses type(model) to instantiate a new instance, avoiding modification of the original
            model_clone = type(model)(model.config, self.logger)
            
            # Set the current hyperparameters on the cloned model
            model_clone.set_params(**params)
            
            # Convert training tensors back to pandas DataFrame and Series
            # - Ensures compatibility with the model's fit method, which may expect pandas input
            # - Preserves column names if X was a DataFrame, otherwise uses None
            X_train_df = pd.DataFrame(X_train.numpy(), columns=X.columns if isinstance(X, pd.DataFrame) else None)
            y_train_series = pd.Series(y_train.numpy())
            X_val_df = pd.DataFrame(X_val.numpy(), columns=X.columns if isinstance(X, pd.DataFrame) else None)
            
            # Train the model on the training subset
            model_clone.fit(X_train_df, y_train_series)
            
            # Make predictions on the validation subset
            y_pred = model_clone.predict(X_val_df)
            
            # Calculate the score for this fold based on the specified scoring metric
            # - Currently supports 'accuracy' (fraction of correct predictions)
            # - Raises an error if an unsupported metric is specified
            if self.scoring == "accuracy":
                score = np.mean(y_pred == y_val.numpy())
            else:
                raise ValueError(f"Scoring method '{self.scoring}' not supported yet.")
            
            # Add the fold's score to the list
            scores.append(score)
        
        # Compute and return the mean score across all folds
        # - Provides a robust estimate of model performance for these parameters
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
        # Log the start of the optimization process to track progress
        self.logger.info("Starting Grid Search optimization...")
        
        # Generate all possible hyperparameter combinations to test, using the param_grid
        param_combinations = self._generate_param_combinations()
        
        # Initialize variables to track the best performance
        # - best_score starts at negative infinity so any real score will be better
        # - best_params will store the parameters yielding the highest score
        best_score = -float('inf')
        best_params = None
        
        # Iterate over each combination of hyperparameters
        for params in param_combinations:
            # Evaluate the model's performance with the current parameter set
            # - _cross_validate runs k-fold cross-validation and returns the mean score
            score = self._cross_validate(
                model=model, 
                params=params, 
                X=input_features_for_train,
                y=target_labels_for_train
            )
            
            # Check if this score is better than the previous best
            # - If so, update best_score and best_params to reflect the new optimum
            if score > best_score:
                best_score = score
                best_params = params
        
        # Log the results of the optimization for debugging or record-keeping
        # - Includes the best parameters and their corresponding score
        self.logger.debug(f"Best parameters: {best_params}, Best score: {best_score}")
        
        # Log the completion of the process to indicate the search is finished
        self.logger.info("Grid Search optimization completed.")
        
        # Return the best parameters and their score as a tuple
        # - This allows the caller to use the optimized settings for the model
        return (best_params, best_score)
