"""
Module for optimizing model's hyperparameters using Random Search with TensorFlow.
We should have in mind that "random" in these search strategies refers to the selection of hyperparameters
from a distribution, not the model's architecture itself. The model's architecture is fixed and defined
by the BaseModel class.
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
    Class for optimizing model's hyperparameters using Random Search with TensorFlow.

    Attributes:
        config (dict):
            Configuration for the Random Search, including param_distributions, n_iter, cv, and scoring.
        logger (logging.Logger):
            Logger instance for tracking optimization process.
    """

    def __init__(
        self,
        config: dict,
        logger: logging.Logger
    ) -> None:
        """
        Initializes the RandomSearchOptimizer with a configuration dictionary.

        Args:
            config (dict):
                Configuration dictionary for the search.
            logger (logging.Logger):
                Logger object shared in the project.
        """
        # Call the parent class (BaseSearch) initializer with config and logger
        # - Sets up shared attributes or methods from the base class
        super().__init__(config, logger)
        
        # Extract random search-specific settings from the config dictionary
        # - param_distributions: dict of parameter names to lists of possible values to sample from
        # - n_iter: number of random parameter combinations to test
        # - cv: number of folds for cross-validation
        # - scoring: metric to evaluate model performance (e.g., 'accuracy')
        self.param_distributions = self.config["config"]["param_distributions"]
        self.n_iter = self.config["config"]["n_iter"]
        self.cv = self.config["config"]["cv"]
        self.scoring = self.config["config"]["scoring"]

    def _generate_random_combinations(self) -> list:
        """
        Generate a fixed number of random parameter combinations from the param_distributions.

        Returns:
            list:
                List of dictionaries, each representing a random parameter combination.
        """
        # Initialize an empty list to store the random parameter combinations
        combinations = []
        
        # Get the hyperparameter names from the param_distributions dictionary
        # - e.g., ['learning_rate', 'batch_size']
        keys = list(self.param_distributions.keys())
        
        # Generate n_iter random combinations
        # - Loops self.n_iter times to create the specified number of samples
        for _ in range(self.n_iter):
            # Randomly sample one value for each hyperparameter from its distribution
            # - np.random.choice picks a value from each list in param_distributions
            # - Creates a dict mapping each key to a randomly chosen value
            params = {key: np.random.choice(self.param_distributions[key]) for key in keys}
            
            # Add the randomly generated parameter set to the list
            combinations.append(params)
        
        # Return the list of random combinations to be evaluated
        # - Length will be n_iter, each item a dict of parameter settings
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
                The model to evaluate.
            params (dict):
                Hyperparameters to set on the model.
            X (pd.DataFrame or np.ndarray):
                Training input features.
            y (pd.DataFrame, pd.Series, or np.ndarray):
                Target labels.

        Returns:
            float: Mean score across all folds.
        """
        # Convert input features and labels to TensorFlow tensors for efficient processing
        # - Handles pandas DataFrame/Series by extracting .values, otherwise assumes numpy
        X_tf = tf.convert_to_tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y, dtype=tf.int32)
        
        # Get the total number of samples in the dataset from the tensor shape
        n_samples = X_tf.shape[0]
        
        # Calculate the size of each fold for k-fold cross-validation
        # - Integer division ensures equal splits (except possibly the last fold)
        fold_size = n_samples // self.cv
        
        # Create and shuffle an array of indices for all samples
        # - Randomizes the data split to reduce bias in fold assignments
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Initialize a list to store the score for each fold
        scores = []
        
        # Iterate over each fold in the k-fold cross-validation
        for fold in range(self.cv):
            # Define the start and end indices for the validation set of this fold
            # - Ensures all samples are used, with the last fold possibly larger
            start = fold * fold_size
            end = start + fold_size if fold < self.cv - 1 else n_samples
            
            # Split indices into validation and training sets
            # - val_indices for validation, train_indices for training
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            # Split the tensor data into training and validation sets using the indices
            # - tf.gather extracts subsets based on the shuffled indices
            X_train = tf.gather(X_tf, train_indices)
            y_train = tf.gather(y_tf, train_indices)
            X_val = tf.gather(X_tf, val_indices)
            y_val = tf.gather(y_tf, val_indices)
            
            # Create a new instance of the model with its original config
            # - Ensures each fold uses a fresh model, avoiding carryover effects
            model_clone = type(model)(model.config, self.logger)
            
            # Apply the current random hyperparameters to the model
            model_clone.set_params(**params)
            
            # Convert tensors back to pandas for compatibility with the model's methods
            # - Preserves column names if input was a DataFrame
            X_train_df = pd.DataFrame(X_train.numpy(), columns=X.columns if isinstance(X, pd.DataFrame) else None)
            y_train_series = pd.Series(y_train.numpy())
            X_val_df = pd.DataFrame(X_val.numpy(), columns=X.columns if isinstance(X, pd.DataFrame) else None)
            
            # Train the model on the training subset
            model_clone.fit(X_train_df, y_train_series)
            
            # Predict on the validation subset
            y_pred = model_clone.predict(X_val_df)
            
            # Calculate the score for this fold based on the scoring metric
            # - Supports 'accuracy' now, with room for future expansion
            if self.scoring == "accuracy":
                score = np.mean(y_pred == y_val.numpy())
            else:
                raise ValueError(f"Scoring method '{self.scoring}' not supported yet.")
            
            # Store the score for this fold
            scores.append(score)
        
        # Return the average score across all folds
        # - Provides a robust performance estimate for the parameter set
        return np.mean(scores)

    def optimize(
        self,
        model: BaseModel,
        input_features_for_train: pd.DataFrame,
        target_labels_for_train: pd.DataFrame
    ) -> Tuple[Dict[str, Any], float]:
        """
        Perform Random Search hyperparameter optimization on the model using TensorFlow.

        Args:
            model (BaseModel):
                The model to be optimized.
            input_features_for_train (pd.DataFrame or np.ndarray):
                Training input features.
            target_labels_for_train (pd.DataFrame or np.ndarray):
                Target labels.

        Returns:
            tuple:
                The best parameters and the best score from the Random Search.
        """
        # Log the start of the random search process for tracking
        self.logger.info("Starting Random Search optimization...")
        
        # Generate a fixed number of random parameter combinations to test
        # - Uses n_iter to limit the search space unlike grid search's exhaustive approach
        param_combinations = self._generate_random_combinations()
        
        # Initialize variables to track the best performance
        # - best_score starts at negative infinity to ensure any score improves it
        best_score = -float('inf')
        best_params = None
        
        # Iterate over each randomly generated parameter combination
        for params in param_combinations:
            # Log the current parameters being tested for debugging
            self.logger.debug(f"Evaluating parameters: {params}")
            
            # Evaluate the model's performance with these parameters via cross-validation
            score = self._cross_validate(model, params, input_features_for_train, target_labels_for_train)
            
            # Log the score for this parameter set
            self.logger.debug(f"Score for params {params}: {score}")
            
            # Update the best score and parameters if this score is higher
            if score > best_score:
                best_score = score
                best_params = params
        
        # Log the final best results for record-keeping or analysis
        self.logger.debug(f"Best parameters: {best_params}, Best score: {best_score}")
        
        # Log the completion of the random search process
        self.logger.info("Random Search optimization completed.")
        
        # Return the best parameters and their score as a tuple
        # - Allows the caller to use the optimized settings
        return best_params, best_score
