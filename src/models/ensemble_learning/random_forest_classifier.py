"""
Module for implementing a Random Forest Classifier using TensorFlow.
- Provides a custom implementation of a random forest for classification tasks.
- Integrates TensorFlow for tensor operations and pandas/NumPy for data handling.
"""
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from src.models.templates.template_classification_model import ClassificationModel


class RandomForestClassifier(ClassificationModel):
    def __init__(
        self,
        config: dict,
        logger: logging.Logger
    ) -> None:
        """
        Initializes the RandomForestClassifier with configuration and logger.

        Args:
            config (dict):
                Configuration dictionary containing model parameters.
            logger (logging.Logger):
                Logger object for tracking model operations.
        """
        # Call the parent class (ClassificationModel) initializer with config and logger
        # - Sets up shared attributes or methods from the base class
        super().__init__(config, logger)
        
        # Extract random forest hyperparameters from config with defaults
        # - n_estimators: number of trees in the forest (default 100)
        # - max_depth: maximum depth of each tree (default 10)
        # - min_samples_split: minimum samples required to split a node (default 2)
        self.n_estimators = self.config["config"].get("n_estimators", 100)
        self.max_depth = self.config["config"].get("max_depth", 10)
        self.min_samples_split = self.config["config"].get("min_samples_split", 2)
        
        # Initialize model as None; will store the list of decision trees after fitting
        self.model = None
        
        # Initialize classes_ as None; will store unique class labels after fitting
        self.classes_ = None

    def _build_tree(self, X: tf.Tensor, y: tf.Tensor, depth: int = 0) -> dict:
        """
        Recursively build a decision tree for the random forest.

        Args:
            X (tf.Tensor):
                Input features as a TensorFlow tensor.
            y (tf.Tensor):
                Target labels as a TensorFlow tensor.
            depth (int):
                Current depth of the tree (default 0).

        Returns:
            dict:
                Dictionary representing a tree node (leaf or split).
        """
        # Get the number of samples in the current subset
        n_samples = tf.shape(X)[0].numpy()
        
        # Check stopping conditions for creating a leaf node
        # - If max depth reached, too few samples, or all labels are the same
        if (depth >= self.max_depth or n_samples < self.min_samples_split or 
            tf.reduce_all(y == y[0])):
            # Count unique labels and their frequencies
            values, _, counts = tf.unique_with_counts(y)
            # Return a leaf node with the most common class
            return {"leaf": True, "class": values[tf.argmax(counts)].numpy()}
        
        # Get the number of features for random selection
        n_features = tf.shape(X)[1].numpy()
        
        # Randomly select a feature index to split on
        # - Randomness is a key aspect of random forests
        feature_idx = np.random.randint(0, n_features)
        
        # Extract feature values for the selected feature
        feature_values = X[:, feature_idx]
        
        # Use the mean of the feature values as the split threshold
        threshold = tf.reduce_mean(feature_values).numpy()
        
        # Create masks for left and right splits based on the threshold
        left_mask = feature_values <= threshold
        right_mask = feature_values > threshold
        
        # If either split is empty, stop and return a leaf node
        # - Prevents degenerate trees with no further splits
        if tf.reduce_sum(tf.cast(left_mask, tf.int32)) == 0 or tf.reduce_sum(tf.cast(right_mask, tf.int32)) == 0:
            values, _, counts = tf.unique_with_counts(y)
            return {"leaf": True, "class": values[tf.argmax(counts)].numpy()}
        
        # Recursively build left and right subtrees
        # - Increases depth by 1 for each recursive call
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        # Return a split node with feature, threshold, and child trees
        return {
            "leaf": False,
            "feature_idx": feature_idx,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree
        }

    def _to_tensor(self, data: any) -> tf.Tensor:
        """
        Convert input data to a TensorFlow tensor.

        Args:
            data (any):
                Input data (pandas DataFrame, Series, or NumPy array).

        Returns:
            tf.Tensor:
                Converted tensor with appropriate dtype.
        """
        # Handle pandas DataFrame or Series by extracting values
        # - Uses float32 for features (DataFrame), int32 for labels (Series)
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return tf.convert_to_tensor(data.values, dtype=tf.float32 if isinstance(data, pd.DataFrame) else tf.int32)
        
        # Handle NumPy arrays, checking dimensionality for dtype
        # - 2D arrays (features) get float32, 1D (labels) get int32
        elif isinstance(data, np.ndarray):
            return tf.convert_to_tensor(data, dtype=tf.float32 if len(data.shape) > 1 else tf.int32)
        
        # Raise an error for unsupported data types
        else:
            raise ValueError("Input data must be a pandas DataFrame, Series, or NumPy array.")

    def fit(
        self,
        input_features: pd.DataFrame,
        target_labels: pd.Series
    ) -> None:
        """
        Fit the random forest model to the training data.

        Args:
            input_features (pd.DataFrame):
                Training input features.
            target_labels (pd.Series):
                Training target labels.
        """
        # Convert features and labels to TensorFlow tensors
        X = self._to_tensor(input_features)
        y = self._to_tensor(target_labels)
        
        # Store unique class labels for later use (e.g., predict_proba)
        self.classes_ = np.unique(target_labels if isinstance(target_labels, (pd.Series, np.ndarray)) else target_labels.values)
        
        # Get the number of samples in the dataset
        n_samples = X.shape[0]
        
        # Initialize the model as a list to store decision trees
        self.model = []
        
        # Build n_estimators trees with bootstrap sampling
        # - Each tree uses a random subset of the data with replacement
        for _ in range(self.n_estimators):
            # Randomly sample indices with replacement (bootstrap)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            
            # Extract the sampled subset of data
            X_sample = tf.gather(X, indices)
            y_sample = tf.gather(y, indices)
            
            # Build a decision tree with the sampled data
            tree = self._build_tree(X_sample, y_sample)
            
            # Add the tree to the forest
            self.model.append(tree)

    def _predict_tree(self, tree: dict, x: tf.Tensor) -> int:
        """
        Predict the class for a single sample using a decision tree.

        Args:
            tree (dict):
                Decision tree node.
            x (tf.Tensor):
                Single sample as a tensor.

        Returns:
            int:
                Predicted class label.
        """
        # If at a leaf node, return the class label
        if tree["leaf"]:
            return tree["class"]
        
        # Get the feature value for the split feature
        feature_value = x[tree["feature_idx"]]
        
        # Recursively traverse left or right based on the threshold
        if feature_value <= tree["threshold"]:
            return self._predict_tree(tree["left"], x)
        else:
            return self._predict_tree(tree["right"], x)

    def predict(
        self,
        input_features: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict class labels for input features using the random forest.

        Args:
            input_features (pd.DataFrame):
                Features to predict on.

        Returns:
            np.ndarray:
                Array of predicted class labels.
        """
        # Convert input features to a TensorFlow tensor
        X = self._to_tensor(input_features)
        
        # Get the number of samples to predict
        n_samples = X.shape[0]
        
        # Initialize an array to store predictions from each tree
        # - Shape is (n_samples, n_estimators)
        predictions = np.zeros((n_samples, len(self.model)), dtype=np.int32)
        
        # Iterate over each tree in the forest
        for i, tree in enumerate(self.model):
            # Predict for each sample using the current tree
            for j in range(n_samples):
                predictions[j, i] = self._predict_tree(tree, X[j])
        
        # Aggregate predictions by majority vote
        # - np.bincount counts votes per sample, argmax selects the most common class
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)

    def predict_proba(
        self,
        input_features: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict class probabilities for input features using the random forest.

        Args:
            input_features (pd.DataFrame):
                Features to predict probabilities for.

        Returns:
            np.ndarray:
                Array of class probabilities for each sample.
        """
        # Convert input features to a TensorFlow tensor
        X = self._to_tensor(input_features)
        
        # Get the number of samples to predict
        n_samples = X.shape[0]
        
        # Initialize an array to store predictions from each tree
        predictions = np.zeros((n_samples, len(self.model)), dtype=np.int32)
        
        # Collect predictions from each tree
        for i, tree in enumerate(self.model):
            for j in range(n_samples):
                predictions[j, i] = self._predict_tree(tree, X[j])
        
        # Initialize probability array for all classes
        # - Shape is (n_samples, n_classes)
        proba = np.zeros((n_samples, len(self.classes_)))
        
        # Calculate probabilities as the fraction of trees voting for each class
        for i in range(n_samples):
            counts = np.bincount(predictions[i], minlength=len(self.classes_))
            proba[i] = counts / len(self.model)
        
        # Return the probability distribution for each sample
        return proba

    def evaluate(
        self,
        input_features: pd.DataFrame,
        target_labels: pd.Series
    ) -> dict:
        """
        Evaluate the model’s performance on the given data.

        Args:
            input_features (pd.DataFrame):
                Features to evaluate on.
            target_labels (pd.Series):
                True labels for evaluation.

        Returns:
            dict:
                Dictionary with accuracy, precision, recall, and F1 score.
        """
        # Get predictions for the input features
        predictions = self.predict(input_features)
        
        # Extract true labels as a numpy array
        y_true = target_labels.values if isinstance(target_labels, pd.Series) else target_labels
        
        # Calculate accuracy as the fraction of correct predictions
        accuracy = np.mean(predictions == y_true)
        
        # Determine the averaging method for multiclass metrics from config
        average_method = self.config["evaluation"]["average"]
        
        # Handle binary or two-class cases separately
        if average_method == "binary" or len(self.classes_) == 2:
            # Calculate true positives, false positives, and false negatives
            tp = np.sum((predictions == 1) & (y_true == 1))
            fp = np.sum((predictions == 1) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))
            
            # Compute precision, recall, and F1 with zero checks
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            # Multiclass case: compute metrics per class and average
            precision, recall, f1 = [], [], []
            for cls in self.classes_:
                tp = np.sum((predictions == cls) & (y_true == cls))
                fp = np.sum((predictions == cls) & (y_true != cls))
                fn = np.sum((predictions != cls) & (y_true == cls))
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
                precision.append(p)
                recall.append(r)
                f1.append(f)
            precision = np.mean(precision)
            recall = np.mean(recall)
            f1 = np.mean(f1)

        # Return a dictionary with all computed metrics
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def get_params(self, deep: bool = True) -> dict:
        """
        Get the model's parameters.

        Args:
            deep (bool):
                If True, return a deep copy (unused here but included for compatibility).

        Returns:
            dict:
                Dictionary of model parameters.
        """
        # Return a dictionary of all relevant parameters
        # - Includes hyperparameters, config, and logger for full state
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "config": self.config,
            "logger": self.logger
        }

    def set_params(self, **params) -> 'RandomForestClassifier':
        """
        Set the model's parameters.

        Args:
            **params:
                Keyword arguments of parameters to set.

        Returns:
            RandomForestClassifier:
                Self for method chaining.
        """
        # Iterate over provided parameters to update the model
        for param, value in params.items():
            # Update specific hyperparameters directly
            if param in ["n_estimators", "max_depth", "min_samples_split"]:
                setattr(self, param, value)
            
            # Handle config updates, resetting hyperparameters with new values
            elif param == "config":
                self.config = value
                self.n_estimators = self.config["config"].get("n_estimators", self.n_estimators)
                self.max_depth = self.config["config"].get("max_depth", self.max_depth)
                self.min_samples_split = self.config["config"].get("min_samples_split", self.min_samples_split)
            
            # Update the logger if provided
            elif param == "logger":
                self.logger = value
        
        # Return self to allow method chaining (e.g., model.set_params(...).fit(...))
        return self