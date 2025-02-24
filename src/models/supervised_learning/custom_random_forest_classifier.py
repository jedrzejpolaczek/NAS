# No changes needed from the last version; included for reference
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from src.models.templates.template_classification_model import ClassificationModel


class CustomRandomForestClassifier(ClassificationModel):
    def __init__(
        self,
        config: dict,
        logger: logging.Logger
    ) -> None:
        super().__init__(config, logger)
        self.n_estimators = self.config["config"].get("n_estimators", 100)
        self.max_depth = self.config["config"].get("max_depth", 10)
        self.min_samples_split = self.config["config"].get("min_samples_split", 2)
        self.model = None
        self.classes_ = None

    def _build_tree(self, X: tf.Tensor, y: tf.Tensor, depth: int = 0) -> dict:
        n_samples = tf.shape(X)[0].numpy()
        if (depth >= self.max_depth or n_samples < self.min_samples_split or 
            tf.reduce_all(y == y[0])):
            values, _, counts = tf.unique_with_counts(y)
            return {"leaf": True, "class": values[tf.argmax(counts)].numpy()}
        
        n_features = tf.shape(X)[1].numpy()
        feature_idx = np.random.randint(0, n_features)
        feature_values = X[:, feature_idx]
        threshold = tf.reduce_mean(feature_values).numpy()
        
        left_mask = feature_values <= threshold
        right_mask = feature_values > threshold
        
        if tf.reduce_sum(tf.cast(left_mask, tf.int32)) == 0 or tf.reduce_sum(tf.cast(right_mask, tf.int32)) == 0:
            values, _, counts = tf.unique_with_counts(y)
            return {"leaf": True, "class": values[tf.argmax(counts)].numpy()}
        
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            "leaf": False,
            "feature_idx": feature_idx,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree
        }

    def _to_tensor(self, data: any) -> tf.Tensor:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return tf.convert_to_tensor(data.values, dtype=tf.float32 if isinstance(data, pd.DataFrame) else tf.int32)
        elif isinstance(data, np.ndarray):
            return tf.convert_to_tensor(data, dtype=tf.float32 if len(data.shape) > 1 else tf.int32)
        else:
            raise ValueError("Input data must be a pandas DataFrame, Series, or NumPy array.")

    def fit(
        self,
        input_features: pd.DataFrame,
        target_labels: pd.Series
    ) -> None:
        X = self._to_tensor(input_features)
        y = self._to_tensor(target_labels)
        self.classes_ = np.unique(target_labels if isinstance(target_labels, (pd.Series, np.ndarray)) else target_labels.values)
        
        n_samples = X.shape[0]
        self.model = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = tf.gather(X, indices)
            y_sample = tf.gather(y, indices)
            tree = self._build_tree(X_sample, y_sample)
            self.model.append(tree)

    def _predict_tree(self, tree: dict, x: tf.Tensor) -> int:
        if tree["leaf"]:
            return tree["class"]
        
        feature_value = x[tree["feature_idx"]]
        if feature_value <= tree["threshold"]:
            return self._predict_tree(tree["left"], x)
        else:
            return self._predict_tree(tree["right"], x)

    def predict(
        self,
        input_features: pd.DataFrame
    ) -> np.ndarray:
        X = self._to_tensor(input_features)
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.model)), dtype=np.int32)
        
        for i, tree in enumerate(self.model):
            for j in range(n_samples):
                predictions[j, i] = self._predict_tree(tree, X[j])
        
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)

    def predict_proba(
        self,
        input_features: pd.DataFrame
    ) -> np.ndarray:
        X = self._to_tensor(input_features)
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.model)), dtype=np.int32)
        
        for i, tree in enumerate(self.model):
            for j in range(n_samples):
                predictions[j, i] = self._predict_tree(tree, X[j])
        
        proba = np.zeros((n_samples, len(self.classes_)))
        for i in range(n_samples):
            counts = np.bincount(predictions[i], minlength=len(self.classes_))
            proba[i] = counts / len(self.model)
        
        return proba

    def evaluate(
        self,
        input_features: pd.DataFrame,
        target_labels: pd.Series
    ) -> dict:
        predictions = self.predict(input_features)
        y_true = target_labels.values if isinstance(target_labels, pd.Series) else target_labels
        
        accuracy = np.mean(predictions == y_true)
        
        average_method = self.config["evaluation"]["average"]
        if average_method == "binary" or len(self.classes_) == 2:
            tp = np.sum((predictions == 1) & (y_true == 1))
            fp = np.sum((predictions == 1) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
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

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "config": self.config,
            "logger": self.logger
        }

    def set_params(self, **params) -> 'CustomRandomForestClassifier':
        for param, value in params.items():
            if param in ["n_estimators", "max_depth", "min_samples_split"]:
                setattr(self, param, value)
            elif param == "config":
                self.config = value
                self.n_estimators = self.config["config"].get("n_estimators", self.n_estimators)
                self.max_depth = self.config["config"].get("max_depth", self.max_depth)
                self.min_samples_split = self.config["config"].get("min_samples_split", self.min_samples_split)
            elif param == "logger":
                self.logger = value
        return self
