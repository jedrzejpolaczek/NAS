"""
This module defines the `TabularDataLoader` class for
handling tabular datasets, such as the Iris dataset.
The class inherits from the abstract base class `DataLoader`
and implements methods for loading, preprocessing, and splitting tabular data.

The `TabularDataLoader` class provides functionality to:
1. Load tabular datasets based on a given dataset name.
2. Preprocess the loaded data by applying standard scaling.
3. Split the preprocessed data into training, validation, and test sets.

Classes:
    TabularDataLoader (DataLoader):
        A concrete implementation of the
        `DataLoader` class for tabular datasets.
        It supports loading datasets like Iris,
        preprocessing them by scaling,
        and splitting them into training, validation, and test sets.

Usage Example:
    # Initialize the data loader with a configuration dictionary
    config = {'test_size': 0.2, 'validation_size': 0.2}
    loader = TabularDataLoader(config)
    
    # Load and preprocess the Iris dataset
    data = loader.load_data('iris')
    preprocessed_data = loader.preprocess_data()
    
    # Split the data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = \
        loader.split_data(preprocessed_data)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from src.data.template_data_loader import DataLoader


class TabularDataLoader(DataLoader):
    """
    Data loader for tabular datasets, such as the Iris dataset.

    This class inherits from the `DataLoader` abstract base class and 
    implements the methods for loading and preprocessing tabular data.

    Attributes:
        config (dict):
            Configuration dictionary containing parameters for
            data loading, preprocessing, and splitting.
        data (dict):
            A dictionary to store the loaded data, typically containing
            features (X) and labels (y).

    Methods:
        load_data(dataset_name):
            Loads a tabular dataset based on the given dataset name.
        preprocess_data():
            Preprocesses the loaded tabular data (e.g., normalization).
    """

    def load_data(self, dataset_name):
        """
        Loads a tabular dataset based on the dataset name.

        This implementation currently supports loading the Iris dataset
        and can be extended to load other datasets.

        Args:
            dataset_name (str):
                The name of the dataset to load.
        """
        if dataset_name.lower() == "iris":
            iris = load_iris()
            x = pd.DataFrame(iris.data, columns=iris.feature_names)
            y = pd.Series(iris.target, name="species")
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

        self.data = {"x": x, "y": y}

    def preprocess_data(self):
        """
        Preprocesses the loaded tabular data.

        This implementation standardizes the features by removing the mean
        and scaling to unit variance.
        """
        if self.data is None:
            raise ValueError(
                "No data loaded. Please load data before preprocessing."
            )

        scaler = StandardScaler()
        self.data["x"] = pd.DataFrame(
            scaler.fit_transform(self.data["x"]), 
            columns=self.data["x"].columns
        )

