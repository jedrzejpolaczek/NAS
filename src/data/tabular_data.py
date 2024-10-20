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
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from src.data.templates.template_data_loader import BaseDataLoader


class TabularDataLoader(BaseDataLoader):
    """
    Data loader for tabular datasets, such as the Iris dataset.

    This class inherits from the `BaseDataLoader` abstract base class and 
    implements the methods for loading and preprocessing tabular data.

    Attributes:
        config (dict):
            Configuration dictionary containing parameters for
            data loading, preprocessing, and splitting.
        data (dict):
            A dictionary to store the loaded data, typically containing
            features (X) and labels (y).

    Methods:
        download_data():
            Downloads specified dataset and saves it to a CSV file.    
        load_data():
            Loads a tabular dataset based on the given dataset name.
        preprocess_data():
            Preprocesses the loaded tabular data (e.g., normalization).
    """
    def download_data(self) -> None:
        """
        Downloads specified dataset and saves it to a CSV file.

        This function currently supports only the Iris dataset.

        Raises:
            ValueError:
                If the specified dataset is not supported.

        Returns:
            None
        """
        self._create_directory()

        if self.config["dataset_name"].lower() == "iris":
            iris = load_iris()

            # Create a DataFrame from the dataset for easier manipulation
            iris_df = pd.DataFrame(
                            data= iris.data,
                            columns= iris.feature_names
                        )

            # Also add the target variable to the DataFrame
            iris_df["target"] = iris.target

            # Save the dataset to a CSV file
            iris_df.to_csv(self.file_path, index=False)

        elif self.config.get("dataset_name") is None:
            error_msg = "Missing dataset name in configuration file."
            raise ValueError(error_msg)

        else:
            dataset_name = self.config.get("dataset_name")
            raise ValueError(f"Dataset {dataset_name} is not supported.")

    def load_data(self) -> None:
        """
        Loads a tabular dataset based on the dataset name.

        This implementation currently supports loading the Iris dataset
        and can be extended to load other datasets.
        """
        if self.config.get("dataset_name").lower() == "iris":
            self.data = pd.read_csv(self.file_path)

        elif self.config.get("dataset_name") is None:
            error_msg = "Missing dataset name in configuration file."
            raise ValueError(error_msg)

        else:
            dataset_name = self.config.get("dataset_name")
            raise ValueError(f"Dataset {dataset_name} is not supported.")

    def preprocess_data(self) -> None:
        """
        Preprocesses loaded data for further analysis.

        Raises:
            ValueError:
                If no data is loaded or the data structure is unsupported.

        Return:
            None
        """

        if self.data is None:
            raise ValueError(
                "No data loaded. Please load data before preprocessing."
            )

        if not isinstance(self.data, pd.DataFrame):
            raise ValueError(
                "Unsupported data structure for preprocessing. \
                    Only DataFrames are supported."
            )

        # Separate features and target in one line
        x, y = self.data.drop("target", axis=1), self.data["target"]

        # Impute missing values and convert to numeric in one step
        x = x.apply(pd.to_numeric, errors="coerce").ffill()

        # Standardize features with fit_transform for efficiency
        scaler = StandardScaler().fit_transform(x)

        # Update data directly, avoiding unnecessary dictionary creation
        self.data = {"X": scaler, "y": y}

    def _create_directory(self) -> None:
        """
        Creates a directory at the specified path.
        If the directory already exists, it does nothing.
        If there are any subdirectories in the path, it creates them as well.

        Raises:
            FileExistsError:
                Path alread exist.

        Returns:
            None
        """
        dataset_path = self.config["path"]

        try:
            os.makedirs(dataset_path)
            self.logger.debug(
                f"Directory '{dataset_path}' was created successfully."
            )
        except FileExistsError:
            self.logger.warning(
                f"Directory '{dataset_path}' already exists."
            )
