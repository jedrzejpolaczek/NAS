"""
This module defines an base class `DataLoader`
for loading and preprocessing datasets in a machine learning pipeline.
The `DataLoader` class provides a template for implementing
specific dataset loaders that can handle various types of data
(e.g., tabular, image, text) and offers functionality to
split data into training, validation, and test sets.

Classes:
    DataLoader(ABC):
        An base class for dataset loading and preprocessing.

Usage Example:
    # Define a custom data loader by inheriting from DataLoader
    class CustomDataLoader(DataLoader):
        def load_data(self, dataset_name):
            # Custom logic for loading the dataset
            pass

        def preprocess_data(self):
            # Custom logic for preprocessing the dataset
            pass

    # Initialize the custom loader with a configuration dictionary
    config = {"test_size": 0.2, "validation_size": 0.2}
    loader = CustomDataLoader(config)
    data = loader.load_data("dataset_name")
    loader.preprocess_data()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(data)
"""
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Base class for loading and preprocessing datasets.

    This class provides a template for creating specific dataset loaders. 
    It defines the structure for loading data, preprocessing data, and splitting 
    the dataset into training, validation, and test sets.

    Attributes:
        config (dict):
            A configuration dictionary that contains parameters for 
            data loading, preprocessing, and splitting.
        data (dict):
            A dictionary to store the loaded data, typically containing
            features (X) and labels (y).

    Methods:
        load_data(dataset_name):
            Abstract method for loading dataset based on 
            the given dataset name. Must be implemented in the subclass.
        preprocess_data():
            Abstract method for preprocessing the loaded data. 
            Must be implemented in the subclass.
        split_data(data):
            Splits the data into training, validation, and test 
            sets based on the configuration provided.
    """
    def __init__(self, config: dict) -> None:
        """
        Initializes the DataLoader with a configuration dictionary.

        Args:
            config (dict):
                Configuration dictionary containing parameters for
                data loading, preprocessing, and splitting.
        """
        self.config = config
        self.data = None

    def get_data(self) -> dict:
        """
        Retrieves the currently loaded data.

        This method returns the data that has been loaded and potentially
        preprocessed by the `DataLoader`. The data is typically a dictionary
        containing features and labels.

        Returns:
            dict:
                A dictionary containing the loaded features and labels, with
                keys like 'x' for features and 'y' for labels.
        """
        return self.data

    def load_data(self, dataset_name: str) -> None:
        """
        Loads data based on the dataset name.

        This method should be overridden by subclasses to handle the 
        specific logic required to load different types of datasets.

        Args:
            dataset_name (str):
                The name of the dataset to load.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def preprocess_data(self) -> None:
        """
        Preprocesses the loaded data.

        This method should be overridden by subclasses to handle the 
        specific preprocessing steps required for different types of datasets.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def split_data(self, data: dict) -> tuple:
        """
        Splits the data into training, validation, and test sets.

        This method uses the configuration provided to determine the sizes
        of the test and validation sets.

        Args:
            data (dict):
                A dictionary containing the
                features (X) and labels (y) to be split.

        Returns:
            tuple:
                Six arrays representing the training, validation, and test
                sets for features and labels respectively: 
                (x_train, x_val, x_test, y_train, y_val, y_test).
        """
        test_size = self.config.get("test_size", 0.2)
        validation_size = self.config.get("validation_size", 0.2)

        # Split into training and a temporary set
        # (which will be further split into validation and test sets)
        x_train, x_temp, y_train, y_temp = train_test_split(
            data["x"],
            data["y"],
            test_size=(test_size + validation_size)
        )

        # Split the temporary set into validation and test sets
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=test_size / (test_size + validation_size)
        )

        return x_train, x_val, x_test, y_train, y_val, y_test
