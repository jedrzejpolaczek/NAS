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

    # Option 1
    data = loader.load_data("dataset_name")
    loader.preprocess_data()
    x_train, x_val, x_test, y_train, y_val, y_test = loader.split_data(data)

    # Option 2
    x_train, x_val, x_test, y_train, y_val, y_test = loader.get_data_sets()

"""
import os
import logging
from sklearn.model_selection import train_test_split


class BaseDataLoader:
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
        download_data():
            Abstract method for downloading dataset. 
            Must be implemented in the subclass.
        load_data():
            Abstract method for loading dataset. 
            Must be implemented in the subclass.
        preprocess_data():
            Abstract method for preprocessing the loaded data. 
            Must be implemented in the subclass.
        data_pipeline():
            Executes the complete data pipeline which includes 
            downloading, loading and preprocessing the data.
        get_data():
            Retrieves the currently loaded and preprocessed data.
        split_data(data):
            Splits the preprocessed data into training, validation, 
            and test sets based on the configuration provided.
        get_data_sets():
            Executes the data pipeline and
            returns the resulting split datasets.
    """
    def __init__(
        self,
        config: dict,
        logger: logging.Logger
    ) -> None:
        """
        Initializes the DataLoader with a configuration dictionary.

        Args:
            config (dict):
                Configuration dictionary containing parameters for
                data loading, preprocessing, and splitting.
            logger (logging.Logger):
                An instance of the logger object shared in whole project.
        """
        self.config = config

        # Error handling for paths
        required_keys = ["path", "file_name"]
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required configuration key: '{key}'")

        self.file_path = os.path.join(
            self.config["path"],
            self.config["file_name"]
        )
        self.data = None
        self.logger = logger

    def download_data(self) -> None:
        """
        Downloads specified dataset and saves it.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def load_data(self) -> None:
        """
        Loads data based on the dataset name.

        This method should be overridden by subclasses to handle the 
        specific logic required to load different types of datasets.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def preprocess_data(self) -> None:
        """
        Preprocesses the loaded data.

        This method should be overridden by subclasses to handle the 
        specific preprocessing steps required for different types of datasets.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def data_pipeline(self) -> None:
        """
        Executes the data pipeline which includes downloading,
        loading and preprocessing the data.

        This method calls the methods `download_data`,
        `load_data` and `preprocess_data` 
        in that order, which manage the complete data pipeline.

        Returns:
            None
        """
        self.logger.info("Starting data pipeline...")
        self.download_data()
        self.load_data()
        self.preprocess_data()
        self.logger.info("Data pipeline completed.")

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

    def split_data(
        self,
        data: dict
    ) -> tuple:
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
        self.logger.info("Starting splitting data...")
        test_size = self.config.get("test_size", 0.2)
        validation_size = self.config.get("validation_size", 0.2)

        # Split into training and a temporary set
        # (which will be further split into validation and test sets)
        x_train, x_temp, y_train, y_temp = train_test_split(
            data["X"],
            data["y"],
            test_size=(test_size + validation_size),
            random_state=self.config.get("random_state", 42)
        )

        # Split the temporary set into validation and test sets
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=test_size / (test_size + validation_size),
            random_state=self.config.get("random_state", 42)
        )
        self.logger.info("Splitting data completed.")

        return x_train, x_val, x_test, y_train, y_val, y_test

    def get_data_sets(self) -> tuple:
        """
        Executes the data pipeline and
        returns the resulting split datasets.

        This method first calls the `data_pipeline`
        method to manage downloading,
        loading, and preprocessing the data.
        Then, it splits the preprocessed data into
        training, validation, and test sets by calling `split_data`.

        Returns:
            tuple:
                Six arrays representing the training, validation, and test
                sets for features and labels respectively:
                (x_train, x_val, x_test, y_train, y_val, y_test)
        """
        self.data_pipeline()
        return self.split_data(self.data)
