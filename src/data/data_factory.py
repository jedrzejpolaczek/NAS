from src.data.tabular_data import TabularDataLoader


class DataFactory:
    """
    Factory class for creating data loaders based on dataset type.

    This class encapsulates the logic for selecting and instantiating the
    appropriate data loader for a given dataset type.
    """

    def __init__(self):
        self.data_loaders = {
            "tabular_data": TabularDataLoader,
        }

    def create_data_loader(self, dataset_name, dataset_config):
        """
        Creates a data loader based on the dataset type.

        Args:
            dataset_name (str):
                The name of the dataset to load.
            dataset_config (dict):
                Configuration dictionary containing parameters
                for data loading, preprocessing, and splitting.

        Returns:
            DataLoader:
                An instance of the appropriate data loader class.
        """

        data_loader_class = self.data_loaders.get(dataset_name.lower())
        if data_loader_class is None:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

        return data_loader_class(dataset_config)

