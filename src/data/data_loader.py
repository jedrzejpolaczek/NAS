import pandas as pd


class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_data(self, dataset_name):
        if dataset_name == 'example_dataset':
            return pd.read_csv(self.config['data_path'])
        # Add other dataset loading mechanisms here
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
