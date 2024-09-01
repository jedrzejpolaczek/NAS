from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, config):
        self.config = config

    def preprocess(self, data):
        # Placeholder for data preprocessing steps
        # Example: handling missing data, normalization, etc.
        data = data.dropna()
        return data

    def split_data(self, data):
        return train_test_split(data, test_size=self.config['test_size'])
