from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class ModelFactory:
    def __init__(self):
        pass

    def create_model(self, model_name, hyperparameters):
        if model_name == 'random_forest':
            return RandomForestClassifier(**hyperparameters)
        elif model_name == 'svm':
            return SVC(**hyperparameters)
        # Add other models here
        else:
            raise ValueError(f"Unknown model: {model_name}")
