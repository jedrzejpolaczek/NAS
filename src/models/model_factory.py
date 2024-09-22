from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class ModelFactory:
    def __init__(self):
        self.models = {
            "random_forest_classifier": RandomForestClassifier,
            "support_cector_classifier": SVC,
        }

    def create_model(self, model_name: str, model_config: dict):
        """
        Creates a model instance based on the name and configuration.

        Args:
            model_name (str):
                The name of the model to create.
            model_config (dict):
                The configuration dictionary for the model.

        Returns:
            BaseEstimator:
                An instance of the specified model class.

        Raises:
            ValueError:
                If the specified model is not supported.
        """

        model_class = self.models.get(model_name.lower())
        if model_class is None:
            raise ValueError(f"Unknown model: {model_name}")

        return model_class(**model_config)
