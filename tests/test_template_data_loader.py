import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from src.data.templates.template_data_loader import DataLoader


class TestDataLoader(DataLoader):
    def load_data(self, dataset_name):
        # Mock data loading
        if dataset_name == "iris":
            iris = load_iris()
            return {"x": iris.data, "y": iris.target}
        else:
            raise ValueError("Dataset not found")

    def preprocess_data(self):
        # Mock preprocessing: Apply standard scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.data["x"])
        return {"x": X_scaled, "y": self.data["y"]}

# Example test data
@pytest.fixture
def test_loader():
    config = {"test_size": 0.2, "validation_size": 0.2}
    loader = TestDataLoader(config)
    return loader


def test_load_data(test_loader):
    data = test_loader.load_data("iris")
    assert "x" in data
    assert "y" in data
    assert data["x"].shape == (150, 4)  # 150 samples, 4 features
    assert data["y"].shape == (150, )


def test_preprocess_data(test_loader):
    test_loader.data = test_loader.load_data("iris")
    preprocessed_data = test_loader.preprocess_data()
    assert "x" in preprocessed_data
    assert "y" in preprocessed_data
    assert preprocessed_data["x"].shape == (150, 4)  # Same shape as input data
    assert preprocessed_data["y"].shape == (150, )


def test_split_data(test_loader):
    test_loader.data = test_loader.load_data("iris")
    preprocessed_data = test_loader.preprocess_data()
    X_train, X_val, X_test, y_train, y_val, y_test = test_loader.split_data(preprocessed_data)

    assert X_train.shape[0] > 0
    assert X_val.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_val) == X_val.shape[0]
    assert len(y_test) == X_test.shape[0]

    assert (X_train.shape[0] + X_val.shape[0] + X_test.shape[0]) == preprocessed_data["x"].shape[0]
    assert (len(y_train) + len(y_val) + len(y_test)) == len(preprocessed_data["y"])
