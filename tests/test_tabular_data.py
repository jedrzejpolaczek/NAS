import pytest
import pandas as pd
from src.data.tabular_data import TabularDataLoader


@pytest.fixture
def tabular_loader():
    config = {"test_size": 0.2, "validation_size": 0.2}
    loader = TabularDataLoader(config)
    return loader


def test_load_data(tabular_loader):
    tabular_loader.load_data("iris")
    data = tabular_loader.get_data()
    assert "x" in data
    assert "y" in data
    assert isinstance(data["x"], pd.DataFrame)
    assert isinstance(data["y"], pd.Series)
    # Iris dataset has 150 samples and 4 features
    assert data["x"].shape == (150, 4)
    assert data["y"].shape == (150, )


def test_preprocess_data(tabular_loader):
    tabular_loader.load_data("iris")
    tabular_loader.preprocess_data()  # Process the data
    data = tabular_loader.get_data()  # Access the modified data

    # Check the mean and standard deviation for each feature
    means = data["x"].mean()
    std_devs = data["x"].std()

    # Ensure the means are close to 0 with a tolerance
    assert all(abs(means) < 1e-6), f"Means are not close to 0: {means}"

    # Ensure the standard deviations are close to 1 with a tolerance
    assert all(abs(std_devs - 1) < 1e-2),\
        f"Standard deviations are not close to 1: {std_devs}"


def test_split_data(tabular_loader):
    tabular_loader.load_data("iris")
    tabular_loader.preprocess_data()  # No need to capture return value
    X_train, X_val, X_test, y_train, y_val, y_test = \
        tabular_loader.split_data(tabular_loader.get_data())

    assert X_train.shape[0] > 0
    assert X_val.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_val) == X_val.shape[0]
    assert len(y_test) == X_test.shape[0]

    assert (X_train.shape[0] + X_val.shape[0] + X_test.shape[0]) == \
        tabular_loader.get_data()["x"].shape[0]
    assert (len(y_train) + len(y_val) + len(y_test)) == \
        len(tabular_loader.get_data()["y"])


def test_load_data_invalid_dataset(tabular_loader):
    with pytest.raises(
        ValueError,
        match="Dataset non_existent_dataset is not supported"
    ):
        tabular_loader.load_data("non_existent_dataset")
