from src.data.template_data_loader import DataLoader
from src.data.tabular_data import TabularDataLoader

from src.models.model_factory import ModelFactory
from src.optimization.search_strategies.grid_search import GridSearchOptimizer
from src.optimization.search_strategies.random_search import RandomSearchOptimizer
from src.evaluation.metrics import Metrics
from src.evaluation.result_aggregator import ResultAggregator
from src.experiment_management.experiment_tracker import ExperimentTracker


class Orchestrator:
    def __init__(self, config):
        """
        Initializes the Orchestrator with configuration and various components.

        Args:
            config (dict):
                Configuration dictionary containing parameters for
                data loading, preprocessing, model training, and optimization.
        """
        self.config = config
        self.data_loader = DataLoader(config)
        self.model_factory = ModelFactory()
        self.optimizer = None  # TODO
        self.result_aggregator = ResultAggregator()
        self.experiment_tracker = ExperimentTracker(config)

    def _select_loader(self, experiment_config: dict, dataset_name: str):
        if dataset_name == "tabular_data":
            self.data_loader = TabularDataLoader(experiment_config)

    def _select_optimizer(self, experiment_config: dict, optimizer_name: str):
        if optimizer_name == "grid_search":
            self.optimizer = GridSearchOptimizer(experiment_config)
        elif optimizer_name == "random_search":
            self.optimizer = RandomSearchOptimizer(experiment_config)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def run_experiment(
        self,
        experiment_config,
        dataset_name,
        model_name,
        optimizer_name
    ):
        """
        Runs a single experiment including data loading,
        preprocessing, model training, optimization, and evaluation.

        Args:
            experiment_config (dict):

            dataset_name (str):
                The name of the dataset to load.
            model_name (str):
                The name of the model to train.
            optimizer_name (str):
                The name of the optimization strategy to use.

        Raises:
            ValueError:
                If the optimizer name is not recognized.
        """
        # Load and preprocess data
        self._select_loader(experiment_config, dataset_name)
        self.data_loader.load_data(dataset_name)
        self.data_loader.preprocess_data()
        preprocessed_data = self.data_loader.get_data()
        x_train, x_val, x_test, y_train, y_val, y_test = \
            self.data_loader.split_data(preprocessed_data)

        # Create and train the model
        model = self.model_factory.create_model(model_name, {})

        # Select the optimizer
        self._select_optimizer(experiment_config, dataset_name)

        # Optimize the model
        best_params, best_score = self.optimizer.optimize(model, x_train, y_train)
        model.set_params(**best_params)
        model.fit(x_train, y_train)

        # Evaluate the model
        metrics = Metrics.evaluate(model, x_test, y_test)
        self.result_aggregator.add_result(
            model_name,
            optimizer_name,
            best_params,
            best_score
        )

        # Log the experiment
        experiment_data = {
            "dataset": dataset_name,
            "model": model_name,
            "optimizer": optimizer_name,
            "best_params": best_params,
            "best_score": best_score,
            "metrics": metrics
        }
        self.experiment_tracker.log_experiment(experiment_data)

    def run(self):
        """
        Runs all experiments defined in the configuration.

        The configuration should contain a list of
        experiments to run, where each experiment
        specifies a dataset, model, and optimizer.
        """
        for experiment in self.config["experiments"]:
            self.run_experiment(
                experiment,
                experiment["dataset"],
                experiment["model"],
                experiment["optimizer"]
            )
