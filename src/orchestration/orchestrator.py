from src.optimization.optimizer_factory import OptimizerFactory
from src.data.data_factory import DataFactory
from src.models.model_factory import ModelFactory

from src.evaluation.metrics import Metrics
from src.evaluation.result_aggregator import ResultAggregator
from src.experiment_management.experiment_tracker import ExperimentTracker


class Orchestrator:
    def __init__(self, config: dict) -> None:
        """
        Initializes the Orchestrator with configuration and various components.

        Args:
            config (dict):
                Configuration dictionary containing parameters for
                data loading, preprocessing, model training, and optimization.
        """
        self.config = config
        self.data_factory = DataFactory()
        self.model_factory = ModelFactory()
        self.optimizer_factory = OptimizerFactory()
        self.result_aggregator = ResultAggregator()
        self.experiment_tracker = ExperimentTracker(config)

    def run_experiment(
        self,
        experiment_config: dict
    ) -> None:
        """
        Runs an experiment with the specified configuration.

        This method orchestrates the entire experiment workflow, including
        data loading, preprocessing, model training, optimization, and 
        result aggregation.

        Args:
            experiment_config (dict):
                Configuration dictionary for the experiment.
        Raises:
            ValueError:
                If the optimizer name is not recognized.
        """
        # Select the optimizer using factory
        optimizer_name = experiment_config["optimizer"]["name"]
        optimizer_config = experiment_config["optimizer"]["config"]

        optimizer = self.optimizer_factory.create_optimizer(
            optimizer_name=optimizer_name,
            optimizer_config=optimizer_config
        )

        # Select the data loader usinf factory
        dataset_name = experiment_config["dataset"]["name"]
        dataset_config = experiment_config["dataset"]["config"]
    
        data_loader = self.data_factory.create_data_loader(
            dataset_name=dataset_name,
            dataset_config=dataset_config
        )

        # Select the model using factory
        model_name = experiment_config["model"]["name"]
        model_config = experiment_config["model"]["config"]

        model = self.model_factory.create_model(
            model_name=model_name,
            model_config=model_config
        )

        # Load and pre-process data
        data = data_loader.load_data(dataset_name)
        data_loader.preprocess_data()

        # Split data (can be done within data loader or here)
        x_train, x_val, x_test, y_train, y_val, y_test = data_loader.split_data(data)

        # Optimize the model
        best_params, best_score = optimizer.optimize(model, x_train, y_train)
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

    def run(self) -> None:
        """
        Runs all experiments defined in the configuration.

        The configuration should contain a list of
        experiments to run, where each experiment
        specifies a dataset, model, and optimizer.
        """
        for experiment in self.config["experiments"]:
            self.run_experiment(
                experiment
            )
