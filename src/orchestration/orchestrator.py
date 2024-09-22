"""
Orchestrator for managing experiment workflows.

This module provides the `Orchestrator` class, which is responsible for
orchestrating the entire experiment workflow. It utilizes various components
like `ComponentFactory`, `ResultAggregator`, and `ExperimentTracker`
to manage data loading, preprocessing, model training,
optimization, and result aggregation.
"""
from src.utils.component_factory import ComponentFactory
from src.evaluation.metrics import Metrics
from src.evaluation.result_aggregator import ResultAggregator
from src.experiment_management.experiment_tracker import ExperimentTracker


class Orchestrator:
    """
    Orchestrates the entire experiment workflow.

    This class facilitates running machine learning experiments by managing
    different stages and utilizing various components.

    Attributes:
        config (dict):
            Configuration dictionary containing parameters for
            data loading, preprocessing, model training, and optimization.
        component_factory (ComponentFactory):
            Factory for creating experiment components.
        result_aggregator (ResultAggregator):
            Aggregator for storing experiment results.
        experiment_tracker (ExperimentTracker):
            Tracker for logging experiment data.
    """
    def __init__(
        self,
        config: dict
    ) -> None:
        """
        Initializes the Orchestrator with configuration and various components.

        Args:
            config (dict):
                Configuration dictionary containing parameters for
                data loading, preprocessing, model training, and optimization.
        """
        self.config = config
        self.component_factory = ComponentFactory()
        self.result_aggregator = ResultAggregator()
        self.experiment_tracker = ExperimentTracker(config)

    def get_component(
        self,
        component_type: str,
        component_config: dict
    ) -> object:
        return self.component_factory.create_component(
            component_type=component_type,
            component_name=component_config["name"],
            component_config=component_config["config"]
        )
    
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
        # Select the optimizer, data loader and model using factory
        optimizer = self.get_component("optimizer", experiment_config["optimizer"])
        data_loader = self.get_component("data_loader", experiment_config["dataset"])
        model = self.get_component("model", experiment_config["model"])

        # Extract experiment details (avoid redundant variable assignments)
        optimizer_name = experiment_config["optimizer"]["name"]
        dataset_name = experiment_config["dataset"]["name"]
        model_name = experiment_config["model"]["name"]

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
