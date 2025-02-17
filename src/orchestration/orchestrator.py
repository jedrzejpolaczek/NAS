"""
Orchestrator for managing experiment workflows.

This module provides the `Orchestrator` class, which is responsible for
orchestrating the entire experiment workflow. It utilizes various components
like `ComponentFactory`, `ResultAggregator`, and `ExperimentTracker`
to manage data loading, preprocessing, model training,
optimization, and result aggregation.
"""
import os

from src.utils.logger import get_logger
from src.utils.component_factory import ComponentFactory
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
        experiment_tracker (ExperimentTracker):
            Tracker for logging experiment data.
    """
    def __init__(
        self,
        config: dict,
        logger
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
        self.experiment_tracker = ExperimentTracker(config["log_dir"])
        self.logger = logger

    def get_component(
        self,
        component_type: str,
        component_config: dict
    ) -> object:
        """
        Get a component from the component factory.

        Args:
            component_type (str):
                The type of the component to get.
            component_config (dict):
                The configuration of the component.

        Returns:
            object:
                The component object created by the component factory.
        """
        return self.component_factory.create_component(
            component_type=component_type,
            component_name=component_config["name"],
            component_config=component_config,
            logger=self.logger
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
        optimizer = self.get_component(
            component_type="optimizer",
            component_config=experiment_config["optimizer"]
        )
        data_loader = self.get_component(
            component_type="data_loader",
            component_config=experiment_config["dataset"]
        )
        model = self.get_component(
            component_type="model",
            component_config=experiment_config["model"]
        )

        # Extract experiment details (avoid redundant variable assignments)
        optimizer_name = experiment_config["optimizer"]["name"]
        dataset_name = experiment_config["dataset"]["name"] + \
            " (" + experiment_config["dataset"]["dataset_name"] + ")"
        model_name = experiment_config["model"]["name"]

        # Load and pre-process data
        data_loader.data_pipeline()
        data = data_loader.get_data()

        # Split data (can be done within data loader or here)
        (
            input_features_for_train,
            _,  # input_features_for_validation
            input_features_for_test,
            target_labels_for_train,
            _,  # target_labels_for_validation
            target_labels_for_test
        ) = data_loader.split_data(data)

        # Optimize the model
        best_params, best_score = optimizer.optimize(
            model.model,
            input_features_for_train,
            target_labels_for_train
        )
        model.model.set_params(**best_params)
        model.model.fit(
            input_features_for_train,
            target_labels_for_train
        )

        # Evaluate the model
        metrics = model.evaluate(
            input_features_for_test,
            target_labels_for_test
        )

        # Log the experiment
        experiment_data = {
            "optimizer": optimizer_name,
            "dataset": dataset_name,
            "model": model_name,            
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
