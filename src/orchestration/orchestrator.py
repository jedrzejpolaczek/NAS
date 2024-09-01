from data.data_loader import DataLoader
from data.preprocess import Preprocessor
from models.model_factory import ModelFactory
from optimization.grid_search import GridSearchOptimizer
from optimization.random_search import RandomSearchOptimizer
from evaluation.metrics import Metrics
from evaluation.result_aggregator import ResultAggregator
from experiment_management.experiment_tracker import ExperimentTracker


class Orchestrator:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.preprocessor = Preprocessor(config)
        self.model_factory = ModelFactory()
        self.result_aggregator = ResultAggregator()
        self.experiment_tracker = ExperimentTracker(config)

    def run_experiment(self, dataset_name, model_name, optimizer_name):
        data = self.data_loader.load_data(dataset_name)
        data = self.preprocessor.preprocess(data)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(data)

        model = self.model_factory.create_model(model_name, {})
        
        if optimizer_name == 'grid_search':
            optimizer = GridSearchOptimizer(self.config)
        elif optimizer_name == 'random_search':
            optimizer = RandomSearchOptimizer(self.config)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        best_params, best_score = optimizer.optimize(model, X_train, y_train)
        model.set_params(**best_params)
        model.fit(X_train, y_train)

        metrics = Metrics.evaluate(model, X_test, y_test)
        self.result_aggregator.add_result(model_name, optimizer_name, best_params, best_score)

        experiment_data = {
            'dataset': dataset_name,
            'model': model_name,
            'optimizer': optimizer_name,
            'best_params': best_params,
            'best_score': best_score,
            'metrics': metrics
        }
        self.experiment_tracker.log_experiment(experiment_data)

    def run(self):
        for experiment in self.config['experiments']:
            self.run_experiment(experiment['dataset'], experiment['model'], experiment['optimizer'])
