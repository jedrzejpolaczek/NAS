import json
import os


class ExperimentTracker:
    def __init__(self, config):
        self.config = config
        if not os.path.exists(self.config['log_dir']):
            os.makedirs(self.config['log_dir'])

    def log_experiment(self, experiment_data):
        with open(os.path.join(self.config['log_dir'], 'experiments.json'), 'a') as f:
            f.write(json.dumps(experiment_data) + '\n')
