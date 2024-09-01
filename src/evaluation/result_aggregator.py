class ResultAggregator:
    def __init__(self):
        self.results = []

    def add_result(self, model_name, optimizer_name, best_params, score):
        self.results.append({
            'model': model_name,
            'optimizer': optimizer_name,
            'best_params': best_params,
            'score': score
        })

    def aggregate_results(self):
        # Placeholder for aggregating and analyzing results
        return self.results
