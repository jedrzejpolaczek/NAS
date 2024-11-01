from sklearn.model_selection import RandomizedSearchCV


class RandomSearchOptimizer:
    def __init__(self, config):
        self.config = config

    def optimize(self, model, X_train, y_train):
        random_search = RandomizedSearchCV(model, self.config['param_distributions'], cv=self.config['cv'], n_iter=self.config['n_iter'])
        random_search.fit(X_train, y_train)
        return random_search.best_params_, random_search.best_score_
