from sklearn.model_selection import GridSearchCV


class GridSearchOptimizer:
    def __init__(self, config):
        self.config = config

    def optimize(self, model, X_train, y_train):
        grid_search = GridSearchCV(model, self.config["config"]["param_grid"], cv=self.config["config"]["cv"])
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_, grid_search.best_score_
