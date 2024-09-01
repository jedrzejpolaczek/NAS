from sklearn.metrics import accuracy_score, roc_auc_score


class Metrics:
    @staticmethod
    def evaluate(model, X_test, y_test):
        y_pred = model.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred)
        }
