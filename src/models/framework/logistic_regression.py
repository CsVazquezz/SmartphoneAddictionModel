"""Framework version (sklearn)."""


class LogisticRegression:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError
