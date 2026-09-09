"""Framework version (scikit-learn), to compare against the scratch one."""
import numpy as np
from sklearn.linear_model import SGDClassifier


class LogisticRegression:
    def __init__(self, alpha=0.0001, max_epochs=1000):   # alpha = scratch lam / m
        self.model = SGDClassifier(loss="log_loss", alpha=alpha,
                                   max_iter=max_epochs, random_state=0)

    def fit(self, X, y):
        self.model.fit(X, y)
        # sklearn stores the intercept apart from the weights; join them
        self.params = np.concatenate([self.model.intercept_, self.model.coef_[0]])
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
