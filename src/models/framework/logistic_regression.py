"""Framework version (scikit-learn), to compare against the scratch one."""
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import PolynomialFeatures


class LogisticRegression:
    def __init__(self, alpha=0.001, max_epochs=1000, degree=2):   # alpha = scratch lam / m
        # degree 2 adds every x^2 and x*y column so the boundary can curve; 1 = plain linear
        self.poly = PolynomialFeatures(degree, include_bias=False)
        self.model = SGDClassifier(loss="log_loss", alpha=alpha,
                                   max_iter=max_epochs, random_state=0)

    def fit(self, X, y):
        self.model.fit(self.poly.fit_transform(X), y)
        self.params = np.concatenate([self.model.intercept_, self.model.coef_[0]])
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.poly.transform(X))[:, 1]
