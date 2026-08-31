"""Logistic regression from scratch -- no framework, same code as the notebook.

    hypothesis:   h(x) = 1 / (1 + e^-(params' * x))
    cost:         J = -(1/m) * sum[ y*log(h) + (1-y)*log(1-h) ]
    update:       params := params - alfa * (1/m) * sum[ (h - y) * x ]
"""
import numpy as np


def h(params, samples):
    """Evaluates h(x) = 1 / (1 + e^-(params' * x)) for every sample at once."""
    # @ is Python's matrix multiplication operator: it already does the dot 
    # product, so it replacesa a standard double loop over samples and params.
    acum = samples @ params        # a + b*x1 + c*x2 + ... for every row
    acum = acum * (-1)
    acum = np.clip(acum, -500, 500)   # keeps e^acum from overflowing
    return 1 / (1 + np.exp(acum))


def show_errors(params, samples, y):
    """Mean error of the current params. Only for plotting, not used to update."""
    hyp = np.clip(h(params, samples), 1e-10, 1 - 1e-10)   # avoid the log(0) error
    error = np.where(y == 1, -np.log(hyp), -np.log(1 - hyp))
    return error.sum() / len(samples)


def GD(params, samples, y, alfa):
    """One run of gradient descent over the whole sample set."""
    error = h(params, samples) - y
    # again @ does the dot product, summing (h - y) * x over every sample
    acum = samples.T @ error                  # sumatory part of the formula
    return params - alfa * (1 / len(samples)) * acum


def add_bias(X):
    """x0 = 1 on every row, so params[0] is the intercept."""
    return np.hstack([np.ones((len(X), 1)), X])


class LogisticRegression:
    def __init__(self, alfa=0.5, max_epochs=2000):
        self.alfa = alfa
        self.max_epochs = max_epochs
        self.errors = []

    def fit(self, X, y):
        samples = add_bias(X)

        params = np.zeros(samples.shape[1])
        params[0] = np.log(y.mean() / (1 - y.mean()))   # base-rate log-odds

        self.errors = []
        for epoch in range(1, self.max_epochs + 1):
            oldparams = params.copy()
            params = GD(params, samples, y, self.alfa)
            error = show_errors(params, samples, y)
            self.errors.append(error)

            if epoch % 100 == 0:
                print(f"epoch {epoch:4d}   error {error:.5f}")

            # local minimum: params stopped moving
            if np.allclose(oldparams, params, atol=1e-6):
                break

        print(f"stopped at epoch {epoch}, final error {error:.5f}")
        self.params = params
        return self

    def predict_proba(self, X):
        return h(self.params, add_bias(X))
