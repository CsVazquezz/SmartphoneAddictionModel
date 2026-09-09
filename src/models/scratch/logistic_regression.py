"""Logistic regression from scratch -- no framework, same code as the notebook.

    hypothesis:   h(x) = 1 / (1 + e^-(params' * x))
    cost:         J = -(1/m) * sum[ y*log(h) + (1-y)*log(1-h) ]
    update:       params := params - alfa * (1/m) * sum[ (h - y) * x ]

With L2 (lam > 0) the cost gains (lam / 2m) * sum(w^2) and the update lam * w,
for every parameter except the intercept params[0].
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


def show_errors(params, samples, y, lam=0.0):
    """Mean error of the current params. Only for plotting, not used to update."""
    hyp = np.clip(h(params, samples), 1e-10, 1 - 1e-10)   # avoid the log(0) error
    error = np.where(y == 1, -np.log(hyp), -np.log(1 - hyp))
    mean_error = error.sum() / len(samples)

    # L2: add (lam / 2m) * sum(w^2), leaving out the intercept params[0]
    weights = params[1:]
    penalty = lam * (weights ** 2).sum() / (2 * len(samples))

    return mean_error + penalty


def GD(params, samples, y, alfa, lam=0.0):
    """One run of gradient descent over the whole sample set."""
    error = h(params, samples) - y
    # again @ does the dot product, summing (h - y) * x over every sample
    acum = samples.T @ error                  # sumatory part of the formula

    # L2: the derivative of (lam / 2m) * w^2 is (lam / m) * w
    penalty = lam * params
    penalty[0] = 0                            # the intercept is not penalised

    return params - alfa * (1 / len(samples)) * (acum + penalty)


def add_bias(X):
    """x0 = 1 on every row, so params[0] is the intercept."""
    return np.hstack([np.ones((len(X), 1)), X])


class LogisticRegression:
    def __init__(self, alfa=0.5, max_epochs=2000, lam=0.0):
        self.alfa = alfa
        self.max_epochs = max_epochs
        self.lam = lam
        self.errors = []

    def fit(self, X, y):
        samples = add_bias(X)

        params = np.zeros(samples.shape[1])
        params[0] = np.log(y.mean() / (1 - y.mean()))   # base-rate log-odds

        # record the cost before the first step, so the curve starts at epoch 0
        self.errors = [show_errors(params, samples, y, self.lam)]
        for epoch in range(1, self.max_epochs + 1):
            oldparams = params.copy()
            params = GD(params, samples, y, self.alfa, self.lam)
            error = show_errors(params, samples, y, self.lam)
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
