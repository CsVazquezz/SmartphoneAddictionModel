# SmartphoneAddictionModel

Logistic regression for the Kaggle competition
[Playground Series S6E8](https://www.kaggle.com/competitions/playground-series-s6e8):
predicting smartphone addiction from usage habits. Binary classification,
scored on ROC AUC.

Two implementations, one written directly in NumPy and one on scikit-learn,
so they can be compared. Both are evaluated on a held-out validation set and
on the competition's test set.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

`train.csv` and `test.csv` are already in `SmartPhoneAddiction/data/`.

## Run

```bash
./.venv/bin/python run.py                      # scratch model  -> outputs/submission_scratch.csv
./.venv/bin/python run.py --model framework    # sklearn model  -> outputs/submission_framework.csv
./.venv/bin/python run.py --lam 10000          # scratch model with L2 regularization
```

The scratch model takes about a minute; the framework one a few seconds.
Each prints train and validation AUC and writes a Kaggle-ready
`id,addicted_label` file of predicted probabilities.

## Layout

```
run.py                                        entry point
src/data.py                                   preprocessing and train/val split
src/metrics.py                                ROC AUC
src/models/scratch/logistic_regression.py     NumPy implementation, optional L2
src/models/framework/logistic_regression.py   scikit-learn implementation
notebooks/logistic_regression_pipeline.ipynb  EDA, the reasoning, and every figure in the report
report.pdf                                    write-up
```

## Pipeline

`train.csv` is split by position: the first 80% for training (553,095 rows),
the last 20% for validation (138,274). `test.csv` has no labels, so it is the
test set and is scored by submitting to Kaggle.

The split happens before any preprocessing is fitted. Median imputation
(skewness is under 0.56 on every column, so median costs nothing and protects
against outliers) → z-score → clip at ±3σ → PCA keeping PC1–PC7 (92.55% of
variance) → one-hot encode the three categorical columns. Twelve features in
total, all learned from the training rows only.

**Scratch:** full-batch gradient descent with `alfa = 0.5`, stopping when the
parameters stop moving. An optional L2 penalty (`--lam`) leaves the intercept
unpenalized.

**Framework:** `SGDClassifier(loss="log_loss")`, the same cost function by
gradient descent, on degree-2 polynomial features (12 → 90 columns) with
`alpha = 0.001`.

## Results

| Model | Train AUC | Val AUC | Test AUC (Kaggle) |
|---|---|---|---|
| Linear, scratch | 0.91096 | 0.91106 | 0.91342 |
| Linear, framework | 0.91063 | 0.91073 | 0.91312 |
| Poly2 + L2, framework | 0.92229 | 0.92243 | **0.92484** |

The linear model has **low variance** (train and validation cost sit on top of
each other at 0.347) and **moderate bias** (both stop improving well above
zero), so it is **mildly underfitting**. Regularization alone buys +0.0004;
adding capacity with polynomial features buys +0.011, with the train/val gap
still at zero.

See [`report.pdf`](report.pdf) for the full write-up: exploratory analysis,
the skewness test behind the imputation choice, PCA, the bias/variance
diagnosis, confusion matrices, ROC curves and the Kaggle scores.
