# SmartphoneAddictionModel

Logistic regression for the Kaggle competition
[Playground Series S6E8](https://www.kaggle.com/competitions/playground-series-s6e8):
predicting smartphone addiction from usage habits. Binary classification,
scored on ROC AUC.

Two implementations are planned — one written directly in NumPy, one on top of
a framework — so they can be compared. Only the first is done.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

`train.csv` and `test.csv` are already in `SmartPhoneAddiction/data/`.

## Run

```bash
./.venv/bin/python run.py                  # -> outputs/submission_scratch.csv
./.venv/bin/python run.py --model scratch  # same thing, explicit
```

Takes about 40 seconds on 691,369 rows. Writes a Kaggle-ready
`id,addicted_label` file of predicted probabilities.

## Layout

```
run.py                                        entry point
src/data.py                                   preprocessing
src/metrics.py                                ROC AUC
src/models/scratch/logistic_regression.py     the model
src/models/framework/                         not implemented yet
notebooks/logistic_regression_pipeline.ipynb  EDA and the reasoning behind it
report.pdf                                    write-up
```

## Pipeline

Median imputation (skewness is under 0.56 on every column, so median costs
nothing and protects against outliers) → z-score → clip at ±3σ → PCA keeping
PC1–PC7 (92.55% of variance) → one-hot encode the three categorical columns.
Twelve features in total.

The model is full-batch gradient descent with `alfa = 0.5`, stopping when the
parameters stop moving.

## Result

Converges at epoch 1,678. Cost 0.60267 → 0.34658, **train AUC 0.911**.

See [`report.pdf`](report.pdf) for the full write-up: exploratory analysis,
the skewness test behind the imputation choice, PCA, and the results.
