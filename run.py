"""Entry point.  python run.py --model scratch|framework"""
import argparse
from pathlib import Path

import pandas as pd

from src import data
from src.metrics import roc_auc
from src.models.scratch import logistic_regression as scratch_lr
from src.models.framework import logistic_regression as framework_lr

MODELS = {
    "scratch": scratch_lr.LogisticRegression,
    "framework": framework_lr.LogisticRegression,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default="scratch")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or Path(__file__).resolve().parent / f"outputs/submission_{args.model}.csv"

    X_train, y_train, X_test, test_ids, columns = data.load()
    print("X_train", X_train.shape, "  X_test", X_test.shape)

    model = MODELS[args.model]()
    model.fit(X_train, y_train)

    print("\ntrain AUC:", round(roc_auc(y_train, model.predict_proba(X_train)), 5))
    print(pd.Series(model.params, index=["intercept"] + columns).round(4))

    data.write_submission(test_ids, model.predict_proba(X_test), out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
