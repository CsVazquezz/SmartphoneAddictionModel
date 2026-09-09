"""Entry point.  python run.py --model scratch|framework"""
import argparse
from pathlib import Path

import pandas as pd

from src import data
from src.metrics import roc_auc
from src.models.scratch import logistic_regression as scratch_lr
from src.models.framework import logistic_regression as framework_lr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["scratch", "framework"], default="scratch")
    p.add_argument("--lam", type=float, default=0.0, help="L2 strength (scratch only)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or Path(__file__).resolve().parent / f"outputs/submission_{args.model}.csv"

    d = data.load()
    X_train, y_train = d["train"]
    X_val, y_val = d["val"]
    print("train", X_train.shape, " val", X_val.shape, " test", d["test"].shape)

    if args.model == "scratch":
        model = scratch_lr.LogisticRegression(lam=args.lam)
    else:
        model = framework_lr.LogisticRegression()
    model.fit(X_train, y_train)

    print("\nAUC   train %.5f   val %.5f   (test = Kaggle leaderboard)"
          % (roc_auc(y_train, model.predict_proba(X_train)),
             roc_auc(y_val, model.predict_proba(X_val))))
    print(pd.Series(model.params, index=["intercept"] + d["columns"]).round(4))

    data.write_submission(d["test_ids"], model.predict_proba(d["test"]), out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
