"""Quick evaluation: compare submission CSV vs groundtruth CSV.

Usage:
    python eval.py --pred ../sub_phikon.csv --gt ../frozen_test/_groundtruth.csv
"""
import argparse
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--gt",   required=True)
args = ap.parse_args()

pred = pd.read_csv(args.pred)
gt   = pd.read_csv(args.gt)
df   = pred.merge(gt, on="filename", suffixes=("_pred", "_true"))

f1 = f1_score(df["label_true"], df["label_pred"], average="macro")
ba = balanced_accuracy_score(df["label_true"], df["label_pred"])
print(f"macro-F1      = {f1:.4f}")
print(f"balanced acc  = {ba:.4f}")
print(f"n_samples     = {len(df)}")
print("\nConfusion matrix (rows=true, cols=pred):")
classes = sorted(df["label_true"].unique())
print("       " + "  ".join(f"{c[-1]:>5}" for c in classes))
cm = confusion_matrix(df["label_true"], df["label_pred"], labels=classes)
for cls, row in zip(classes, cm):
    print(f"{cls}  " + "  ".join(f"{v:>5}" for v in row))
