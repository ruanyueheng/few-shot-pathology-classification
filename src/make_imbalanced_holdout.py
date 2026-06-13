"""Build an IMBALANCED holdout to simulate the teacher's class-imbalanced test set.

The retrieval pool stays BALANCED (mirrors your balanced labeled support), but
the test set is deliberately imbalanced (e.g. majority class 18 imgs, minority
class 3 imgs). This is the one thing internal balanced CV cannot probe: whether
macro-F1 / balanced-acc hold up when minority classes have few samples.

  imb_pool/   <- pool_per_class images per class  (balanced retrieval pool)
  imb_test/   <- test_counts[c] images per class  (imbalanced) + _groundtruth.csv

Usage:
    cd src
    python make_imbalanced_holdout.py --src ../dev_few_shot \
        --pool_per_class 20 --test_counts 18,12,8,4,3 --seed 42
"""
from __future__ import annotations
import argparse
import shutil
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import CLASS_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../dev_few_shot")
    ap.add_argument("--pool_per_class", type=int, default=20,
                    help="balanced retrieval pool size per class")
    ap.add_argument("--test_counts", default="18,12,8,4,3",
                    help="comma-separated test images per class (imbalanced)")
    ap.add_argument("--pool_out", default="../imb_pool")
    ap.add_argument("--test_out", default="../imb_test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    test_counts = [int(x) for x in args.test_counts.split(",")]
    if len(test_counts) != len(CLASS_NAMES):
        raise SystemExit(f"--test_counts needs {len(CLASS_NAMES)} values, got {len(test_counts)}")

    rng = random.Random(args.seed)
    src = Path(args.src)
    pool_out, test_out = Path(args.pool_out), Path(args.test_out)
    for d in (pool_out, test_out):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    gt_rows = []
    summary = []
    for ci, cls in enumerate(CLASS_NAMES):
        pngs = sorted((src / cls).glob("*.png"))
        rng.shuffle(pngs)
        pool = pngs[:args.pool_per_class]
        remaining = pngs[args.pool_per_class:]
        n_test = test_counts[ci]
        if n_test > len(remaining):
            raise SystemExit(f"{cls}: want {n_test} test but only {len(remaining)} left "
                             f"after pool_per_class={args.pool_per_class}")
        test = remaining[:n_test]

        (pool_out / cls).mkdir(parents=True, exist_ok=True)
        for p in pool:
            shutil.copy2(p, pool_out / cls / p.name)
        for p in test:
            shutil.copy2(p, test_out / p.name)
            gt_rows.append({"filename": p.name, "label": cls})
        summary.append((cls, len(pool), n_test))

    pd.DataFrame(gt_rows).to_csv(test_out / "_groundtruth.csv", index=False)

    print(f"[ok] imbalanced holdout (seed={args.seed}):")
    print(f"  {'class':<9} {'pool':>5} {'test':>5}")
    for cls, np_, nt in summary:
        print(f"  {cls:<9} {np_:>5} {nt:>5}")
    total_test = sum(test_counts)
    ratio = max(test_counts) / max(1, min(test_counts))
    print(f"  total test = {total_test},  imbalance ratio = {ratio:.1f}:1")
    print(f"\n  pool -> {pool_out}")
    print(f"  test -> {test_out}  (gt: _groundtruth.csv)")


if __name__ == "__main__":
    main()
