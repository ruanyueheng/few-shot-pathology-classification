"""Build an INDEPENDENT holdout split from dev_few_shot for honest validation.

frozen_test has already been used for model selection (we picked Qwen2.5 / K=4
by looking at its score), so a 1.0 on it only proves the code works, not that
the method generalizes. This script carves a fresh holdout the model has never
been tuned against:

  holdout_test/   <- N images per class (flat PNGs + _groundtruth.csv)
  holdout_pool/   <- the remaining images, as Class_X/ subfolders (retrieval pool)

Then run vlm_rag_predict.py on it. If it stays near 1.0 -> the method is robust.
If it drops to ~0.9x -> you've found the true level (and frozen_test's 1.0 was
partly luck / selection bias).

Usage:
    cd src
    python make_holdout.py --src ../dev_few_shot --n_per_class 10 --seed 42
    # then point vlm_rag_predict.py at ../holdout_pool and ../holdout_test
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
    ap.add_argument("--src", default="../dev_few_shot",
                    help="source labeled dir with Class_X/ subfolders")
    ap.add_argument("--n_per_class", type=int, default=10,
                    help="images per class to hold out as the test set")
    ap.add_argument("--pool_out", default="../holdout_pool")
    ap.add_argument("--test_out", default="../holdout_test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.src)
    pool_out = Path(args.pool_out)
    test_out = Path(args.test_out)

    # fresh dirs
    for d in (pool_out, test_out):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    gt_rows = []
    n_test = n_pool = 0
    for cls in CLASS_NAMES:
        pngs = sorted((src / cls).glob("*.png"))
        rng.shuffle(pngs)
        test_imgs = pngs[:args.n_per_class]
        pool_imgs = pngs[args.n_per_class:]

        # test -> flat dir + groundtruth
        for p in test_imgs:
            shutil.copy2(p, test_out / p.name)
            gt_rows.append({"filename": p.name, "label": cls})
            n_test += 1

        # pool -> Class_X/ subfolders
        (pool_out / cls).mkdir(parents=True, exist_ok=True)
        for p in pool_imgs:
            shutil.copy2(p, pool_out / cls / p.name)
            n_pool += 1

    pd.DataFrame(gt_rows).to_csv(test_out / "_groundtruth.csv", index=False)
    print(f"[ok] holdout_test: {n_test} imgs  ({args.n_per_class}/class) -> {test_out}")
    print(f"[ok] holdout_pool: {n_pool} imgs (retrieval pool)        -> {pool_out}")
    print(f"[ok] groundtruth : {test_out / '_groundtruth.csv'}")
    print(f"\nseed={args.seed} — change --seed to test a different random split.")


if __name__ == "__main__":
    main()
