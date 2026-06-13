"""Split train_few_shot/ into dev/ (40/class) + frozen_test/ (10/class).

The frozen_test is a *balanced* internal test set, ONLY used to:
  - validate the pseudo-labeling pipeline end-to-end (treat as unlabeled)
  - sanity-check final model performance before submitting

It is NOT representative of the teacher's real (large, imbalanced) test set.
Do NOT tune hyperparameters against frozen_test. All HP selection must be
done with k-fold CV inside dev/.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import numpy as np

from data import CLASS_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../train_few_shot")
    ap.add_argument("--dev_out", default="../dev_few_shot")
    ap.add_argument("--test_out", default="../frozen_test")
    ap.add_argument("--test_per_class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    src = Path(args.src)
    dev_out = Path(args.dev_out)
    test_out = Path(args.test_out)

    if dev_out.exists() or test_out.exists():
        raise SystemExit(
            f"Refusing to overwrite. Remove {dev_out} and {test_out} first."
        )

    test_manifest = []
    for cls in CLASS_NAMES:
        cls_src = src / cls
        files = sorted(cls_src.glob("*.png"))
        idx = np.arange(len(files))
        rng.shuffle(idx)
        test_idx = set(idx[:args.test_per_class].tolist())

        (dev_out / cls).mkdir(parents=True, exist_ok=True)
        (test_out).mkdir(parents=True, exist_ok=True)

        for i, f in enumerate(files):
            if i in test_idx:
                # rename to test_XXXXX.png style (so we can pretend they're "test")
                new_name = f"test_{cls}_{f.stem}.png"
                shutil.copy2(f, test_out / new_name)
                test_manifest.append((new_name, cls))
            else:
                shutil.copy2(f, dev_out / cls / f.name)

        print(f"  {cls}: {len(files)} -> dev {len(files)-len(test_idx)} "
              f"+ test {len(test_idx)}")

    # write ground-truth labels for frozen test (only used for our internal eval)
    gt_path = test_out / "_groundtruth.csv"
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write("filename,label\n")
        for name, lbl in test_manifest:
            f.write(f"{name},{lbl}\n")
    print(f"\n[ok] dev   -> {dev_out}  ({sum(1 for _ in dev_out.rglob('*.png'))} imgs)")
    print(f"[ok] test  -> {test_out}  ({len(test_manifest)} imgs + groundtruth)")
    print(f"[note] {gt_path.name} is the TRUE labels for internal eval only — "
          "do NOT let the model see them.")


if __name__ == "__main__":
    main()
