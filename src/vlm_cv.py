"""K-fold cross-validation of the VLM+RAG method, to estimate internal
generalization (mean ± std macro-F1 / balanced-acc).

CAVEAT: source data is balanced and in-distribution; the LoRA checkpoint was
trained on dev_few_shot so it has "seen" most of these images. This CV measures
INTERNAL stability, NOT performance on the teacher's large imbalanced set.

Usage:
    cd src
    python vlm_cv.py --src ../train_few_shot --folds 5 \
        --ckpt ../artifacts/best_v02_f0.8351_mahalanobis/round_3/ckpt.pt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score

from data import list_train_samples, CLASS_NAMES
from train_lora import LoRAViTClassifier
from vlm_rag_qualitative import extract_lora_feats, cosine_topk, build_prompt, call_qwen_vl
from vlm_rag_predict import parse_prediction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../train_few_shot")
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_json", default="../cv_vlm_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}, vlm={args.vlm}, folds={args.folds}")

    # ---- LoRA retrieval backbone ----
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    model = LoRAViTClassifier(
        backbone_key=cfg["backbone"], image_size=cfg["image_size"],
        lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"], head_dropout=cfg["head_dropout"],
        lora_mlp=cfg.get("lora_mlp", False),
        use_dora=cfg.get("dora", False),
        lora_init=cfg.get("lora_init", "default"),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    paths, labels = list_train_samples(args.src)
    paths = list(paths)
    print(f"[info] {len(paths)} images from {args.src}")
    all_feats, _ = extract_lora_feats(model, paths, cfg["image_size"],
                                      args, device, labels=labels)

    # ---- VLM (load once, reuse across folds) ----
    print(f"[info] loading {args.vlm} …")
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(args.vlm)
    vlm = AutoModelForImageTextToText.from_pretrained(
        args.vlm, torch_dtype=torch.bfloat16, device_map="auto",
    )
    vlm.eval()

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []
    for fold, (pool_idx, test_idx) in enumerate(skf.split(paths, labels)):
        pool_feats = all_feats[pool_idx]
        topk_idx, topk_sim = cosine_topk(all_feats[test_idx], pool_feats, args.topk)

        preds, y_true = [], []
        for i, ti in enumerate(test_idx):
            qpath = paths[ti]
            nb_global = [pool_idx[j] for j in topk_idx[i]]
            nb_paths = [paths[g] for g in nb_global]
            nb_classes = [CLASS_NAMES[labels[g]] for g in nb_global]
            sims = topk_sim[i]

            prompt = build_prompt(qpath.name, list(zip(nb_paths, nb_classes, sims)))
            query_img = Image.open(qpath).convert("RGB").resize((224, 224))
            nb_imgs = [Image.open(p).convert("RGB").resize((224, 224)) for p in nb_paths]
            try:
                vlm_out = call_qwen_vl(processor, vlm, query_img, nb_imgs, prompt, device)
            except Exception as e:
                print(f"[warn] {qpath.name}: {e}")
                vlm_out = ""
            preds.append(parse_prediction(vlm_out))
            y_true.append(CLASS_NAMES[labels[ti]])
            if (i + 1) % 10 == 0:
                print(f"  fold{fold+1} {i+1}/{len(test_idx)}")

        f1 = f1_score(y_true, preds, average="macro")
        ba = balanced_accuracy_score(y_true, preds)
        print(f"[fold {fold+1}/{args.folds}] macro-F1={f1:.4f}  bal-acc={ba:.4f}  (n={len(test_idx)})")
        fold_results.append({"fold": fold + 1, "macro_f1": float(f1),
                             "bal_acc": float(ba), "n": int(len(test_idx))})

    f1s = [r["macro_f1"] for r in fold_results]
    bas = [r["bal_acc"] for r in fold_results]
    summary = {
        "folds": fold_results,
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "bal_acc_mean": float(np.mean(bas)), "bal_acc_std": float(np.std(bas)),
        "src": args.src, "vlm": args.vlm, "topk": args.topk, "seed": args.seed,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n===== {args.folds}-fold CV =====")
    print(f"macro-F1     = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"balanced-acc = {np.mean(bas):.4f} +/- {np.std(bas):.4f}")
    print(f"[ok] saved {args.out_json}")


if __name__ == "__main__":
    main()
