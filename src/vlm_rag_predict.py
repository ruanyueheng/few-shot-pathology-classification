"""Production VLM+RAG inference for the FINAL (unlabeled) test set.

Unlike vlm_rag_qualitative.py (which needs ground truth to pick error/control
cases), this script classifies EVERY test image and writes a submission CSV.
No ground-truth required — use it on the teacher's test set.

Pipeline (per test image):
  1. Encode with LoRA-tuned DINOv2 -> 384-d feature
  2. Retrieve top-K nearest support images (cosine) from the labeled pool
  3. Build a RAG prompt (K labeled neighbors + their classes)
  4. Ask the VLM -> parse "PREDICTION: Class_X"

Single model only (no TTA, no ensembling).

Usage:
    cd src
    python vlm_rag_predict.py ^
        --vlm Qwen/Qwen2.5-VL-7B-Instruct ^
        --train_dir ../train_few_shot ^          # full 250-image retrieval pool
        --test_dir  ../<TEACHER_TEST_DIR> ^       # unlabeled test images
        --ckpt ../artifacts/best_v02_f0.8351_mahalanobis/round_3/ckpt.pt ^
        --out ../submission_final.csv ^
        --topk 4

    # Optional: if a _groundtruth.csv exists, pass --gt_csv to also print metrics.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from data import (
    list_train_samples, list_test_samples,
    IDX_TO_CLASS, CLASS_TO_IDX, CLASS_NAMES,
)
from train_lora import LoRAViTClassifier, make_eval_tf
# reuse the exact same building blocks as the validated qualitative pipeline
from vlm_rag_qualitative import (
    extract_lora_feats, cosine_topk, build_prompt, call_qwen_vl,
)


def collect_test_images(test_dir: str) -> list[Path]:
    """Find test PNGs whether the dir is flat or has subfolders."""
    root = Path(test_dir)
    flat = sorted(root.glob("*.png"))
    if flat:
        return flat
    return sorted(root.rglob("*.png"))


def parse_prediction(vlm_out: str) -> str:
    """Extract Class_X from VLM output (look before REASONING section)."""
    head = vlm_out.split("REASONING")[0]
    for c in CLASS_NAMES:
        if c in head:
            return c
    # fallback: search whole string
    for c in CLASS_NAMES:
        if c in vlm_out:
            return c
    return CLASS_NAMES[0]  # last-resort default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--train_dir", default="../train_few_shot",
                    help="labeled retrieval pool (default: full 250-image set)")
    ap.add_argument("--test_dir", required=True,
                    help="unlabeled test images (flat dir of PNGs)")
    ap.add_argument("--ckpt", required=True,
                    help="LoRA checkpoint for the DINOv2 retrieval backbone")
    ap.add_argument("--out", default="../submission_final.csv")
    ap.add_argument("--class_desc_json", default=None,
                    help="optional pairwise-hint JSON from gen_class_desc.py")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--gt_csv", default=None,
                    help="optional: if given, print metrics (for sanity checks only)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}, vlm={args.vlm}")

    # optional pairwise hints (per-class hand-written CLASS_DESC kept as-is)
    extra_desc = None
    if args.class_desc_json:
        import json
        desc_data = json.loads(Path(args.class_desc_json).read_text(encoding="utf-8"))
        extra_desc = {k: v for k, v in desc_data.items() if "_vs_" in k}
        print(f"[info] loaded pairwise hints: {list(extra_desc.keys())}")

    # ---- load LoRA retrieval backbone ----
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

    # ---- features ----
    sup_paths, sup_labels = list_train_samples(args.train_dir)
    test_paths = collect_test_images(args.test_dir)
    if not test_paths:
        raise SystemExit(f"No .png images found under {args.test_dir}")
    print(f"[info] retrieval pool={len(sup_paths)}, test images={len(test_paths)}")

    sup_feats, sup_lab = extract_lora_feats(model, sup_paths, cfg["image_size"],
                                            args, device, labels=sup_labels)
    test_feats, test_names = extract_lora_feats(model, test_paths, cfg["image_size"],
                                                args, device, labels=None)

    topk_idx, topk_sim = cosine_topk(test_feats, sup_feats, args.topk)

    # ---- load VLM ----
    print(f"[info] loading {args.vlm}…")
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(args.vlm)
    vlm = AutoModelForImageTextToText.from_pretrained(
        args.vlm, torch_dtype=torch.bfloat16, device_map="auto",
    )
    vlm.eval()

    # ---- classify every test image ----
    preds = []
    for i, qname in enumerate(test_names):
        qpath = Path(args.test_dir) / qname
        if not qpath.exists():
            # test_names may be bare names; reconstruct from test_paths order
            qpath = test_paths[i]
        nb_local = topk_idx[i]
        nb_sims = topk_sim[i]
        nb_paths = [sup_paths[j] for j in nb_local]
        nb_classes = [CLASS_NAMES[sup_lab[j]] for j in nb_local]

        prompt = build_prompt(qname, list(zip(nb_paths, nb_classes, nb_sims)),
                              extra_desc=extra_desc)
        query_img = Image.open(qpath).convert("RGB").resize((224, 224))
        nb_imgs = [Image.open(p).convert("RGB").resize((224, 224)) for p in nb_paths]

        try:
            vlm_out = call_qwen_vl(processor, vlm, query_img, nb_imgs, prompt, device)
        except Exception as e:
            print(f"[warn] VLM failed on {qname}: {e}")
            vlm_out = ""
        pred = parse_prediction(vlm_out)
        preds.append(pred)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{len(test_names)}] {qname} -> {pred}")

    # ---- write submission ----
    out_df = pd.DataFrame({"filename": test_names, "label": preds})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\n[ok] wrote {args.out} ({len(out_df)} rows)")
    print(out_df["label"].value_counts().to_string())

    # ---- optional sanity metrics ----
    if args.gt_csv and Path(args.gt_csv).exists():
        from sklearn.metrics import f1_score, balanced_accuracy_score
        gt = pd.read_csv(args.gt_csv)
        merged = out_df.merge(gt, on="filename", suffixes=("_pred", "_true"))
        if len(merged):
            f1 = f1_score(merged["label_true"], merged["label_pred"], average="macro")
            ba = balanced_accuracy_score(merged["label_true"], merged["label_pred"])
            print(f"\n[sanity] macro-F1={f1:.4f}  balanced-acc={ba:.4f}  (n={len(merged)})")


if __name__ == "__main__":
    main()
