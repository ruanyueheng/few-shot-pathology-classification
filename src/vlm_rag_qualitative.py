"""LLM+RAG qualitative experiment (Plan B).

For each MISCLASSIFIED test image in frozen_test (under v02 best Mahalanobis):
  1. Retrieve top-K nearest support images by cosine similarity in LoRA feature space
  2. Build a multi-image prompt: "Here are K labeled examples. Now classify this query."
  3. Query a local VLM (Qwen2-VL-2B-Instruct) for prediction + reasoning
  4. Write a markdown report (with image refs + VLM rationales) for inclusion in
     the final docx as a qualitative error-case analysis.

This is *qualitative* — output is human-readable narrative, not a new metric.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import (
    list_train_samples, list_test_samples, ImagePathDataset,
    IDX_TO_CLASS, CLASS_TO_IDX, CLASS_NAMES, build_transform,
)
from train_lora import LoRAViTClassifier, make_eval_tf
from transductive import mahalanobis


# -------- feature extraction (reuse our pipeline) --------

@torch.no_grad()
def extract_lora_feats(model, paths, image_size, args, device, labels=None):
    eval_tf = make_eval_tf(image_size)
    ds = ImagePathDataset(paths, labels, eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    model.eval()
    feats, second = [], []
    for x, y_or_n in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
            f = model.backbone(x)
        feats.append(f.float().cpu().numpy())
        if labels is not None:
            second.append(y_or_n.numpy())
        else:
            second.extend(list(y_or_n))
    feats = np.concatenate(feats, axis=0)
    if labels is not None:
        second = np.concatenate(second, axis=0)
    return feats, second


def cosine_topk(query_feats: np.ndarray, support_feats: np.ndarray, k: int):
    """Return [Nq, k] indices into support, sorted by descending cosine sim."""
    qn = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-12)
    sn = support_feats / (np.linalg.norm(support_feats, axis=1, keepdims=True) + 1e-12)
    sim = qn @ sn.T
    idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    # sort within top-k
    row = np.arange(idx.shape[0])[:, None]
    sims = sim[row, idx]
    order = np.argsort(-sims, axis=1)
    idx = idx[row, order]
    sims = sims[row, order]
    return idx, sims


# -------- VLM prompting --------

CLASS_DESC = {
    # Optional handcrafted visual descriptors (user can edit). We KEEP them generic
    # because we don't know the true biological meaning of each class.
    "Class_0": "(visual notes) typically pink/lavender stroma with few dark nuclei",
    "Class_1": "(visual notes) typically dense clusters of dark purple nuclei",
    "Class_2": "(visual notes) typically diffuse purple gradient with blurred structure",
    "Class_3": "(visual notes) typically sparse, light-colored texture",
    "Class_4": "(visual notes) typically mixed: smooth gradient + scattered nuclei",
}


def build_prompt(query_filename: str, neighbors: list[tuple[str, str, float]],
                 extra_desc: dict | None = None) -> str:
    """neighbors: list of (image_path, true_class, cosine_sim).
    extra_desc: optional dict with pairwise discriminative hints, e.g.
        {"Class_2_vs_Class_3": "...", "Class_0_vs_Class_3": "..."}
    """
    lines = [
        "You are an expert at fine-grained image classification on H&E-stained "
        "histopathology thumbnails (32x32, RGB, bicubically upsampled to 224x224).",
        "There are 5 classes. Below are visual descriptors and labeled support examples.",
        "",
    ]
    for c in CLASS_NAMES:
        lines.append(f"- {c}: {CLASS_DESC[c]}")

    # inject pairwise discriminative hints if available
    if extra_desc:
        pair_hints = {k: v for k, v in extra_desc.items() if "_vs_" in k}
        if pair_hints:
            lines += ["", "Key distinctions between easily confused classes:"]
            for k, v in pair_hints.items():
                ca, cb = k.split("_vs_")
                lines.append(f"  {ca} vs {cb}: {v}")

    lines += [
        "",
        f"I will show you {len(neighbors)} labeled support images "
        "(their classes are given), then a query image. Based on visual similarity "
        "and the descriptors above, predict the query's class and explain in 2 sentences.",
        "",
    ]
    for i, (_, cls, sim) in enumerate(neighbors):
        lines.append(f"Support {i+1}: class = {cls}  (cosine sim to query = {sim:.3f})")
    lines += [
        "",
        "The LAST image shown is the QUERY image to classify. "
        "Its filename and true label are deliberately withheld — judge ONLY from "
        "the image's visual content and the labeled support examples above.",
        "",
        "Output strictly in this format:",
        "  PREDICTION: Class_X",
        "  REASONING: <short explanation>",
    ]
    return "\n".join(lines)


def call_qwen_vl(processor, model, query_img: Image.Image,
                 neighbor_imgs: list[Image.Image], prompt_text: str,
                 device: str = "cuda") -> str:
    """Send query image + neighbors + prompt to Qwen2-VL, return decoded string."""
    # Build messages with multiple images
    content = []
    for i, img in enumerate(neighbor_imgs):
        content.append({"type": "image", "image": img})
    content.append({"type": "image", "image": query_img})
    content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=neighbor_imgs + [query_img],
        return_tensors="pt", padding=True,
    ).to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                             temperature=1.0)
    gen = out[:, inputs.input_ids.shape[1]:]
    text_out = processor.batch_decode(gen, skip_special_tokens=True)[0]
    return text_out.strip()


# -------- main --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="../dev_few_shot")
    ap.add_argument("--test_dir", default="../frozen_test")
    ap.add_argument("--gt_csv", default="../frozen_test/_groundtruth.csv")
    ap.add_argument("--ckpt", default="../artifacts/best_v02_f0.8351_mahalanobis/round_3/ckpt.pt")
    ap.add_argument("--out_md", default="../report/vlm_rag_qualitative.md")
    ap.add_argument("--vlm", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--tukey_beta", type=float, default=0.5)
    ap.add_argument("--maha_shrink", type=float, default=0.3)
    ap.add_argument("--also_correct", type=int, default=2,
                    help="also analyze N correctly-classified cases for contrast")
    ap.add_argument("--class_desc_json", default=None,
                    help="path to JSON file with data-driven class descriptions "
                         "(generated by gen_class_desc.py); overrides hardcoded CLASS_DESC")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}, ckpt={args.ckpt}, vlm={args.vlm}")

    # ---- load data-driven class descriptions (optional) ----
    import json as _json
    extra_desc: dict | None = None
    if args.class_desc_json:
        desc_data = _json.loads(Path(args.class_desc_json).read_text(encoding="utf-8"))
        # only keep pairwise hints — do NOT override per-class CLASS_DESC
        # (hand-written per-class descriptions are more discriminative)
        extra_desc = {k: v for k, v in desc_data.items() if "_vs_" in k}
        print(f"[info] loaded pairwise hints from {args.class_desc_json}: {list(extra_desc.keys())}")

    # ---- load LoRA backbone ----
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

    sup_paths, sup_labels = list_train_samples(args.train_dir)
    test_paths = list_test_samples(args.test_dir)
    gt_df = pd.read_csv(args.gt_csv)
    gt_map = {row.filename: CLASS_TO_IDX[row.label] for row in gt_df.itertuples()}

    print(f"[info] support {len(sup_paths)}, test {len(test_paths)}")
    sup_feats, sup_lab = extract_lora_feats(model, sup_paths, cfg["image_size"],
                                            args, device, labels=sup_labels)
    test_feats, test_names = extract_lora_feats(model, test_paths, cfg["image_size"],
                                                 args, device, labels=None)

    # ---- Mahalanobis predictions ----
    preds = mahalanobis(sup_feats, sup_lab, test_feats, num_classes=5,
                        tukey_beta=args.tukey_beta, shrink=args.maha_shrink)
    y_true = np.array([gt_map[n] for n in test_names])
    wrong_mask = preds != y_true
    print(f"[info] Mahalanobis errors: {wrong_mask.sum()}/{len(preds)}")

    # ---- pick cases: all wrong + a few correct for contrast ----
    wrong_idx = np.where(wrong_mask)[0].tolist()
    correct_idx = np.where(~wrong_mask)[0].tolist()
    rng = np.random.default_rng(0)
    rng.shuffle(correct_idx)
    correct_sample = correct_idx[:args.also_correct]
    case_idx = wrong_idx + correct_sample

    # ---- find neighbors for each case ----
    topk_idx, topk_sim = cosine_topk(test_feats[case_idx], sup_feats, args.topk)

    # ---- load VLM ----
    print(f"[info] loading {args.vlm}…")
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(args.vlm)
    vlm = AutoModelForImageTextToText.from_pretrained(
        args.vlm, torch_dtype=torch.bfloat16, device_map="auto",
    )
    vlm.eval()

    # ---- run RAG per case ----
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VLM + RAG Qualitative Analysis",
        "",
        "*This experiment uses Qwen2-VL-2B-Instruct as an external visual reasoner "
        "on each misclassified frozen_test case, with top-K nearest-neighbor "
        "support images retrieved via cosine similarity in the LoRA-tuned DINOv2 "
        "feature space (RAG). Output is qualitative — predictions are NOT used to "
        "update submission.csv.*",
        "",
        f"- Backbone: LoRA-tuned DINOv2 ViT-S/14 (v02 best ckpt)",
        f"- Retriever: top-{args.topk} cosine NN in 384-d feature space",
        f"- VLM: {args.vlm}",
        f"- Cases: {wrong_mask.sum()} errors + {args.also_correct} controls",
        "",
        "---",
        "",
    ]

    n_vlm_right = 0
    for i, ci in enumerate(case_idx):
        qname = test_names[ci]
        qpath = Path(args.test_dir) / qname
        true_cls = CLASS_NAMES[y_true[ci]]
        maha_pred = CLASS_NAMES[preds[ci]]
        nb_local = topk_idx[i]
        nb_sims = topk_sim[i]
        nb_paths = [sup_paths[j] for j in nb_local]
        nb_classes = [CLASS_NAMES[sup_lab[j]] for j in nb_local]

        # build prompt
        prompt = build_prompt(qname, list(zip(nb_paths, nb_classes, nb_sims)),
                              extra_desc=extra_desc)
        # load images for VLM
        query_img = Image.open(qpath).convert("RGB").resize((224, 224))
        nb_imgs = [Image.open(p).convert("RGB").resize((224, 224)) for p in nb_paths]

        print(f"\n[case {i+1}/{len(case_idx)}] {qname}  true={true_cls}  maha={maha_pred}")
        try:
            vlm_out = call_qwen_vl(processor, vlm, query_img, nb_imgs, prompt, device)
        except Exception as e:
            vlm_out = f"[VLM error] {e}"
        print(vlm_out[:200])

        # parse VLM prediction (regex-lite)
        vlm_pred = "?"
        for c in CLASS_NAMES:
            if c in vlm_out.split("REASONING")[0]:
                vlm_pred = c
                break
        if vlm_pred == true_cls:
            n_vlm_right += 1

        # write markdown
        tag = "❌ ERROR" if ci in wrong_idx else "✅ CONTROL (correct by maha)"
        lines += [
            f"## Case {i+1}: `{qname}` — {tag}",
            "",
            f"- **True label:** `{true_cls}`",
            f"- **Mahalanobis prediction:** `{maha_pred}`",
            f"- **VLM prediction:** `{vlm_pred}`",
            f"- **Top-{args.topk} support neighbors (cosine sim):**",
        ]
        for p, c, s in zip(nb_paths, nb_classes, nb_sims):
            lines.append(f"  - `{Path(p).name}` ({c}, sim={s:.3f})")
        lines += [
            "",
            "**VLM raw output:**",
            "",
            "```",
            vlm_out,
            "```",
            "",
            "---",
            "",
        ]

    lines.insert(11, f"- VLM agreed with true label on {n_vlm_right}/{len(case_idx)} cases  "
                     f"(maha agreed on {(~wrong_mask).sum()}/{len(preds)} for reference)")

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[ok] wrote {args.out_md}")


if __name__ == "__main__":
    main()
