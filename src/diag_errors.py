"""Diagnose every frozen_test error: is it a RETRIEVAL failure (neighbors are
wrong class -> feature-space problem) or a VLM-judgment failure (neighbors are
right but VLM still wrong)?"""
import sys
from argparse import Namespace
from pathlib import Path
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data import list_train_samples, list_test_samples, CLASS_NAMES
from train_lora import LoRAViTClassifier
from vlm_rag_qualitative import extract_lora_feats, cosine_topk

base = Path(__file__).parent.parent
ckpt_path = base / "artifacts" / "best_v02_f0.8351_mahalanobis" / "round_3" / "ckpt.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = ck["args"]
model = LoRAViTClassifier(
    backbone_key=cfg["backbone"], image_size=cfg["image_size"],
    lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg["lora_dropout"], head_dropout=cfg["head_dropout"],
    lora_mlp=cfg.get("lora_mlp", False), use_dora=cfg.get("dora", False),
    lora_init=cfg.get("lora_init", "default"),
).to(device)
model.load_state_dict(ck["state_dict"], strict=False)
model.eval()

sup_paths, sup_lab = list_train_samples(base / "dev_few_shot")
test_paths = list_test_samples(base / "frozen_test")
args = Namespace(batch_size=32, num_workers=0)
sup_feats, sup_lab = extract_lora_feats(model, sup_paths, cfg["image_size"], args, device, labels=sup_lab)
test_feats, test_names = extract_lora_feats(model, test_paths, cfg["image_size"], args, device, labels=None)

idx, sim = cosine_topk(test_feats, sup_feats, 4)

gt = {r.filename: r.label for r in pd.read_csv(base / "frozen_test" / "_groundtruth.csv").itertuples()}
pred = {r.filename: r.label for r in pd.read_csv(base / "sub_frozen_fixed.csv").itertuples()}

print("=" * 70)
print("ALL ERRORS (true != VLM prediction):")
print("=" * 70)
n_retr_fail = n_vlm_fail = 0
for i, name in enumerate(test_names):
    t, v = gt[name], pred[name]
    if t == v:
        continue
    nbr = [CLASS_NAMES[sup_lab[j]] for j in idx[i]]
    nbr_majority = max(set(nbr), key=nbr.count)
    # retrieval "correct" if majority neighbor == true class
    if nbr_majority == t:
        kind = "VLM-FAIL (neighbors OK, VLM wrong)"
        n_vlm_fail += 1
    else:
        kind = "RETRIEVAL-FAIL (neighbors wrong class)"
        n_retr_fail += 1
    print(f"\n{name}")
    print(f"   true={t}  vlm_pred={v}  neighbors={nbr}")
    print(f"   -> {kind}")

print("\n" + "=" * 70)
print(f"RETRIEVAL failures (feature-space problem): {n_retr_fail}")
print(f"VLM-judgment failures (neighbors were OK):  {n_vlm_fail}")
print("=" * 70)
