# ML-2 期末项目 · 少样本病理图像分类(H&E 5 类)

> 每类 50 张、共 250 张 32×32 H&E 染色图;测试集 61,881 张、类别不均衡。
> 评测指标:macro-F1(主) / balanced accuracy。

## 方法概述

**DINOv2 ViT-S/14 + LoRA 微调(r=16) + 自训练 + LaplacianShot 直推式推理。**

- **Backbone**:DINOv2 自监督 ViT(冻结),仅用 LoRA 微调注意力层(~64 万参数)
- **训练**:80 epoch + EMA + 监督对比损失(SupCon) + HED 染色增广
- **推理**:7 种直推式方法对比,最终选 LaplacianShot(无平衡假设 + 利用测试集近邻结构)
- **核心发现**:训练均衡而测试不均衡时,「强制类别均衡」的方法(TIM 边际熵 / PT-MAP Sinkhorn / alpha-TIM)**有害**,应选无平衡假设的方法
- **真实泛化**:约 0.6~0.7(5 折交叉验证),并诚实排查了一次数据泄漏

不做 TTA / 模型集成(题目禁止)。完整技术报告见 `report/机器学习2课程报告_v5.docx`。

## 环境

```powershell
conda env create -f environment.yml
conda activate ml2_fewshot
```
需 CUDA 12.8 + PyTorch cu128(Blackwell GPU)。验证:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 复现步骤

```powershell
cd src

# (可选) 从 train_few_shot 切出内部验证集 dev/frozen
python make_splits.py

# 1) 训练 LoRA(全部 250 张,带进度条)
python self_train.py --backbone vits14 \
  --train_dir ../train_few_shot --test_dir ../test_shuffled \
  --out_dir ../artifacts/selftrain_final \
  --lora_r 16 --lora_alpha 16 --epochs 80 --rounds 0 --ema \
  --hed_aug --supcon_weight 0.1 --final_method head

# 2) 生成最终提交(LaplacianShot,4 方法对比 → 选最优)
python submit_best.py
# 或单方法: python infer_lora.py --method laplacianshot ...
```
提交文件:`24124035.csv`(61,881 行,格式 `filename,label`)。

## 目录结构

```
final_project/
├── README.md  environment.yml  .gitignore
├── train_few_shot/            # 250 张训练图(5类×50)
├── 24124035.csv               # 最终提交
├── src/
│   ├── 核心:  data.py  train_lora.py  self_train.py  transductive.py
│   ├── 推理:  infer_lora.py  submit_best.py  final_submit.py  infer_phikon.py
│   ├── 训练辅助: stain_aug.py(HED增广)  features.py(backbone)
│   ├── 调参:  tune_nobalance.py  tune_v2.py  tim_search.py
│   │          lora_rank_search.py  reg_search.py
│   ├── 自训练: self_train_61k.py
│   ├── 消融/诊断: exp_learning_curve.py  exp_maha.py  diag_errors.py
│   └── 数据构造: make_splits.py  make_holdout.py  make_imbalanced_holdout.py
└── report/
    ├── 机器学习2课程报告_v5.docx   # 最终报告
    ├── make_report_v5.js          # 报告生成脚本(Node + docx)
    └── figures/                   # 报告插图
```

## 关键脚本说明

| 脚本 | 作用 |
|------|------|
| `train_lora.py` | LoRA 模型定义 + 训练(EMA / SupCon / HED) |
| `transductive.py` | 7 种直推式方法(LaplacianShot / TIM / PT-MAP / Mahalanobis / SimpleShot …) |
| `submit_best.py` | 最终提交:4 种方法在 61k 上对比并选最优 |
| `tune_nobalance.py` / `tune_v2.py` | 非平衡方法调参 + 训练配置搜索 |
| `exp_learning_curve.py` | 学习曲线 — 证明数据量是主要瓶颈 |
| `diag_errors.py` | 误差诊断 — 定位 Class_2↔4 混淆 |

## 主要结论

- 最终方案真实泛化约 **0.6~0.7**(5 折 CV),受限于「每类仅 50 张」的数据量(学习曲线证明未饱和)
- 误差集中在 **Class_2 ↔ Class_4**(弥散梯度 vs 平滑梯度+散在核,32×32 下视觉接近)
- 不平衡测试集上,**无平衡假设**的方法(LaplacianShot / TIM λ=0 / PT-MAP nosink)显著优于强制均衡的方法
