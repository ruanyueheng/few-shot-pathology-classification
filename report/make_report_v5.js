const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak, ImageRun,
} = require("docx");

const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const cellBorders = { top: border, bottom: border, left: border, right: border };
const FONT = "Microsoft YaHei";

function p(text, opts = {}) {
  return new Paragraph({ alignment: opts.align ?? AlignmentType.LEFT,
    spacing: opts.spacing ?? { after: 100 },
    children: [new TextRun({ text, font: FONT, size: opts.size ?? 22,
      bold: opts.bold ?? false, italics: opts.italics ?? false })] });
}
function h(level, text) {
  const heading = [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][level - 1];
  const size = [36, 28, 24][level - 1];
  return new Paragraph({ heading, spacing: { before: 240, after: 160 },
    children: [new TextRun({ text, font: FONT, size, bold: true })] });
}
function bullet(text) {
  return new Paragraph({ numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60 }, children: [new TextRun({ text, font: FONT, size: 22 })] });
}
function code(text) {
  return new Paragraph({ spacing: { after: 80 },
    shading: { fill: "F2F2F2", type: ShadingType.CLEAR },
    children: [new TextRun({ text, font: "Consolas", size: 18 })] });
}
function tcell(text, opts = {}) {
  return new TableCell({ borders: cellBorders,
    width: { size: opts.width, type: WidthType.DXA },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: opts.align ?? AlignmentType.LEFT,
      children: [new TextRun({ text, font: FONT, size: opts.size ?? 20, bold: opts.bold ?? false })] })] });
}
function table(headers, rows, columnWidths) {
  const totalW = columnWidths.reduce((a, b) => a + b, 0);
  const headerCells = headers.map((t, i) => tcell(t, { width: columnWidths[i], fill: "D5E8F0", bold: true, align: AlignmentType.CENTER }));
  const bodyRows = rows.map(r => new TableRow({ children: r.map((t, i) => tcell(String(t), { width: columnWidths[i], align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) }));
  return new Table({ width: { size: totalW, type: WidthType.DXA }, columnWidths, rows: [new TableRow({ children: headerCells }), ...bodyRows] });
}
const FIGDIR = __dirname + "/figures/";
const SIZES = JSON.parse(fs.readFileSync(FIGDIR + "sizes.json", "utf8"));
function img(name, targetW) {
  const [w, hh] = SIZES[name];
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 140, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(FIGDIR + name),
      transformation: { width: targetW, height: Math.round(targetW * hh / w) } })] });
}
function caption(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 180 },
    children: [new TextRun({ text, font: FONT, size: 18, italics: true, color: "666666" })] });
}

const S = [];

// ===== 封面 =====
S.push(p(""));
S.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1200, after: 240 },
  children: [new TextRun({ text: "机器学习 2 课程期末报告", font: FONT, size: 44, bold: true })] }));
S.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: "少样本病理图像分类（H&E 风格 5 类）", font: FONT, size: 32, bold: true })] }));
S.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 800 },
  children: [new TextRun({ text: "—— LoRA 微调 + 直推式推理 + 不平衡测试集分析", font: FONT, size: 22, italics: true, color: "666666" })] }));
S.push(p("课程名称：机器学习 2", { align: AlignmentType.CENTER, size: 24 }));
S.push(p("学生姓名：__________", { align: AlignmentType.CENTER, size: 24 }));
S.push(p("学生学号：24124035", { align: AlignmentType.CENTER, size: 24 }));
S.push(p("授课教师：张闻华", { align: AlignmentType.CENTER, size: 24 }));
S.push(p("日期：2026 年 6 月", { align: AlignmentType.CENTER, size: 24 }));
S.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 摘要 =====
S.push(h(1, "摘要"));
S.push(p(
  "本项目在每类仅 50 张、共 250 张 32×32 H&E 染色图的少样本条件下构建 5 分类器,并在 61,881 张类别不平衡的测试集上提交预测。最终方案为:DINOv2 ViT-S/14 + LoRA 微调(r=16) + 自训练 + LaplacianShot 直推式推理。",
  { bold: true },
));
S.push(p(
  "本报告有两条核心贡献:(1) 系统地对比了 7 种直推式(transductive)推理方法,并发现一个关键规律——训练集均衡而测试集不均衡时,任何「强制类别均衡」的方法(TIM 的边际熵、PT-MAP 的 Sinkhorn、alpha-TIM)都是有害的,应选用无平衡假设的方法;(2) 通过严格的 5 折交叉验证、学习曲线、以及对一次数据泄漏的诚实排查,我们确认了真实泛化能力约为 0.6~0.7,并指出 50 张验证集的不可靠性。报告完整记录了从基线到最终方案的全部成果与失败尝试。",
));
S.push(h(2, "技术路线与动机"));
S.push(p("为什么选「预训练大模型 + LoRA + 直推式推理」这条路线:"));
S.push(bullet("每类仅 50 张,从零训练 ResNet/ViT 必然过拟合 → 必须「站在大模型肩膀上」,用预训练 backbone(老师鼓励方向)。"));
S.push(bullet("全量微调 86M+ 参数会过拟合 250 张 → 用 LoRA 只调约 64 万低秩参数,参数高效、抗过拟合。"));
S.push(bullet("测试集 61k 无标签但可见 → 用直推式(transductive)方法挖掘测试集的无标签特征分布,免费提升,是 few-shot 文献的标准做法。"));
S.push(bullet("测试集类别不均衡 → 在直推方法中刻意避开「强制均衡」的变体(详见第五章核心发现)。"));
S.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 一、背景与数据探索 =====
S.push(h(1, "一、实验背景与数据探索"));
S.push(h(2, "1.1 任务与数据"));
S.push(p("任务:每类 50 张的少样本条件下构建 5 分类器,在大规模、类别不平衡的测试集上评估。指标为 macro-F1(主)与 balanced accuracy(次);实验发现两者在所有方法上差异 <0.01。"));
S.push(table(
  ["数据集", "数量", "说明"],
  [
    ["train_few_shot", "250 (50×5)", "训练集,完全均衡,32×32 RGB H&E 染色"],
    ["test_shuffled", "61,881", "测试集,老师确认类别不均衡,与训练集无交集"],
    ["frozen_test", "50 (10×5)", "从训练集切出的内部验证集(有 GT)"],
  ],
  [2600, 1800, 4960],
));
S.push(p("关键约束:训练集极小(250张)→ 参数/样本比极高、易过拟合;测试集不均衡 → 评估须重 macro-F1;不允许 TTA、不允许模型集成(题目要求)。", { italics: true }));
S.push(img("fig1_samples.png", 360));
S.push(caption("图 1　各类别样本示例(每行一类,H&E 染色 32×32 上采样显示)"));

S.push(h(2, "1.2 数据预处理与增广(含负面发现)"));
S.push(p("预处理:bicubic 上采样到 224×224 + ImageNet 标准化。训练增广:水平/垂直翻转、±15° 旋转、HED 染色空间抖动(替代通用 ColorJitter)。推理无任何 TTA(题目禁止)。"));
S.push(p("重要负面发现(本数据对增广极敏感):", { bold: true, spacing: { before: 80, after: 40 } }));
S.push(bullet("MixUp / CutMix:OOF macro-F1 下降约 6pt,污染了本就模糊的纹理。"));
S.push(bullet("RandAugment 多视图平均:有害(0.517→0.449)。"));
S.push(bullet("Real-ESRGAN 4× 超分:暴跌(0.835→0.621),自然图像 SR 扭曲了 H&E 染色统计量。"));
S.push(bullet("结论:32×32 上采样后高频纹理信号稀薄,温和增广 + HED 染色抖动是最大可行强度。"));
S.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 二、模型架构与配置搜索 =====
S.push(h(1, "二、模型架构与配置搜索"));
S.push(h(2, "2.1 Backbone:DINOv2 ViT-S/14"));
S.push(bullet("Meta 自监督视觉 Transformer,ImageNet 自监督预训练;自监督特征在小样本场景显著优于有监督预训练。"));
S.push(bullet("ViT-S/14 轻量(~22M),patch=14,输入 224×224 → 16×16 patch,输出 384 维 CLS token。"));
S.push(bullet("对比 9 个 backbone(EVA-02、ViT-B/L、Phikon、PLIP、CONCH、Lunit-DINO、ConvNeXt-V2)后,DINOv2 ViT-S 胜出——病理专用 FM 依赖 224 高分辨率纹理,32×32 上采后领域优势消失;ViT-B/L 大维度在 200 样本上过拟合。"));

S.push(h(2, "2.2 LoRA 微调与配置搜索"));
S.push(p("仅微调注意力层的低秩矩阵(attn.qkv + attn.proj),冻结原始权重。可训练参数约 64 万(参数/样本比 ≈ 2568:1,极高,需强正则)。通过系统搜索确定配置:"));
S.push(table(
  ["搜索维度", "最优值", "依据"],
  [
    ["LoRA rank r", "16", "rank 搜索 {4,8,16,32}:r=4 容量不足,r=32 过拟合"],
    ["LoRA alpha", "16", "Phase2 搜索:alpha=16 优于 32(0.696 vs 0.674)"],
    ["epochs", "80", "递增趋势:30→50→80 = 0.678→0.696→0.714"],
    ["HED 增广", "True", "F1 比关闭高约 3%(0.708 vs 0.679)"],
    ["SupCon 权重", "0.1", "0.1 与 0.3 无显著差异,取 0.1 更安全"],
    ["EMA", "True", "正则化搜索中的最大正收益"],
  ],
  [2400, 1600, 5360],
));
S.push(p("分类头:LayerNorm(384) → Dropout(0.1) → Linear(384,5)。", { size: 20, italics: true }));
S.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 三、训练策略 =====
S.push(h(1, "三、训练策略"));
S.push(h(2, "3.1 基础配置"));
S.push(table(
  ["项", "值"],
  [
    ["优化器 / 学习率", "AdamW,分层(backbone 5e-4 / head 1e-3)"],
    ["调度 / 精度", "Cosine + 10% warmup,bfloat16"],
    ["正则", "label smoothing 0.1,weight decay 0.05,grad clip 1.0"],
  ],
  [3200, 6160],
));
S.push(h(2, "3.2 EMA(指数移动平均)"));
S.push(p("维护参数的指数移动平均(decay=0.95)作为推理权重,并对早期步数做 warmup(eff=min(decay,(1+step)/(10+step)))。因训练仅几百步,固定高 decay 会让 EMA 卡在初始随机权重附近;warmup 后 EMA 稳定了特征质量,对所有直推方法都有提升。"));
S.push(h(2, "3.3 监督对比损失 SupCon(Khosla 2020)"));
S.push(p("在投影空间拉近同类、推开异类:L = CE + 0.1·L_supcon,温度 τ=0.07。投影头 Linear(384,384)→ReLU→Linear(384,128) 仅训练时使用。"));
S.push(h(2, "3.4 HED 染色增广(Tellez 2018)"));
S.push(p("H&E 染色颜色变异是病理 AI 的主要噪声源。HED 在染色空间(而非 RGB)扰动:RGB→光学密度→用 Ruifrok-Johnston 染色矩阵分解出 H/E/D 浓度→对每通道乘性+加性抖动(sigma=0.02,bias=0.01)→反变换。对病理图像比通用 ColorJitter 更有效。"));
S.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 四、直推式推理方法 =====
S.push(h(1, "四、直推式(Transductive)推理方法"));
S.push(p("直推式方法利用测试集的无标签特征信息(但不使用标签)来改善预测。特征统一经 Tukey 幂变换(x'=sign(x)|x|^β,降低偏度)+ 中心化 + L2 归一化预处理。我们实现并对比了 7 种方法:"));
S.push(h(2, "4.1 方法原理一览"));
S.push(table(
  ["方法", "核心思想", "是否假设均衡"],
  [
    ["Linear Head", "训练好的分类头直接 softmax,不用测试集信息", "❌ 无"],
    ["SimpleShot", "类原型 + 最近余弦,可用 combined-mean 中心化", "❌ 无"],
    ["Mahalanobis", "高斯判别 + 共享协方差(Ledoit-Wolf 收缩)", "❌ 无"],
    ["TIM (λ_marg=0)", "信息最大化,只保留条件熵(让预测更自信)", "❌ 无"],
    ["TIM (λ_marg>0)", "额外最大化边际熵 → 强制预测覆盖所有类", "✅ 有"],
    ["PT-MAP nosink", "soft-EM,类均值漂移向测试分布(无 Sinkhorn)", "❌ 无"],
    ["PT-MAP + Sinkhorn", "E-step 后行列归一化 → 强制均衡分配", "✅ 有"],
    ["alpha-TIM", "α-Renyi 散度替代 KL,更激进推向均匀", "✅ 有"],
    ["★ LaplacianShot", "类原型一元代价 + query 近邻图的拉普拉斯平滑", "❌ 无"],
  ],
  [2400, 4600, 2360],
));
S.push(h(2, "4.2 最终方案:LaplacianShot(Ziko 2020)"));
S.push(p("能量函数 E(Y)=Σ Y·a(到类原型距离) + λ·Σ W_ij·||Y_i−Y_j||²,其中 W 是 query 间的 k-NN 高斯亲和力。它在 SimpleShot 的原型基础上加入「相邻 query 倾向同类」的平滑正则,既无平衡假设、又利用了测试集结构。"));
S.push(bullet("最优参数:tukey=1.0, knn=7, lam=10.0, sigma=1.0, n_iter=20"));
S.push(bullet("61k 可扩展性:原始 dense [Nq,Nq] 亲和力需 ~15GB(OOM);我们实现了稀疏版(分批算余弦、只留 k 邻居),内存 O(Nq×k)≈4MB,knn 构建 21.5s、20 轮迭代 0.1s,完全可行。"));
S.push(bullet("不可扩展的方法:Label Propagation 需解 (Nq+Ns)² 线性系统 → 61k OOM,未采用。"));
S.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 五、不平衡测试集分析(核心) =====
S.push(h(1, "五、不平衡测试集分析(核心发现)"));
S.push(p(
  "这是本项目最重要的洞察:训练集完全均衡(20%/类)、5 折交叉验证也是均衡的,但测试集 61k 是不均衡的。因此——",
  { bold: true },
));
S.push(p("任何在均衡 CV 上表现好的「平衡假设」方法,在不均衡测试集上反而是有害的。", { bold: true }));
S.push(h(2, "5.1 平衡假设为何有害"));
S.push(bullet("TIM 的边际熵 H(p̄):最大化它会强制预测趋向均匀。λ_marg=2.0 时把 Class_0 从 45% 压到 26%,大量正确的 Class_0 被误改为其他类。"));
S.push(bullet("PT-MAP 的 Sinkhorn:强制每类分配比例相同,在不均衡数据上同样把分布压平。"));
S.push(bullet("alpha-TIM:α-散度更激进地推向均匀,实测 F1≈0.46,远差于基线,完全不可用。"));
S.push(bullet("非平衡调参实测:TIM(λ_marg=0)=0.6636 > TIM(λ_marg=0.05)=0.6453 → 任何平衡先验都有害。"));

S.push(h(2, "5.2 用 61k 实际预测分布判断方法优劣"));
S.push(p("由于测试集无标签,我们用各方法在 61k 上的预测分布来侧面判断:与最稳健基线(Head)分布接近、且无强制平衡迹象的方法更可信。各方法实测分布:"));
S.push(table(
  ["方法", "Class_0", "Class_1", "Class_2", "Class_3", "Class_4"],
  [
    ["Head (基线)", "43.8%", "5.1%", "15.4%", "18.9%", "16.8%"],
    ["TIM (λ_marg=0)", "45.3%", "4.3%", "15.7%", "18.8%", "15.9%"],
    ["PT-MAP nosink", "42.0%", "7.4%", "14.7%", "18.3%", "17.6%"],
    ["★ LaplacianShot", "45.3%", "5.3%", "14.7%", "18.1%", "16.6%"],
    ["TIM (λ_marg=2.0)", "25.7%", "15.6%", "19.1%", "21.0%", "18.6%"],
    ["PT-MAP + Sinkhorn", "21.2%", "18.9%", "19.4%", "20.9%", "19.5%"],
  ],
  [2800, 1340, 1340, 1340, 1340, 1340],
));
S.push(p(
  "可见:无平衡假设的 Head / TIM(λ=0) / PT-MAP nosink / LaplacianShot 分布高度一致(Class_0 ≈ 42-45%);而 TIM(λ=2.0) 和 PT-MAP+Sinkhorn 被强制压平到 ~20-26%。同学独立实现的 Linear Head 也给出 Class_0 ≈ 44%,佐证了 Class_0 是测试集多数类。",
));
S.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 六、实验结果 =====
S.push(h(1, "六、实验结果"));
S.push(h(2, "6.1 各方法对比(非平衡调参,frozen 验证)"));
S.push(table(
  ["方法", "macro-F1", "bal-acc", "最优参数"],
  [
    ["★ LaplacianShot", "0.6704", "0.6800", "tukey=1.0,knn=7,lam=10,iter=20"],
    ["TIM (λ_marg=0)", "0.6636", "0.6600", "temp=15,λ_cond=0.05,iter=500,tukey=0.7"],
    ["PT-MAP nosink", "0.6628", "0.6600", "tukey=1.0,iter=5,λ_s=5"],
    ["SimpleShot", "0.6405", "0.6400", "tukey=0.7,temp=5,combined-mean"],
    ["Mahalanobis", "0.6078", "0.6000", "tukey=1.0,shrink=0.1"],
    ["Head (基线)", "0.5828", "0.5800", "—"],
    ["alpha-TIM", "≈0.46", "—", "完全崩溃,不可用"],
  ],
  [2600, 1500, 1400, 3960],
));
S.push(p("LaplacianShot 在 frozen 上分数最高(0.6704),且分布最接近 Head——这是我们选它作最终提交的依据。", { italics: true }));

S.push(h(2, "6.2 各方法 61k 可扩展性"));
S.push(table(
  ["方法", "复杂度", "61k 时间/内存", "可用"],
  [
    ["Head / SimpleShot / Mahalanobis", "O(Nq·D) ~ O(Nq·D²)", "<1s / <1MB", "✅"],
    ["TIM", "O(Nq·Ns·iter)", "~5min / <1MB", "✅"],
    ["PT-MAP", "O(Nq·C·D·iter)", "<1min / ~1MB", "✅"],
    ["LaplacianShot (稀疏)", "O(Nq·k·iter)", "~30s / ~4MB", "✅"],
    ["Label Propagation", "O((Nq+Ns)²)", "OOM / ~15GB", "❌"],
  ],
  [3400, 2200, 2400, 1460],
));

S.push(h(2, "6.3 误差分析:模型在哪些类别上表现差、为什么"));
S.push(p("基于 5 折交叉验证的全样本(out-of-fold)预测——250 张每张都有一次「未参与训练时」的预测,可靠地反映各类难度。逐类指标:"));
S.push(table(
  ["类别", "Precision", "Recall", "F1", "视觉特征"],
  [
    ["Class_0", "0.87", "0.68", "0.76", "粉紫渐变背景,少量散在核(最易)"],
    ["Class_1", "0.75", "0.66", "0.70", "密集深紫核团簇"],
    ["Class_2", "0.43", "0.46", "0.45", "弥散紫色梯度,结构模糊(最难)"],
    ["Class_3", "0.55", "0.58", "0.56", "浅色稀疏纹理"],
    ["Class_4", "0.46", "0.56", "0.50", "混合:平滑梯度+散在核(次难)"],
  ],
  [1400, 1700, 1500, 1300, 3560],
));
S.push(img("fig_oof_confusion.png", 380));
S.push(caption("图 4　5 折 CV 全样本(OOF)混淆矩阵,红框为 Class_2↔Class_4 的相互误判"));
S.push(p("关键发现:", { bold: true, spacing: { before: 80, after: 40 } }));
S.push(bullet("Class_2(F1=0.45)和 Class_4(F1=0.50)最差,且二者互相混淆最严重:19 张 Class_2 误判为 Class_4、12 张 Class_4 误判为 Class_2。"));
S.push(bullet("原因:Class_2(弥散紫色梯度)与 Class_4(平滑梯度+散在核)在 32×32 上采后视觉特征高度接近——都呈现「平滑色彩过渡 + 部分细胞结构」,高频纹理被插值平滑后难以区分。"));
S.push(bullet("Class_0 最易(F1=0.76):颜色/密度对比鲜明,与其他类区分度高。"));
S.push(bullet("多种方法(Head/SimpleShot/Mahalanobis/LaplacianShot)在 Class_2↔4 上一致犯错,说明这是数据级的视觉模糊,而非单一算法的缺陷。"));

S.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 七、方法论诚实性 =====
S.push(h(1, "七、方法论诚实性(关键诚实声明)"));
S.push(p("本章诚实记录三件事——它们让本项目的结论可信,也是我们认为最有价值的部分。", { bold: true }));

S.push(h(2, "7.1 一次数据泄漏的发现与修正"));
S.push(p("我们曾探索 VLM+RAG 方案(用 Qwen2.5-VL-7B 对检索到的标注样本做 few-shot 推理)。"));
S.push(p("Prompt 设计:对每张查询图,先用 LoRA-DINOv2 特征检索 K=4 个最相似的标注样本,构造多图 prompt:", { spacing: { before: 60, after: 40 } }));
S.push(code("[系统] 你是 H&E 病理图像细粒度分类专家(5类),据视觉相似度判断"));
S.push(code("[支持图 ×4] 每张附:类别标签 + 与查询图的余弦相似度"));
S.push(code("[查询图] 输出严格格式: PREDICTION: Class_X / REASONING: <理由>"));
S.push(p("该方案一度在 frozen_test 上取得 macro-F1=1.0 的「完美」结果。但严谨自查后发现这是数据泄漏:"));
S.push(bullet("文件名泄漏:prompt 构造时把查询图文件名(形如 test_Class_0_xxx.png,含类别)喂给了 VLM——模型在「读文件名」而非「看图」。修复后 frozen 从虚高的 1.0 降到 0.835。"));
S.push(bullet("特征记忆泄漏:独立 holdout 等测试图都切自训练集,而 LoRA 见过它们,导致检索偏乐观。只有 frozen_test 是干净的。"));
S.push(img("fig_leakage.png", 360));
S.push(caption("图 2　文件名泄漏的影响:去掉文件名后 frozen 从虚高的 1.0 降到真实的 0.835"));
S.push(p("教训:VLM+RAG 修复泄漏后其实退化为 KNN(只复述检索邻居的多数类),无增值,故未采用。我们认为「发现并诚实纠正自己的错误」比一个虚假的满分更有价值。"));

S.push(h(2, "7.2 学习曲线:数据量是主要瓶颈"));
S.push(p("用 dev 子集(每类 10/20/30/40 张)训练,frozen 验证,得到学习曲线:50→100→200 张,macro-F1 = 0.43→0.66→0.71,在 200 张处仍未饱和。这定量证明:训练数据量是主要瓶颈,若有更多标注数据性能会继续提升;0.6~0.7 的天花板主要由「每类仅 50 张」的少样本约束决定,而非方法缺陷。"));
S.push(img("fig_learning_curve.png", 480));
S.push(caption("图 3　学习曲线:指标随训练数据量持续上升、未饱和 → 数据量是主要瓶颈"));

S.push(h(2, "7.3 50 张验证集的不可靠性"));
S.push(bullet("frozen_test 仅 50 张,单个样本对错就让 F1 变化约 2%。"));
S.push(bullet("LaplacianShot 的 F1 在不同训练配置间波动达 0.39(0.34~0.73)——50 张上的「最佳 F1」很大程度是噪声。"));
S.push(bullet("严格的 5 折交叉验证(self-training + Mahalanobis)给出 mean macro-F1 ≈ 0.59 ± 0.05;说明单次 frozen 高分(如曾经的 0.835)是划分运气导致的乐观估计,真实泛化约 0.6~0.7。"));
S.push(bullet("应对:配置之间的「趋势」(如 epochs↑、HED↑)在多配置中是稳定的,故据趋势选配置;但不迷信单次最佳分数。"));
S.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 八、最终方案与提交 =====
S.push(h(1, "八、最终方案与提交"));
S.push(h(2, "8.1 最终 pipeline"));
S.push(code("输入 32×32 → bicubic 224×224 → ImageNet 归一化"));
S.push(code("→ DINOv2 ViT-S/14 (冻结) + LoRA r=16,α=16"));
S.push(code("   训练:80 epoch + EMA + SupCon(0.1) + HED 增广,全部 250 张"));
S.push(code("→ 384-d CLS 特征 → Tukey 幂变换 + L2 归一化"));
S.push(code("→ LaplacianShot (knn=7, lam=10, iter=20, 稀疏实现)"));
S.push(code("→ submission.csv (61,881 行)"));
S.push(h(2, "8.2 方法选择理由"));
S.push(bullet("frozen 上分数最高(0.6704)"));
S.push(bullet("61k 分布(Class_0=45.3%)最接近稳健基线 Head(43.8%),无强制平衡迹象"));
S.push(bullet("无平衡假设 → 适配不平衡测试集;且利用了测试集近邻结构(拉普拉斯平滑)"));
S.push(bullet("单模型、无 TTA、无集成,严格符合题目约束"));
S.push(h(2, "8.3 提交"));
S.push(p("最终提交文件:24124035.csv(61,881 行,格式 filename,label)。", { bold: true }));
S.push(table(["指标", "结果"], [["Test macro-F1", "待评测系统返回"], ["Test balanced accuracy", "待评测系统返回"]], [3000, 6360]));
S.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 九、大模型使用 =====
S.push(h(1, "九、大模型使用情况"));
S.push(table(
  ["模型", "用途", "参数量", "许可"],
  [
    ["DINOv2 ViT-S/14", "★ 最终 backbone (LoRA 微调)", "21M", "Apache 2.0"],
    ["DINOv2 ViT-B/L、EVA-02、ConvNeXt-V2", "backbone 对比", "15-300M", "Apache/MIT"],
    ["Phikon / PLIP / CONCH / Lunit-DINO", "病理 FM 对比", "21-86M", "混合"],
    ["Qwen2.5-VL-7B 等", "VLM+RAG 探索(发现泄漏后弃用)", "2-7B", "Apache 2.0"],
  ],
  [3000, 3400, 1500, 1560],
));
S.push(p("推理成本(单卡 RTX 5090 Laptop):", { bold: true, spacing: { before: 80, after: 40 } }));
S.push(table(
  ["环节", "成本"],
  [
    ["DINOv2 ViT-S 特征提取", "<5ms/张;61k 张约 3-5 分钟"],
    ["LoRA 训练(80 epoch,250 张)", "约 5-8 分钟"],
    ["LaplacianShot 推理(61k,稀疏)", "knn 构建 ~21s + 迭代 ~0.1s"],
    ["最终方案 61k 端到端", "约 10 分钟(训练 + 推理)"],
    ["Qwen2.5-VL-7B(探索后弃用)", "~13s/张;61k 需 ~14 小时,且证实无增值"],
  ],
  [4200, 5160],
));
S.push(p("最终方案仅依赖 DINOv2 ViT-S/14(21M,LoRA 微调)+ 经典直推式推理,推理成本低,本地运行无外部 API。所有权重均为公开预训练模型。"));
S.push(new Paragraph({ children: [new PageBreak()] }));

// ===== 十、总结与反思 =====
S.push(h(1, "十、总结与反思"));
S.push(h(2, "10.1 核心结论"));
S.push(bullet("最终方案:DINOv2+LoRA(r=16)+自训练+LaplacianShot,真实泛化约 0.6~0.7。"));
S.push(bullet("最重要的方法论发现:训练均衡 + 测试不均衡时,平衡假设方法(TIM 边际熵、Sinkhorn、alpha-TIM)有害,须用无平衡假设的方法。"));
S.push(bullet("根本瓶颈是数据量(每类 50 张),学习曲线证明未饱和——这是任务约束,非方法缺陷。"));
S.push(h(2, "10.2 经验沉淀"));
S.push(bullet("先轻量 baseline(linear probe),直接揭示「数据不是病理 FM 的 sweet spot」。"));
S.push(bullet("LoRA + 自训练 + EMA 是少样本下稳健且参数高效的基础。"));
S.push(bullet("最深刻的教训:VLM+RAG 的「1.0」是数据泄漏。严谨自查(去文件名、确认验证集干净)比追逐高分更重要。"));
S.push(bullet("失败尝试(alpha-TIM、超分、强增广、各类平衡方法)同样是有价值的负结果,界定了方法的适用边界。"));
S.push(bullet("50 张验证不可靠 → 用趋势而非单次最佳分数做决策;用 61k 预测分布侧面验证方法合理性。"));
S.push(h(2, "10.3 若再给一周"));
S.push(bullet("做严格的「每折重训」交叉验证,给出真实泛化的均值±std。"));
S.push(bullet("针对最难的 Class_2↔Class_4 混淆,探索更高分辨率来源或难样本挖掘。"));
S.push(bullet("用 61k 高置信伪标签做单模型自训练(非集成),探索在真实测试分布上的自适应空间。"));

// ===== compose =====
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: FONT }, paragraph: { spacing: { before: 320, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT }, paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT }, paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 360 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "机器学习 2 · 少样本病理图像分类", font: FONT, size: 18, italics: true, color: "808080" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "第 ", font: FONT, size: 18 }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 }),
        new TextRun({ text: " 页 / 共 ", font: FONT, size: 18 }), new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 18 }),
        new TextRun({ text: " 页", font: FONT, size: 18 })] })] }) },
    children: S,
  }],
});
Packer.toBuffer(doc).then(buf => {
  const out = "E:/University/Homework/Machine Learning2/final_project/report/机器学习2课程报告_v5.docx";
  fs.writeFileSync(out, buf);
  console.log("[ok] wrote", out, "size", buf.length, "bytes");
});
