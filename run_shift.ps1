# 染色分布偏移压力测试：依次跑 轻/中/重 三档
# 检索库 = holdout_pool（原始无偏移），测试集 = shift_test_*（偏移）
$py = "C:\Users\Lenovo\.conda\envs\ml2_fewshot\python.exe"
$base = "E:\University\Homework\Machine Learning2\final_project"
$ckpt = "$base\artifacts\best_v02_f0.8351_mahalanobis\round_3\ckpt.pt"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"

foreach ($lvl in @("light", "mid", "heavy")) {
    Write-Host "===== STAIN SHIFT LEVEL: $lvl ====="
    & $py -u "$base\src\vlm_rag_predict.py" `
        --vlm Qwen/Qwen2.5-VL-7B-Instruct `
        --train_dir "$base\holdout_pool" `
        --test_dir "$base\shift_test_$lvl" `
        --ckpt $ckpt `
        --out "$base\sub_shift_$lvl.csv" `
        --gt_csv "$base\shift_test_$lvl\_groundtruth.csv" `
        --topk 4
    Write-Host "===== DONE: $lvl ====="
}
Write-Host "===== ALL STAIN-SHIFT TESTS COMPLETE ====="
