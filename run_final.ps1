# ============================================================================
#  一键最终提交脚本 —— 老师测试集到来时使用
#
#  使用方法：
#    1. 把老师的测试集图片放到某个目录（一堆 .png，无需子目录）
#    2. 修改下面的 $TestDir 为那个目录的路径
#    3. 右键“使用 PowerShell 运行”，或在终端执行：  .\run_final.ps1
#
#  输出：  submission_final.csv  （filename, label 两列，可直接提交）
#
#  说明：单模型 Qwen2.5-VL-7B + RAG，检索池 = 全部 250 张标注数据。
#        无 TTA、无模型集成。
# ============================================================================

# >>>>>>>>>>>>>>>>  只需修改这一行  <<<<<<<<<<<<<<<<
$TestDir = "E:\University\Homework\Machine Learning2\final_project\TEACHER_TEST"
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ---- 固定配置（一般不用改）----
$ProjRoot  = "E:\University\Homework\Machine Learning2\final_project"
$PythonExe = "C:\Users\Lenovo\.conda\envs\ml2_fewshot\python.exe"
$TrainDir  = "$ProjRoot\train_few_shot"                                   # 250 张检索池
$Ckpt      = "$ProjRoot\artifacts\best_v02_f0.8351_mahalanobis\round_3\ckpt.pt"
$OutCsv    = "$ProjRoot\submission_final.csv"
$Vlm       = "Qwen/Qwen2.5-VL-7B-Instruct"
$TopK      = 4

# ---- 环境 ----
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"

# ---- 检查测试目录 ----
if (-not (Test-Path $TestDir)) {
    Write-Host "❌ 测试目录不存在: $TestDir" -ForegroundColor Red
    Write-Host "   请修改脚本顶部的 `$TestDir 变量" -ForegroundColor Yellow
    exit 1
}
$nImg = (Get-ChildItem -Path $TestDir -Filter *.png -Recurse).Count
Write-Host "✅ 测试目录: $TestDir  ($nImg 张图)" -ForegroundColor Green
Write-Host "✅ 检索池:   $TrainDir  (250 张标注数据)" -ForegroundColor Green
Write-Host "✅ 模型:     $Vlm  (K=$TopK)" -ForegroundColor Green
Write-Host ""

# ---- 运行推理 ----
& $PythonExe -u "$ProjRoot\src\vlm_rag_predict.py" `
    --vlm $Vlm `
    --train_dir $TrainDir `
    --test_dir  $TestDir `
    --ckpt      $Ckpt `
    --out       $OutCsv `
    --topk      $TopK

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "🎉 完成！提交文件: $OutCsv" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ 推理失败，退出码 $LASTEXITCODE" -ForegroundColor Red
}
