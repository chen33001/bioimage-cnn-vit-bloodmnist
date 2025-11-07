Param(
    [string]$DataDir = "data",
    [int]$Epochs = 10,
    [int]$Batch = 128,
    [int]$Seed = 42,
    [int]$SamplesPerClass = 10,
    [switch]$ReducedCompute,
    [string]$LogDir = "logs"
)

$ErrorActionPreference = 'Stop'

function Write-Timestamp {
    param(
        [string]$Message
    )
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Write-Host "[$ts] $Message"
}

$startTime = Get-Date

$modeLabel = if ($ReducedCompute) { "reduced" } else { "full" }
$epochsToUse = if ($ReducedCompute) { [Math]::Min($Epochs, 3) } else { $Epochs }
$batchToUse = if ($ReducedCompute) { [Math]::Min($Batch, 64) } else { $Batch }
$imageSize = if ($ReducedCompute) { 128 } else { 224 }

$cnnModelsDir = if ($ReducedCompute) { "models\\reduced\\cnn" } else { "models\\cnn" }
$vitModelsDir = if ($ReducedCompute) { "models\\reduced\\vit" } else { "models\\vit" }
$resultsDir = if ($ReducedCompute) { "results\\reduced" } else { "results" }
$figuresDir = if ($ReducedCompute) { "figures\\reduced" } else { "figures" }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$logPath = Join-Path $LogDir ("run_all_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

Add-Content -Path $logPath -Value "Run-All log ($modeLabel mode)"
Add-Content -Path $logPath -Value "======================================="

function Write-Log {
    param(
        [string]$Message
    )
    Add-Content -Path $logPath -Value $Message
}

Write-Timestamp "[Run-All] Setting up directories (mode: $modeLabel)..."
Write-Log "[Run-All] Setting up directories (mode: $modeLabel)..."
New-Item -ItemType Directory -Force -Path $cnnModelsDir | Out-Null
New-Item -ItemType Directory -Force -Path $vitModelsDir | Out-Null
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $figuresDir | Out-Null

Write-Timestamp "[Run-All] Training CNN (resnet18)..."
Write-Log "[Run-All] Training CNN (resnet18)..."
$cnnStart = Get-Date
python -m src.train_cnn --arch resnet18 --epochs $epochsToUse --batch-size $batchToUse --seed $Seed --data-dir $DataDir --models-dir $cnnModelsDir --results-dir $resultsDir --figures-dir $figuresDir --image-size $imageSize
$cnnEnd = Get-Date
$cnnDuration = New-TimeSpan -Start $cnnStart -End $cnnEnd
$cnnMsg = "[Run-All] CNN training duration : $("{0:hh\:mm\:ss}" -f $cnnDuration)"
Write-Timestamp $cnnMsg
Write-Log $cnnMsg

Write-Timestamp "[Run-All] Training ViT (vit_base_patch16_224)..."
Write-Log "[Run-All] Training ViT (vit_base_patch16_224)..."
$vitStart = Get-Date
python -m src.train_vit --arch vit_base_patch16_224 --epochs $epochsToUse --batch-size $batchToUse --seed $Seed --data-dir $DataDir --models-dir $vitModelsDir --results-dir $resultsDir --figures-dir $figuresDir --image-size $imageSize
$vitEnd = Get-Date
$vitDuration = New-TimeSpan -Start $vitStart -End $vitEnd
$vitMsg = "[Run-All] ViT training duration : $("{0:hh\:mm\:ss}" -f $vitDuration)"
Write-Timestamp $vitMsg
Write-Log $vitMsg

Write-Timestamp "[Run-All] Aggregating metrics..."
Write-Log "[Run-All] Aggregating metrics..."
$aggStart = Get-Date
$finalCsv = Join-Path $resultsDir "final_comparison.csv"
$summaryPath = Join-Path $resultsDir "final_summary.md"
python -m src.aggregate_results --results-dir $resultsDir --figures-dir $figuresDir --output-csv $finalCsv --summary-path $summaryPath --mode-label $modeLabel
$aggEnd = Get-Date
$aggDuration = New-TimeSpan -Start $aggStart -End $aggEnd
$aggMsg = "[Run-All] Aggregation duration : $("{0:hh\:mm\:ss}" -f $aggDuration)"
Write-Timestamp $aggMsg
Write-Log $aggMsg

Write-Timestamp "[Run-All] Generating interpretability overlays..."
Write-Log "[Run-All] Generating interpretability overlays..."
$interpStart = Get-Date
$gradcamDir = Join-Path $figuresDir "gradcam"
$attentionDir = Join-Path $figuresDir "attention"
New-Item -ItemType Directory -Force -Path $gradcamDir | Out-Null
New-Item -ItemType Directory -Force -Path $attentionDir | Out-Null

$cnnCkpt = Join-Path $cnnModelsDir "resnet18_best.pt"
if (Test-Path $cnnCkpt) {
    python -m src.interpretability --model-type cnn --arch resnet18 --checkpoint $cnnCkpt --split test --samples-per-class $SamplesPerClass --output-dir $gradcamDir --seed $Seed --data-dir $DataDir --image-size $imageSize
} else {
    Write-Warning "CNN checkpoint not found at $cnnCkpt; skipping Grad-CAM overlays."
}
$vitCkpt = Join-Path $vitModelsDir "vit_base_patch16_224_best.pt"
if (Test-Path $vitCkpt) {
    python -m src.interpretability --model-type vit --arch vit_base_patch16_224 --checkpoint $vitCkpt --split test --samples-per-class $SamplesPerClass --output-dir $attentionDir --seed $Seed --data-dir $DataDir --image-size $imageSize
} else {
    Write-Warning "ViT checkpoint not found at $vitCkpt; skipping attention overlays."
    Write-Log "ViT checkpoint not found at $vitCkpt; skipping attention overlays."
}
$interpEnd = Get-Date
$interpDuration = New-TimeSpan -Start $interpStart -End $interpEnd
$interpMsg = "[Run-All] Interpretability duration : $("{0:hh\:mm\:ss}" -f $interpDuration)"
Write-Timestamp $interpMsg
Write-Log $interpMsg

$endTime = Get-Date
$duration = New-TimeSpan -Start $startTime -End $endTime
Write-Timestamp "[Run-All] Done. See $resultsDir and $cnnModelsDir / $vitModelsDir."
Write-Log "[Run-All] Done. See $resultsDir and $cnnModelsDir / $vitModelsDir."

$startMsg = "[Run-All] Start time : $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
$endMsg = "[Run-All] End time   : $($endTime.ToString('yyyy-MM-dd HH:mm:ss'))"
$durationMsg = "[Run-All] Duration   : $("{0:hh\:mm\:ss}" -f $duration)"

Write-Timestamp $startMsg
Write-Timestamp $endMsg
Write-Timestamp $durationMsg

Write-Log $startMsg
Write-Log $endMsg
Write-Log $durationMsg

Write-Timestamp "[Run-All] Log saved to $logPath"
Write-Log "[Run-All] Log saved to $logPath"
