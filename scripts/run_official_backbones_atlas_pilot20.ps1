param(
    [ValidateSet("both", "qwen", "llama")]
    [string]$Model = "both",
    [switch]$EvalOnly,
    [switch]$SkipEval,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\

$LocalPy = ".\.venv-local-llm\Scripts\python.exe"
$EvalPy = ".\.venv\Scripts\python.exe"
$Runs = @(
    @{
        Name = "qwen"
        RunName = "atlas_v11_qwen25_7b_fixed_gemini25_phaseb_n10_compact"
        ModelKey = "qwen25_7b_instruct_local_8bit"
    },
    @{
        Name = "llama"
        RunName = "atlas_v11_llama3_8b_fixed_gemini25_phaseb_n10_compact"
        ModelKey = "llama3_8b_instruct_local_8bit"
    }
)

if ($Model -ne "both") {
    $Runs = @($Runs | Where-Object { $_.Name -eq $Model })
}
if (-not (Test-Path $LocalPy)) {
    throw "Missing $LocalPy"
}
if (-not $SkipEval -and -not (Test-Path $EvalPy)) {
    throw "Missing evaluation environment: $EvalPy"
}

$OriginalHFHome = $env:HF_HOME
$LocalHFHome = "$PWD\.cache\huggingface-local-llm"
$env:NLTK_DATA = "$PWD\.nltk_data"
$env:TOKENIZERS_PARALLELISM = "false"

foreach ($Run in $Runs) {
    if (-not $EvalOnly) {
        $env:HF_HOME = $LocalHFHome
        Write-Host "=== Fixed Gemini 2.5 upstream + ATLAS Phase B: $($Run.ModelKey) ===" -ForegroundColor Cyan
        $PipelineArgs = @(
            "-m", "src.thesis_laysumm_v11.run_local_official_backbone",
            "--run-name", $Run.RunName,
            "--model-key", $Run.ModelKey
        )
        if ($NoResume) {
            $PipelineArgs += "--no-resume"
        }
        & $LocalPy @PipelineArgs
        if ($LASTEXITCODE -ne 0) {
            throw "ATLAS pipeline failed for $($Run.ModelKey) (exit $LASTEXITCODE)"
        }
    }

    if (-not $SkipEval) {
        if ($null -eq $OriginalHFHome -or $OriginalHFHome -eq "") {
            Remove-Item Env:HF_HOME -ErrorAction SilentlyContinue
        } else {
            $env:HF_HOME = $OriginalHFHome
        }
        Write-Host "=== Evaluation: $($Run.ModelKey) ===" -ForegroundColor Cyan
        & $EvalPy -m src.thesis_laysumm.run_evaluate `
            --run-name $Run.RunName `
            --model-key $Run.ModelKey `
            --variant both `
            --compare-leaderboard
        if ($LASTEXITCODE -ne 0) {
            throw "Evaluation failed for $($Run.ModelKey) (exit $LASTEXITCODE)"
        }
        & $LocalPy -m src.thesis_laysumm_v11.validate_outputs `
            --run-name $Run.RunName `
            --model-key $Run.ModelKey `
            --mode factuality_chase `
            --stage eval
        if ($LASTEXITCODE -ne 0) {
            throw "Evaluation validation failed for $($Run.ModelKey) (exit $LASTEXITCODE)"
        }
    }
}

if ($Model -eq "both" -and -not $SkipEval) {
    Write-Host "=== Prepare and evaluate same-n10 Gemini 2.5 reference ===" -ForegroundColor Cyan
    & $LocalPy -m src.thesis_laysumm_v11.prepare_gemini_reference_n10
    if ($LASTEXITCODE -ne 0) {
        throw "Gemini 2.5 n10 reference preparation failed (exit $LASTEXITCODE)"
    }
    & $EvalPy -m src.thesis_laysumm.run_evaluate `
        --run-name atlas_v11_gemini25_fixed_upstream_phaseb_n10_compact `
        --model-key gemini25_flash_non_thinking `
        --variant both `
        --compare-leaderboard
    if ($LASTEXITCODE -ne 0) {
        throw "Gemini 2.5 n10 reference evaluation failed (exit $LASTEXITCODE)"
    }
    & $EvalPy -m src.thesis_laysumm_v11.compare_official_backbones
    if ($LASTEXITCODE -ne 0) {
        throw "Final three-system comparison table failed (exit $LASTEXITCODE)"
    }
}

Write-Host "Done. Qwen and Llama n10 Phase-B results are stored in separate data/results run folders." -ForegroundColor Green
