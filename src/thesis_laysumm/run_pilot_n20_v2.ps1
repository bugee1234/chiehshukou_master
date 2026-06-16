# pilot_n20_v2 — same 20 articles as v1, two models, one-shot full pipeline.
#
# One command (LLM + eval, no manual venv switching):
#   cd "C:\Users\bugee\OneDrive\桌面\chiehshukou_master"
#   .\src\thesis_laysumm\run_pilot_n20_v2.ps1
#
# Eval only (after LLM finished):
#   .\src\thesis_laysumm\run_pilot_n20_v2.ps1 -EvalOnly
#
# Force re-download articles from HuggingFace validation (not recommended):
#   .\src\thesis_laysumm\run_pilot_n20_v2.ps1 -PrepareFromHuggingFace

param(
    [string]$RunName = "pilot_n20_v2",
    [string]$ArticlesSourceRun = "pilot_n20_v1",
    [int]$Seed = 42,
    [string[]]$ModelKeys = @("gemini3_flash_preview_minimal", "gpt41_mini"),
    [int]$MaxWorkers = 8,
    [switch]$EvalOnly,
    [switch]$SkipEval,
    [switch]$PrepareFromHuggingFace
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..\

$Py = ".\.venv\Scripts\python.exe"
$EvalPy = ".\.venv-eval\Scripts\python.exe"

function Invoke-Python {
    param([string[]]$Args)
    & $Py @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed (exit $LASTEXITCODE): $Py $($Args -join ' ')"
    }
}

function Invoke-EvalPython {
    param([string[]]$Args)
    & $EvalPy @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed (exit $LASTEXITCODE): $EvalPy $($Args -join ' ')"
    }
}

function Invoke-EvaluatePhase {
    if (-not (Test-Path $EvalPy)) {
        throw "Missing $EvalPy — create .venv-eval first (see evaluation/LOCAL_TEST_MACHINE_SETUP.md)"
    }
    $env:NLTK_DATA = "$PWD\.nltk_data"
    $env:HF_HOME = "C:\hf_cache"
    foreach ($ModelKey in $ModelKeys) {
        Write-Host "=== Phase C: evaluate $ModelKey ===" -ForegroundColor Cyan
        Invoke-EvalPython @(
            "-m", "src.thesis_laysumm.run_evaluate",
            "--run-name", $RunName,
            "--model-key", $ModelKey,
            "--variant", "both",
            "--compare-leaderboard"
        )
    }
}

function Ensure-Articles {
    $articlesDir = "data\laysumm_pipeline\runs\$RunName\00_articles"
    $articlesFile = Join-Path $articlesDir "articles.jsonl"

    if ($PrepareFromHuggingFace) {
        Write-Host "=== Step 0: prepare_articles from HuggingFace validation ===" -ForegroundColor Cyan
        $env:HF_HOME = "C:\hf_cache"
        Invoke-Python @(
            "-m", "src.thesis_laysumm.prepare_articles",
            "--run-name", $RunName,
            "--seed", "$Seed",
            "--n-per-source", "10",
            "--skip-full-split-audit"
        )
        return
    }

    if (Test-Path $articlesFile) {
        $count = (Get-Content $articlesFile | Measure-Object -Line).Lines
        Write-Host "=== Step 0: skip (found $articlesFile, $count articles) ===" -ForegroundColor Cyan
        if ($count -ne 20) {
            Write-Warning "Expected 20 articles, found $count — check run selection."
        }
        return
    }

    $sourceDir = "data\laysumm_pipeline\runs\$ArticlesSourceRun\00_articles"
    $sourceFile = Join-Path $sourceDir "articles.jsonl"
    if (-not (Test-Path $sourceFile)) {
        throw "Missing $sourceFile — run pilot_n20_v1 prepare_articles first, or use -PrepareFromHuggingFace"
    }

    Write-Host "=== Step 0: copy 20 articles from $ArticlesSourceRun -> $RunName ===" -ForegroundColor Cyan
    $runDir = "data\laysumm_pipeline\runs\$RunName"
    if (Test-Path $articlesDir) {
        Remove-Item -Recurse -Force $articlesDir
    }
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    # Copy into run dir so we get runs\<run>\00_articles\articles.jsonl (not nested 00_articles\00_articles)
    Copy-Item -Recurse -Force $sourceDir $runDir
    if (-not (Test-Path $articlesFile)) {
        throw "Copy failed: expected $articlesFile"
    }
    $count = (Get-Content $articlesFile | Measure-Object -Line).Lines
    Write-Host "Copied $count articles (no HuggingFace download)" -ForegroundColor Green
}

if ($EvalOnly) {
    Invoke-EvaluatePhase
    Write-Host "Done (eval only)." -ForegroundColor Green
    exit 0
}

if (-not (Test-Path $Py)) {
    throw "Missing $Py — activate or create .venv for LLM pipeline steps"
}

Ensure-Articles

foreach ($ModelKey in $ModelKeys) {
    Write-Host "=== Phase A: $ModelKey ===" -ForegroundColor Green

    Invoke-Python @("-m", "src.thesis_laysumm.run_module1", "--run-name", $RunName, "--model-key", $ModelKey, "--max-workers", "$MaxWorkers")
    Invoke-Python @("-m", "src.thesis_laysumm.run_module2", "--run-name", $RunName, "--model-key", $ModelKey, "--max-workers", "$MaxWorkers")
    Invoke-Python @("-m", "src.thesis_laysumm.run_module3_questions", "--run-name", $RunName, "--model-key", $ModelKey, "--max-workers", "$MaxWorkers")

    Write-Host "=== Phase B: $ModelKey ===" -ForegroundColor Green

    Invoke-Python @("-m", "src.thesis_laysumm.run_phase_b", "--run-name", $RunName, "--model-key", $ModelKey, "--step", "generate-summary", "--max-workers", "$MaxWorkers")
    Invoke-Python @("-m", "src.thesis_laysumm.run_phase_b", "--run-name", $RunName, "--model-key", $ModelKey, "--step", "repair-summaries")
    Invoke-Python @("-m", "src.thesis_laysumm.run_phase_b", "--run-name", $RunName, "--model-key", $ModelKey, "--step", "answer-questions", "--max-workers", "$MaxWorkers")
    Invoke-Python @("-m", "src.thesis_laysumm.run_phase_b", "--run-name", $RunName, "--model-key", $ModelKey, "--step", "repair-answers")
    Invoke-Python @("-m", "src.thesis_laysumm.run_phase_b", "--run-name", $RunName, "--model-key", $ModelKey, "--step", "rewrite-summaries", "--max-workers", "4")
}

if (-not $SkipEval) {
    Invoke-EvaluatePhase
}

Write-Host "Done. Results: results\laysumm_pipeline\runs\$RunName\<model_key>\" -ForegroundColor Green
