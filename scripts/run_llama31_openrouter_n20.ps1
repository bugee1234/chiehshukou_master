param(
    [switch]$EvalOnly,
    [switch]$SkipEval,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\

$ApiPy = ".\.venv-local-llm\Scripts\python.exe"
$EvalPy = ".\.venv\Scripts\python.exe"
$RunName = "atlas_v11_llama31_8b_openrouter_fixed_gemini25_phaseb_n20_w4"
$ModelKey = "llama31_8b_instruct_openrouter"
$MaxWorkers = 4

if (-not (Test-Path $ApiPy)) {
    throw "Missing API environment: $ApiPy"
}
if (-not $SkipEval -and -not (Test-Path $EvalPy)) {
    throw "Missing evaluation environment: $EvalPy"
}
if (-not $EvalOnly -and [string]::IsNullOrWhiteSpace($env:OPENROUTER_API_KEY)) {
    throw 'OPENROUTER_API_KEY is missing. Set it in this PowerShell session.'
}

$env:NLTK_DATA = "$PWD\.nltk_data"
$env:TOKENIZERS_PARALLELISM = "false"

if (-not $EvalOnly) {
    Write-Host "=== Llama 3.1 OpenRouter: 10 PLOS + 10 eLife, 4 workers ===" -ForegroundColor Cyan
    $PipelineArgs = @(
        "-m", "src.thesis_laysumm_v11.run_local_official_backbone",
        "--run-name", $RunName,
        "--model-key", $ModelKey,
        "--articles-per-source", "10",
        "--max-workers", "$MaxWorkers"
    )
    if ($NoResume) {
        $PipelineArgs += "--no-resume"
    }
    & $ApiPy @PipelineArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Llama 3.1 n20 ATLAS pipeline failed (exit $LASTEXITCODE)"
    }
}

if (-not $SkipEval) {
    Write-Host "=== Llama 3.1 n20 rewritten-summary evaluation ===" -ForegroundColor Cyan
    & $EvalPy -m src.thesis_laysumm.run_evaluate `
        --run-name $RunName `
        --model-key $ModelKey `
        --variant rewritten
    if ($LASTEXITCODE -ne 0) {
        throw "Llama 3.1 n20 evaluation failed (exit $LASTEXITCODE)"
    }
}

Write-Host "Done. Llama 3.1 n20 results were stored separately; the existing n10 run was not modified." -ForegroundColor Green
