param(
    [ValidateSet("both", "qwen", "llama")]
    [string]$Model = "both",
    [switch]$EvalOnly,
    [switch]$SkipEval,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\

$ApiPy = ".\.venv-local-llm\Scripts\python.exe"
$EvalPy = ".\.venv\Scripts\python.exe"
$MaxWorkers = 4
$Runs = @(
    @{
        Name = "qwen"
        RunName = "atlas_v11_qwen25_7b_openrouter_fixed_gemini25_phaseb_n10_w4"
        ModelKey = "qwen25_7b_instruct_openrouter"
    },
    @{
        Name = "llama"
        RunName = "atlas_v11_llama31_8b_openrouter_fixed_gemini25_phaseb_n10_w4"
        ModelKey = "llama31_8b_instruct_openrouter"
    }
)

if ($Model -ne "both") {
    $Runs = @($Runs | Where-Object { $_.Name -eq $Model })
}
if (-not (Test-Path $ApiPy)) {
    throw "Missing API environment: $ApiPy"
}
if (-not $SkipEval -and -not (Test-Path $EvalPy)) {
    throw "Missing evaluation environment: $EvalPy"
}
if (-not $EvalOnly -and [string]::IsNullOrWhiteSpace($env:OPENROUTER_API_KEY)) {
    throw 'OPENROUTER_API_KEY is missing. Set $env:OPENROUTER_API_KEY in this PowerShell session.'
}

$env:NLTK_DATA = "$PWD\.nltk_data"
$env:TOKENIZERS_PARALLELISM = "false"

foreach ($Run in $Runs) {
    if (-not $EvalOnly) {
        Write-Host "=== OpenRouter (4 workers), fixed Gemini upstream: $($Run.ModelKey) ===" -ForegroundColor Cyan
        $PipelineArgs = @(
            "-m", "src.thesis_laysumm_v11.run_local_official_backbone",
            "--run-name", $Run.RunName,
            "--model-key", $Run.ModelKey,
            "--max-workers", "$MaxWorkers"
        )
        if ($NoResume) {
            $PipelineArgs += "--no-resume"
        }
        & $ApiPy @PipelineArgs
        if ($LASTEXITCODE -ne 0) {
            throw "ATLAS OpenRouter pipeline failed for $($Run.ModelKey) (exit $LASTEXITCODE)"
        }
    }

    if (-not $SkipEval) {
        Write-Host "=== Rewritten-summary evaluation: $($Run.ModelKey) ===" -ForegroundColor Cyan
        & $EvalPy -m src.thesis_laysumm.run_evaluate `
            --run-name $Run.RunName `
            --model-key $Run.ModelKey `
            --variant rewritten
        if ($LASTEXITCODE -ne 0) {
            throw "Evaluation failed for $($Run.ModelKey) (exit $LASTEXITCODE)"
        }
    }
}

if (-not $SkipEval) {
    $QwenScore = ".\results\laysumm_pipeline\runs\atlas_v11_qwen25_7b_openrouter_fixed_gemini25_phaseb_n10_w4\qwen25_7b_instruct_openrouter\rewritten_scores.json"
    $LlamaScore = ".\results\laysumm_pipeline\runs\atlas_v11_llama31_8b_openrouter_fixed_gemini25_phaseb_n10_w4\llama31_8b_instruct_openrouter\rewritten_scores.json"
}
if (-not $SkipEval -and (Test-Path $QwenScore) -and (Test-Path $LlamaScore)) {
    Write-Host "=== Validate and compare Qwen/Llama rewritten metrics ===" -ForegroundColor Cyan
    & $ApiPy -m src.thesis_laysumm_v11.compare_openrouter_official_backbones
    if ($LASTEXITCODE -ne 0) {
        throw "Final two-system comparison failed (exit $LASTEXITCODE)"
    }
} elseif ($Model -eq "both" -and -not $SkipEval) {
    throw "Both rewritten score files are required before comparison."
}

Write-Host "Done. Qwen and Llama OpenRouter n10 outputs and evaluations are stored separately. Gemini was not evaluated." -ForegroundColor Green
