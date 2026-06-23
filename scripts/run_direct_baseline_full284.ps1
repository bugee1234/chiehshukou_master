param(
    [int]$MaxWorkers = 8
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

Remove-Item Env:\GEMINI_API_KEY -ErrorAction SilentlyContinue

$PilotRun = "direct_zero_shot_pilot20_raw_article"
$RunName = "direct_zero_shot_validation284_raw_article"
$PipelineRun = "v11_constellation_validation284_fixed_m12_pilot20"
$Models = @("gemini25_flash_non_thinking", "gpt41_mini", "gemini3_flash_preview_minimal")
$Py = ".\.venv\Scripts\python.exe"
$EvalPy = ".\.venv-eval\Scripts\python.exe"

$env:NLTK_DATA = "$PWD\.nltk_data"
$env:HF_HOME = "C:\hf_cache"

function Invoke-Python {
    param(
        [string[]]$PythonArgs,
        [switch]$Eval
    )
    if ($Eval) {
        & $EvalPy @PythonArgs
    } else {
        & $Py @PythonArgs
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed: $($PythonArgs -join ' ')"
    }
}

Invoke-Python @("scripts\check_direct_baseline.py")
Invoke-Python @("-m", "src.thesis_laysumm_direct_baseline.prepare", "create", "--run-name", $RunName, "--scope", "full284")
Invoke-Python @("-m", "src.thesis_laysumm_direct_baseline.validate_outputs", "--run-name", $RunName, "--expected-count", "284")
$SeedArgs = @("-m", "src.thesis_laysumm_direct_baseline.prepare", "seed-pilot", "--pilot-run", $PilotRun, "--full-run", $RunName, "--models") + $Models
Invoke-Python -PythonArgs $SeedArgs

foreach ($Model in $Models) {
    Invoke-Python @("-m", "src.thesis_laysumm_direct_baseline.run_generate", "--run-name", $RunName, "--model-key", $Model, "--max-workers", "$MaxWorkers")
    Invoke-Python @("-m", "src.thesis_laysumm_direct_baseline.validate_outputs", "--run-name", $RunName, "--model-key", $Model, "--expected-count", "284")
}

foreach ($Model in $Models) {
    Invoke-Python -Eval -PythonArgs @("-m", "src.thesis_laysumm_direct_baseline.run_evaluate", "--run-name", $RunName, "--model-key", $Model, "--compare-pipeline-run", $PipelineRun, "--save-per-article")
}

Invoke-Python @("-m", "src.thesis_laysumm_direct_baseline.summarize_comparison", "--run-name", $RunName, "--pipeline-run", $PipelineRun)
Write-Host "[direct baseline] full284 generation, evaluation, and comparison complete: $RunName"
