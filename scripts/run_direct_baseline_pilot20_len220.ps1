param(
    [string]$RunName = "direct_zero_shot_pilot20_raw_article_len220_minimal",
    [int]$MaxWorkers = 6
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

$Models = @("gemini25_flash_non_thinking", "gemini3_flash_preview_minimal", "gpt41_mini")
$PromptPath = "src\thesis_laysumm_direct_baseline\prompts\direct_lay_summary_len220.txt"
$Py = ".\.venv\Scripts\python.exe"

function Invoke-Python {
    param([string[]]$PythonArgs)
    & $Py @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed: $($PythonArgs -join ' ')"
    }
}

Invoke-Python @("-m", "src.thesis_laysumm_direct_baseline.prepare", "create", "--run-name", $RunName, "--scope", "pilot20")
Invoke-Python @("-m", "src.thesis_laysumm_direct_baseline.validate_outputs", "--run-name", $RunName, "--expected-count", "20", "--prompt-path", $PromptPath)

foreach ($Model in $Models) {
    Invoke-Python @(
        "-m", "src.thesis_laysumm_direct_baseline.run_generate",
        "--run-name", $RunName,
        "--model-key", $Model,
        "--max-workers", "$MaxWorkers",
        "--prompt-path", $PromptPath,
        "--max-summary-words", "220"
    )
    Invoke-Python @(
        "-m", "src.thesis_laysumm_direct_baseline.validate_outputs",
        "--run-name", $RunName,
        "--model-key", $Model,
        "--expected-count", "20",
        "--prompt-path", $PromptPath
    )
}

Write-Host "[direct baseline] length-controlled pilot20 generation complete: $RunName"
