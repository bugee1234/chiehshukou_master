# V6 pilot runner.
#
# Examples:
#   .\src\thesis_laysumm_v6\run_pilot_n20_v6.ps1 -Mode factuality_chase
#   .\src\thesis_laysumm_v6\run_pilot_n20_v6.ps1 -Mode factuality_chase -RunNamePrefix pilot_n10_v6 -ArticlesPerSource 5 -ModelKeys @("gpt41_mini", "gemini3_flash_preview_minimal")
#
# Outputs:
#   data\laysumm_pipeline\runs\pilot_n20_v6_factuality_chase
#   results\laysumm_pipeline\runs\pilot_n20_v6_factuality_chase

param(
    [ValidateSet("balanced", "factuality_chase", "both")]
    [string]$Mode = "factuality_chase",
    [string]$RunNamePrefix = "pilot_n20_v6",
    [string]$ArticlesSourceRun = "pilot_n20_v2",
    [int]$ArticlesPerSource = 0,
    [string[]]$ModelKeys = @("gpt41_mini"),
    [int]$MaxWorkers = 8,
    [switch]$EvalOnly,
    [switch]$SkipEval,
    [switch]$NoResumePhaseB
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..\

$Py = ".\.venv\Scripts\python.exe"
$EvalPy = ".\.venv-eval\Scripts\python.exe"

function Invoke-Python {
    param([string[]]$PythonArgs)
    & $Py @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed (exit $LASTEXITCODE): $Py $($PythonArgs -join ' ')"
    }
}

function Invoke-EvalPython {
    param([string[]]$PythonArgs)
    & $EvalPy @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed (exit $LASTEXITCODE): $EvalPy $($PythonArgs -join ' ')"
    }
}

function Ensure-Articles {
    param([string]$RunName)

    $articlesDir = "data\laysumm_pipeline\runs\$RunName\00_articles"
    $articlesFile = Join-Path $articlesDir "articles.jsonl"
    if (Test-Path $articlesFile) {
        $count = (Get-Content $articlesFile | Measure-Object -Line).Lines
        Write-Host "=== Step 0: skip articles for $RunName ($count rows) ===" -ForegroundColor Cyan
        return
    }

    $sourceDir = "data\laysumm_pipeline\runs\$ArticlesSourceRun\00_articles"
    $sourceFile = Join-Path $sourceDir "articles.jsonl"
    if (-not (Test-Path $sourceFile)) {
        throw "Missing $sourceFile. Run v2 article preparation first."
    }

    $runDir = "data\laysumm_pipeline\runs\$RunName"
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    if ($ArticlesPerSource -gt 0) {
        New-Item -ItemType Directory -Force -Path $articlesDir | Out-Null
        $sourceRows = Get-Content $sourceFile | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }
        $selected = @()
        foreach ($Source in @("PLOS", "eLife")) {
            $sourceSelected = @($sourceRows | Where-Object { $_.source_dataset -eq $Source } | Select-Object -First $ArticlesPerSource)
            if ($sourceSelected.Count -ne $ArticlesPerSource) {
                throw "Expected $ArticlesPerSource $Source articles from $sourceFile, found $($sourceSelected.Count)."
            }
            $selected += $sourceSelected
        }
        $jsonLines = @($selected | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 100 })
        $utf8NoBom = New-Object System.Text.UTF8Encoding -ArgumentList $false
        [System.IO.File]::WriteAllLines((Resolve-Path $articlesDir).Path + "\articles.jsonl", $jsonLines, $utf8NoBom)
        $count = $jsonLines.Count
        Write-Host "Selected $count articles into $RunName ($ArticlesPerSource per source)" -ForegroundColor Green
        return
    }

    Copy-Item -Recurse -Force $sourceDir $runDir
    if (-not (Test-Path $articlesFile)) {
        throw "Copy failed: expected $articlesFile"
    }
    $count = (Get-Content $articlesFile | Measure-Object -Line).Lines
    Write-Host "Copied $count articles into $RunName" -ForegroundColor Green
}

function Invoke-Evaluate {
    param([string]$RunName)
    if (-not (Test-Path $EvalPy)) {
        throw "Missing $EvalPy. Create .venv-eval first."
    }
    $env:NLTK_DATA = "$PWD\.nltk_data"
    $env:HF_HOME = "C:\hf_cache"
    foreach ($ModelKey in $ModelKeys) {
        Write-Host "=== V6 Eval: $RunName / $ModelKey ===" -ForegroundColor Cyan
        Invoke-EvalPython @(
            "-m", "src.thesis_laysumm.run_evaluate",
            "--run-name", $RunName,
            "--model-key", $ModelKey,
            "--variant", "both",
            "--compare-leaderboard"
        )
    }
}

function Invoke-V6Run {
    param([string]$OneMode)

    $RunName = "$RunNamePrefix`_$OneMode"
    Write-Host "=== V6 $OneMode :: $RunName ===" -ForegroundColor Green

    if ($EvalOnly) {
        Invoke-Evaluate -RunName $RunName
        return
    }

    if (-not (Test-Path $Py)) {
        throw "Missing $Py. Create .venv first."
    }

    Ensure-Articles -RunName $RunName

    foreach ($ModelKey in $ModelKeys) {
        Write-Host "=== V6 base extraction: $RunName / $ModelKey ===" -ForegroundColor Cyan
        Invoke-Python @("-m", "src.thesis_laysumm.run_module1", "--run-name", $RunName, "--model-key", $ModelKey, "--max-workers", "$MaxWorkers")
        Invoke-Python @("-m", "src.thesis_laysumm.run_module2", "--run-name", $RunName, "--model-key", $ModelKey, "--max-workers", "$MaxWorkers", "--judgements-only")

        Write-Host "=== V6 evidence + questions: $RunName / $ModelKey ===" -ForegroundColor Cyan
        Invoke-Python @("-m", "src.thesis_laysumm_v6.run_evidence", "--run-name", $RunName, "--model-key", $ModelKey, "--mode", $OneMode, "--max-workers", "$MaxWorkers")
        Invoke-Python @("-m", "src.thesis_laysumm_v6.validate_outputs", "--run-name", $RunName, "--model-key", $ModelKey, "--mode", $OneMode, "--stage", "evidence")
        Invoke-Python @("-m", "src.thesis_laysumm_v6.run_questions", "--run-name", $RunName, "--model-key", $ModelKey, "--mode", $OneMode, "--max-workers", "$MaxWorkers")
        Invoke-Python @("-m", "src.thesis_laysumm_v6.validate_outputs", "--run-name", $RunName, "--model-key", $ModelKey, "--mode", $OneMode, "--stage", "questions")

        Write-Host "=== V6 generate/check/rewrite: $RunName / $ModelKey ===" -ForegroundColor Cyan
        $NoResumeArg = if ($NoResumePhaseB) { @("--no-resume") } else { @() }
        $GenerateArgs = @("-m", "src.thesis_laysumm_v6.run_phase_b", "--run-name", $RunName, "--model-key", $ModelKey, "--mode", $OneMode, "--step", "generate-summary", "--max-workers", "$MaxWorkers") + $NoResumeArg
        Invoke-Python $GenerateArgs
        Invoke-Python @("-m", "src.thesis_laysumm_v6.validate_outputs", "--run-name", $RunName, "--model-key", $ModelKey, "--mode", $OneMode, "--stage", "summaries")
        $AnswerArgs = @("-m", "src.thesis_laysumm_v6.run_phase_b", "--run-name", $RunName, "--model-key", $ModelKey, "--mode", $OneMode, "--step", "answer-questions", "--max-workers", "$MaxWorkers") + $NoResumeArg
        Invoke-Python $AnswerArgs
        Invoke-Python @("-m", "src.thesis_laysumm_v6.validate_outputs", "--run-name", $RunName, "--model-key", $ModelKey, "--mode", $OneMode, "--stage", "answers")
        $RewriteArgs = @("-m", "src.thesis_laysumm_v6.run_phase_b", "--run-name", $RunName, "--model-key", $ModelKey, "--mode", $OneMode, "--step", "rewrite-summaries", "--max-workers", "4") + $NoResumeArg
        Invoke-Python $RewriteArgs
        Invoke-Python @("-m", "src.thesis_laysumm_v6.validate_outputs", "--run-name", $RunName, "--model-key", $ModelKey, "--mode", $OneMode, "--stage", "rewritten")
    }

    if (-not $SkipEval) {
        Invoke-Evaluate -RunName $RunName
    }
}

$ModesToRun = if ($Mode -eq "both") { @("balanced", "factuality_chase") } else { @($Mode) }
foreach ($OneMode in $ModesToRun) {
    Invoke-V6Run -OneMode $OneMode
}

Write-Host "Done. V6 outputs are under data/results laysumm_pipeline runs with prefix $RunNamePrefix." -ForegroundColor Green
