param(
    [string]$PipelineRun = "v11_constellation_validation284_fixed_m12_pilot20",
    [string]$JudgeModel = "gpt-5.4-mini-2026-03-17",
    [string]$ReasoningEffort = "medium",
    [int]$MaxWorkers = 4,
    [int]$ArticlesPerDataset = 14,
    [int]$Limit = 0,
    [switch]$DryRun,
    [switch]$NoResume,
    [switch]$SummarizeOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$ArgsList = @(
    "scripts/run_atlas_llm_judge.py",
    "--pipeline-run", $PipelineRun,
    "--judge-model", $JudgeModel,
    "--reasoning-effort", $ReasoningEffort,
    "--max-workers", "$MaxWorkers",
    "--articles-per-dataset", "$ArticlesPerDataset"
)

if ($Limit -gt 0) {
    $ArgsList += @("--limit", "$Limit")
}
if ($DryRun) {
    $ArgsList += "--dry-run"
}
if ($NoResume) {
    $ArgsList += "--no-resume"
}
if ($SummarizeOnly) {
    $ArgsList += "--summarize-only"
}

& $Python @ArgsList
