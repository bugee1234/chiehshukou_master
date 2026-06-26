param(
  [string]$RunName = "m1eng_len220direct_abstractclaim_recall_g25judge_p20",
  [string]$AtlasRun = "v11_constellation_validation284_fixed_m12_pilot20",
  [string]$PilotRun = "pilot_n20_v11_factuality_chase",
  [string]$DirectRun = "direct_zero_shot_pilot20_raw_article_len220",
  [string[]]$ModelKeys = @("gemini25_flash_non_thinking", "gemini3_flash_preview_minimal", "gpt41_mini"),
  [string]$JudgeModelKey = "gemini25_flash_non_thinking",
  [string[]]$EvaluateVariants = @("module1"),
  [int]$MaxWorkers = 6
)

$ErrorActionPreference = "Stop"

$Py = ".\.venv\Scripts\python.exe"
$EvalPy = ".\.venv-eval\Scripts\python.exe"
$Module = "src.thesis_laysumm_ablation_m1_recall.pipeline"

function Invoke-Checked {
  param(
    [string]$Exe,
    [string[]]$ArgsForExe
  )
  Write-Host ""
  Write-Host "> $Exe $($ArgsForExe -join ' ')"
  & $Exe @ArgsForExe
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $Exe $($ArgsForExe -join ' ')"
  }
}

if (-not (Test-Path $Py)) {
  throw "Missing Python environment: $Py"
}
if (-not (Test-Path $EvalPy)) {
  throw "Missing evaluation Python environment: $EvalPy"
}

$LocalNltkData = Resolve-Path ".\.nltk_data"
$NltkPaths = @("$LocalNltkData")
if ($env:NLTK_DATA) {
  $NltkPaths += ($env:NLTK_DATA -split ';' | Where-Object { $_ -and $_ -ne "$LocalNltkData" })
}
$env:NLTK_DATA = ($NltkPaths | Select-Object -Unique) -join ';'
$env:HF_HOME = "C:\hf_cache"
Write-Host "NLTK_DATA=$env:NLTK_DATA"
Write-Host "HF_HOME=$env:HF_HOME"

Invoke-Checked $Py @("-m", $Module, "--run-name", $RunName, "--atlas-run", $AtlasRun, "--pilot-run", $PilotRun, "--direct-run", $DirectRun, "prepare-sample")
Invoke-Checked $Py (@("-m", $Module, "--run-name", $RunName, "--atlas-run", $AtlasRun, "--pilot-run", $PilotRun, "--direct-run", $DirectRun, "validate-inputs", "--model-keys") + $ModelKeys)

foreach ($ModelKey in $ModelKeys) {
  Invoke-Checked $Py @("-m", $Module, "--run-name", $RunName, "--atlas-run", $AtlasRun, "--pilot-run", $PilotRun, "--direct-run", $DirectRun, "--model-key", $ModelKey, "--max-workers", "$MaxWorkers", "generate-module1")
  Invoke-Checked $Py @("-m", $Module, "--run-name", $RunName, "--atlas-run", $AtlasRun, "--pilot-run", $PilotRun, "--direct-run", $DirectRun, "--model-key", $ModelKey, "build-variants")
}

Invoke-Checked $Py @("-m", $Module, "--run-name", $RunName, "--atlas-run", $AtlasRun, "--pilot-run", $PilotRun, "--direct-run", $DirectRun, "--judge-model-key", $JudgeModelKey, "--max-workers", "$MaxWorkers", "extract-expert-af")
Invoke-Checked $Py @("-m", $Module, "--run-name", $RunName, "--atlas-run", $AtlasRun, "--pilot-run", $PilotRun, "--direct-run", $DirectRun, "--judge-model-key", $JudgeModelKey, "--max-workers", "$MaxWorkers", "canonicalize-reference")

foreach ($ModelKey in $ModelKeys) {
  Invoke-Checked $Py @("-m", $Module, "--run-name", $RunName, "--atlas-run", $AtlasRun, "--pilot-run", $PilotRun, "--direct-run", $DirectRun, "--model-key", $ModelKey, "--judge-model-key", $JudgeModelKey, "--max-workers", "$MaxWorkers", "judge-recall")
}

foreach ($ModelKey in $ModelKeys) {
  foreach ($Variant in $EvaluateVariants) {
    Invoke-Checked $Py @("-m", $Module, "--run-name", $RunName, "--atlas-run", $AtlasRun, "--pilot-run", $PilotRun, "--direct-run", $DirectRun, "--model-key", $ModelKey, "export-legacy-eval", "--variant", $Variant)
    $LegacyRun = (& $Py @("-m", $Module, "--run-name", $RunName, "--model-key", $ModelKey, "legacy-eval-name", "--variant", $Variant)).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $LegacyRun) {
      throw "Could not resolve legacy eval run name for $ModelKey $Variant"
    }
    Invoke-Checked $EvalPy @("-m", "src.thesis_laysumm.run_evaluate", "--run-name", $LegacyRun, "--model-key", $ModelKey, "--variant", "generated")
    Invoke-Checked $Py @("-m", $Module, "--run-name", $RunName, "--atlas-run", $AtlasRun, "--pilot-run", $PilotRun, "--direct-run", $DirectRun, "--model-key", $ModelKey, "import-legacy-eval", "--variant", $Variant)
  }
}

Invoke-Checked $Py (@("-m", $Module, "--run-name", $RunName, "--atlas-run", $AtlasRun, "--pilot-run", $PilotRun, "--direct-run", $DirectRun, "summarize", "--model-keys") + $ModelKeys)

Write-Host ""
Write-Host "Done."
Write-Host "Data:    data\laysumm_ablation_m1_recall\runs\$RunName"
Write-Host "Results: results\laysumm_ablation_m1_recall\runs\$RunName"
