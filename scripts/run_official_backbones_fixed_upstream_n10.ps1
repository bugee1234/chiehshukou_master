param(
    [ValidateSet("both", "qwen", "llama")]
    [string]$Model = "both",
    [switch]$EvalOnly,
    [switch]$SkipEval,
    [switch]$NoResume
)

& "$PSScriptRoot\run_official_backbones_atlas_pilot20.ps1" `
    -Model $Model `
    -EvalOnly:$EvalOnly `
    -SkipEval:$SkipEval `
    -NoResume:$NoResume
exit $LASTEXITCODE
