param(
    [Parameter(Mandatory = $true)]
    [int]$TreeProcessId,
    [Parameter(Mandatory = $true)]
    [int]$TsProcessId,
    [string]$CondaExe = "D:\Users\hp\anaconda3\Scripts\conda.exe",
    [string]$Seed = "20260605"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repo "artifacts\reproduction\published\full\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$statusPath = Join-Path $logDir "continuation.status.log"

function Write-Status([string]$Message) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    Add-Content -LiteralPath $statusPath -Value "$timestamp $Message"
}

function Invoke-ReproductionPhase(
    [string]$Experiment,
    [int]$Jobs
) {
    $stdout = Join-Path $logDir "$Experiment.out.log"
    $stderr = Join-Path $logDir "$Experiment.err.log"
    $arguments = @(
        "run", "-n", "pess-pl-legacy", "python", "experiments\reproduce.py",
        "--mode", "full", "--protocol", "published",
        "--experiment", $Experiment, "--jobs", "$Jobs",
        "--resume", "--seed", $Seed
    )
    Write-Status "starting $Experiment with $Jobs workers"
    $process = Start-Process `
        -FilePath $CondaExe `
        -ArgumentList $arguments `
        -WorkingDirectory $repo `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru `
        -Wait
    Write-Status "finished $Experiment with exit code $($process.ExitCode)"
    if ($process.ExitCode -ne 0) {
        throw "$Experiment failed with exit code $($process.ExitCode)"
    }
}

Write-Status "waiting for tree PID $TreeProcessId and TS PID $TsProcessId"
Wait-Process -Id $TreeProcessId, $TsProcessId -ErrorAction SilentlyContinue
Write-Status "tree and TS parent processes finished"

Invoke-ReproductionPhase -Experiment "ts-cv" -Jobs 20
Invoke-ReproductionPhase -Experiment "real" -Jobs 2
Write-Status "all scheduled full reproduction phases finished"
