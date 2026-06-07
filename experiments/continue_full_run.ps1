param(
    [Parameter(Mandatory = $true)]
    [int]$TreeProcessId,
    [Parameter(Mandatory = $true)]
    [int]$TsProcessId,
    [string]$PythonExe = "D:\Users\hp\anaconda3\envs\pess-pl-legacy\python.exe",
    [string]$Seed = "20260605",
    [int]$RealJobs = 4,
    [int[]]$ExcludedContextualT = @()
)

$ErrorActionPreference = "Stop"

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class ReproductionPowerState {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$ES_CONTINUOUS = [Convert]::ToUInt32("80000000", 16)
$ES_SYSTEM_REQUIRED = [uint32]0x00000001
[ReproductionPowerState]::SetThreadExecutionState(
    $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED
) | Out-Null

$repo = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repo "artifacts\reproduction\published\full\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$statusPath = Join-Path $logDir "continuation.status.log"
$envRoot = Split-Path -Parent $PythonExe
$env:CONDA_PREFIX = $envRoot
$env:R_HOME = Join-Path $envRoot "Lib\R"
$env:PYTHONPATH = $repo
$env:PATH = (
    "$envRoot;" +
    "$envRoot\Library\mingw-w64\bin;" +
    "$envRoot\Library\usr\bin;" +
    "$envRoot\Library\bin;" +
    "$envRoot\Scripts;" +
    "$envRoot\bin;" +
    "$envRoot\Lib\R\bin\x64;" +
    $env:PATH
)

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
        "experiments\reproduce.py",
        "--mode", "full", "--protocol", "published",
        "--experiment", $Experiment, "--jobs", "$Jobs",
        "--resume", "--seed", $Seed
    )
    if ($Experiment -in @("tree", "ts", "ts-cv") -and $ExcludedContextualT.Count -gt 0) {
        $arguments += "--exclude-contextual-t"
        $arguments += $ExcludedContextualT | ForEach-Object { "$_" }
    }
    Write-Status "starting $Experiment with $Jobs workers"
    $process = Start-Process `
        -FilePath $PythonExe `
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

function Invoke-PythonPhase(
    [string]$Name,
    [string[]]$Arguments
) {
    $stdout = Join-Path $logDir "$Name.out.log"
    $stderr = Join-Path $logDir "$Name.err.log"
    Write-Status "starting $Name"
    $process = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $Arguments `
        -WorkingDirectory $repo `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru `
        -Wait
    Write-Status "finished $Name with exit code $($process.ExitCode)"
    if ($process.ExitCode -ne 0) {
        throw "$Name failed with exit code $($process.ExitCode)"
    }
}

Write-Status "waiting for tree PID $TreeProcessId and TS PID $TsProcessId"
Wait-Process -Id $TreeProcessId, $TsProcessId -ErrorAction SilentlyContinue
Write-Status "tree and TS parent processes finished"

Invoke-ReproductionPhase -Experiment "ts-cv" -Jobs 40
Invoke-ReproductionPhase -Experiment "real" -Jobs $RealJobs
Invoke-PythonPhase -Name "analysis" -Arguments @(
    "experiments\analyze_full_reproduction.py",
    "--root", "artifacts\reproduction\published\full",
    "--reference-root", "artifacts\reproduction\paper_reference",
    "--alpha", "0.05"
)
Invoke-PythonPhase -Name "report" -Arguments @(
    "experiments\write_reproduction_report_zh.py",
    "--root", "artifacts\reproduction\published\full",
    "--reference-root", "artifacts\reproduction\paper_reference",
    "--output", "artifacts\reproduction\reproduction_report_zh.md"
)
Invoke-PythonPhase -Name "tests" -Arguments @(
    "-m", "pytest", "-q", "tests\test_protocols.py"
)
Write-Status "all full reproduction, analysis, report, and test phases finished"
