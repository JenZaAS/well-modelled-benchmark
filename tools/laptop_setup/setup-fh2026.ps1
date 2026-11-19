<#
.SYNOPSIS
  FH2026 (FORCE Hackathon 2026) laptop setup: Python env, packages, data, model weights, smoke test.

.DESCRIPTION
  Creates everything under one cache folder outside any git repo (default: %USERPROFILE%\fh2026-cache):
    venv\              Python 3.12 virtual environment (CPU-only torch)
    pip-cache\         pip wheel cache (lets you rebuild the venv offline)
    data\force2020\    FORCE 2020 lithology data (GitHub CSVs + Zenodo 4351156 LAS zip, NPD tables)
    data\sodir_factpages\  Sodir FactPages CSV snapshot (only with -WithSodir)
    docs\              offline docs: DuckDB, lasio, pdfplumber, MOMENT/Chronos/TTM READMEs, FORCE retrospective, NLOD
    hf\                Hugging Face cache (HF_HOME): MOMENT-1-small, Chronos-Bolt tiny/small, Chronos-2, TTM r2
    fh2026-env.ps1     helper that activates the venv and sets HF_HOME / offline mode
  Safe to re-run: installed packages and files with the right checksum are skipped.
  No admin rights needed. Nothing is uploaded anywhere.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\setup-fh2026.ps1 -WithSodir
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\setup-fh2026.ps1 -SmokeOnly
#>
[CmdletBinding()]
param(
    # Where the venv, data and model weights go. Must be outside any git repo.
    [string]$CacheRoot = (Join-Path $env:USERPROFILE 'fh2026-cache'),
    # Full path to a python.exe to use instead of auto-detecting (3.11-3.13).
    [string]$Python = '',
    [switch]$SkipInstall,   # don't create/update the venv
    [switch]$SkipData,      # don't download FORCE 2020 data
    [switch]$SkipModels,    # don't download model weights (and skip model smoke tests)
    [switch]$SmokeOnly,     # only run the smoke test
    [switch]$SkipSmoke,     # don't run the smoke test
    [switch]$SkipDocs,      # don't download offline docs (~20 MB)
    [switch]$WithSodir,     # also download the Sodir FactPages CSV snapshot (~10 MB)
    [switch]$RefreshSodir,  # re-download the Sodir snapshot even if present
    [switch]$Force          # continue even if free disk space looks too low
)

$ErrorActionPreference = 'Stop'
$ScriptDir = $PSScriptRoot
$TorchSpec = 'torch==2.11.*'                       # granite-tsfm 0.3.12 needs torch >=2.10,<2.12
$TorchIndex = 'https://download.pytorch.org/whl/cpu'
$MomentSpec = 'momentfm==0.1.4'                    # installed with --no-deps (old pins)
$Stages = New-Object System.Collections.ArrayList

function Write-Step([string]$Text) { Write-Host ''; Write-Host "=== $Text ===" -ForegroundColor Cyan }
function Add-Stage([string]$Name, [bool]$Ok, [string]$Note = '') {
    [void]$Stages.Add([pscustomobject]@{ Stage = $Name; Result = $(if ($Ok) { 'PASS' } else { 'FAIL' }); Note = $Note })
    if ($Ok) { Write-Host "[PASS] $Name $Note" -ForegroundColor Green }
    else { Write-Host "[FAIL] $Name $Note" -ForegroundColor Red }
}
function Show-Summary {
    Write-Host ''
    Write-Host '================ FH2026 SETUP SUMMARY ================' -ForegroundColor Cyan
    $Stages | Format-Table -AutoSize | Out-String | Write-Host
    $failed = @($Stages | Where-Object { $_.Result -eq 'FAIL' }).Count
    if ($failed -eq 0) { Write-Host 'ALL STAGES PASSED' -ForegroundColor Green }
    else { Write-Host "$failed STAGE(S) FAILED - scroll up for details, fix, then re-run (it resumes)." -ForegroundColor Red }
    return $failed
}

if ($SmokeOnly) { $SkipInstall = $true; $SkipData = $true; $SkipModels = $true; $SkipDocs = $true }

# ---------------------------------------------------------------- 0. Paths and disk space
Write-Step 'FH2026 laptop setup'
$CacheRoot = [System.IO.Path]::GetFullPath($CacheRoot)
$VenvDir = Join-Path $CacheRoot 'venv'
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'
Write-Host "Cache root : $CacheRoot"
Write-Host "Script dir : $ScriptDir"

# Refuse to put data/weights inside a git repo (they must never be committed).
$probe = $CacheRoot
while ($probe) {
    if (Test-Path (Join-Path $probe '.git')) {
        Write-Host "Cache root is inside a git repo ($probe). Pick a folder outside any repo with -CacheRoot." -ForegroundColor Red
        exit 2
    }
    $parent = Split-Path $probe -Parent
    if ($parent -eq $probe) { break }
    $probe = $parent
}

# Estimated space in MB (measured 2026-11-17; Windows torch CPU wheel is ~115 MB download).
$need = [ordered]@{}
if (-not $SkipInstall) { $need['Python venv (torch CPU, transformers, xgboost, duckdb, ...)'] = 1550; $need['pip wheel cache'] = 600 }
if (-not $SkipData) { $need['FORCE 2020 data (330 MB download + 280 MB unzipped train.csv)'] = 620 }
if ($WithSodir -and -not $SmokeOnly) { $need['Sodir FactPages CSV snapshot'] = 15 }
if (-not $SkipDocs) { $need['Offline docs (DuckDB PDF+MD, READMEs, notebooks, licences)'] = 20 }
if (-not $SkipModels) { $need['Model weights (MOMENT-small 303, Chronos-2 478, Bolt-small 191, Bolt-tiny 35, TTM 4)'] = 1020 }
$needMB = 0; foreach ($v in $need.Values) { $needMB += $v }

if ($needMB -gt 0) {
    Write-Host ''
    Write-Host 'Estimated disk space needed (first run; re-runs need almost nothing):'
    foreach ($k in $need.Keys) { Write-Host ('  {0,6} MB  {1}' -f $need[$k], $k) }
    Write-Host ('  {0,6} MB  TOTAL (~{1:N1} GB). Keep 2 GB extra as margin.' -f $needMB, ($needMB / 1024))
    try {
        $drive = New-Object System.IO.DriveInfo ([System.IO.Path]::GetPathRoot($CacheRoot))
        $freeMB = [math]::Floor($drive.AvailableFreeSpace / 1MB)
        Write-Host ('Free on {0}: {1:N0} MB' -f $drive.Name, $freeMB)
        if ($freeMB -lt ($needMB + 2048)) {
            if ($Force) { Write-Host 'Low disk space, continuing because of -Force.' -ForegroundColor Yellow }
            else { Write-Host 'Not enough free space (need total + 2 GB margin). Free some space, use another -CacheRoot, or add -Force.' -ForegroundColor Red; exit 2 }
        }
    } catch { Write-Host "Could not read free space ($($_.Exception.Message)); continuing." -ForegroundColor Yellow }
}
New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null

# Environment for pip and Hugging Face: everything stays under the cache root.
$env:PIP_CACHE_DIR = Join-Path $CacheRoot 'pip-cache'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:HF_HOME = Join-Path $CacheRoot 'hf'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

# ---------------------------------------------------------------- 1. Python and venv
function Get-PyVersion([string[]]$Cmd) {
    # Returns "3.12.x|C:\path\python.exe" or $null. Ignores the Microsoft Store stub.
    try {
        $exe = $Cmd[0]; $rest = @(); if ($Cmd.Count -gt 1) { $rest = $Cmd[1..($Cmd.Count - 1)] }
        $out = & $exe @rest -c "import sys; print('%d.%d.%d|%s' % (sys.version_info[:3] + (sys.executable,)))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out -match '^3\.\d+\.\d+\|') { return ($out | Select-Object -First 1) }
    } catch { }
    return $null
}

if (-not $SkipInstall) {
    Write-Step '1. Python 3.12 and virtual environment'
    $basePy = $null
    if ($Python) {
        $info = Get-PyVersion @($Python)
        if (-not $info) { Write-Host "Cannot run -Python '$Python'." -ForegroundColor Red; exit 2 }
        $basePy = $info
    } else {
        # 3.12 first (best wheel coverage), then 3.11, then 3.13. 3.14 is too new for granite-tsfm (<3.14).
        $candidates = @(@('py', '-3.12'), @('py', '-3.11'), @('py', '-3.13'), @('python'), @('python3'))
        foreach ($c in $candidates) {
            $info = Get-PyVersion $c
            if ($info -and $info -match '^3\.(11|12|13)\.') { $basePy = $info; break }
        }
    }
    if (-not $basePy) {
        Write-Host 'No suitable Python found (need 3.11, 3.12 or 3.13; 3.12 preferred).' -ForegroundColor Red
        Write-Host 'Python 3.14 (e.g. C:\Python314) is too new for granite-tsfm/TTM, and 3.10 is too old for the current wheels.'
        Write-Host 'Install Python 3.12 for your user only (no admin needed), then re-run this script:'
        Write-Host '  winget install --id Python.Python.3.12 --scope user'
        Write-Host '  or https://www.python.org/downloads/windows/ -> 3.12 installer, untick "for all users".'
        Write-Host 'Or point at one directly:  -Python "C:\path\to\python.exe"'
        Add-Stage 'Python' $false 'not found'
        [void](Show-Summary); exit 2
    }
    $pyVer, $pyExe = $basePy -split '\|', 2
    if ($pyVer -notmatch '^3\.12\.') { Write-Host "Using Python $pyVer (3.12 preferred, but this works)." -ForegroundColor Yellow }
    Write-Host "Base Python: $pyVer at $pyExe"

    if (Test-Path $VenvPy) {
        $venvInfo = Get-PyVersion @($VenvPy)
        Write-Host "venv already exists: $VenvDir ($(($venvInfo -split '\|')[0]))"
    } else {
        Write-Host "Creating venv: $VenvDir"
        & $pyExe -m venv $VenvDir
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPy)) { Add-Stage 'venv' $false 'python -m venv failed'; [void](Show-Summary); exit 1 }
    }
    Add-Stage 'Python + venv' $true "($pyVer)"

    Write-Step '2. Packages (pip; already-installed ones are skipped)'
    $ok = $true
    & $VenvPy -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { $ok = $false }
    if ($ok) {
        Write-Host "-- torch (CPU only) from $TorchIndex"
        & $VenvPy -m pip install $TorchSpec --index-url $TorchIndex
        if ($LASTEXITCODE -ne 0) { $ok = $false; Write-Host 'torch install failed' -ForegroundColor Red }
    }
    if ($ok) {
        Write-Host '-- requirements.txt'
        & $VenvPy -m pip install -r (Join-Path $ScriptDir 'requirements.txt')
        if ($LASTEXITCODE -ne 0) { $ok = $false; Write-Host 'requirements install failed' -ForegroundColor Red }
    }
    if ($ok) {
        Write-Host "-- $MomentSpec with --no-deps (its own pins: numpy 1.25, transformers 4.33 - too old for Python 3.12)"
        & $VenvPy -m pip install $MomentSpec --no-deps
        if ($LASTEXITCODE -ne 0) { $ok = $false; Write-Host 'momentfm install failed' -ForegroundColor Red }
    }
    if ($ok) {
        # Make sure nothing swapped torch for a CUDA build or a version granite-tsfm can't use.
        $tv = & $VenvPy -c "import torch; print(torch.__version__)"
        Write-Host "torch in venv: $tv"
        & $VenvPy -m pip check
        if ($LASTEXITCODE -ne 0) { Write-Host 'pip check reports conflicts above. momentfm pins are expected and harmless; anything else, look at it.' -ForegroundColor Yellow }
        & $VenvPy -m pip freeze | Out-File -Encoding utf8 (Join-Path $CacheRoot 'pip-freeze.txt')
    }
    Add-Stage 'Packages' $ok
    if (-not $ok) { [void](Show-Summary); exit 1 }
}

if (-not (Test-Path $VenvPy)) {
    Write-Host "No venv at $VenvDir. Run once without -SkipInstall / -SmokeOnly." -ForegroundColor Red
    exit 2
}

# Helper for later sessions (e.g. at the hackathon, offline).
$envHelper = Join-Path $CacheRoot 'fh2026-env.ps1'
@"
# Dot-source this in a PowerShell window:  . '$envHelper'
`$env:HF_HOME = '$($env:HF_HOME)'
`$env:HF_HUB_OFFLINE = '1'          # use cached weights only; set to '0' to allow downloads
`$env:TRANSFORMERS_OFFLINE = '1'
`$env:HF_HUB_DISABLE_TELEMETRY = '1'
`$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
`$env:PIP_CACHE_DIR = '$($env:PIP_CACHE_DIR)'
`$env:FH2026_DATA = '$(Join-Path $CacheRoot 'data')'
. '$(Join-Path $VenvDir 'Scripts\Activate.ps1')'
Write-Host 'FH2026 env active. Data: ' `$env:FH2026_DATA
"@ | Out-File -Encoding utf8 $envHelper

# ---------------------------------------------------------------- 3. Data and models
$fetchArgs = @()
if (-not $SkipData) { $fetchArgs += '--data' }
if ($WithSodir -and -not $SmokeOnly) { $fetchArgs += '--sodir' }
if ($RefreshSodir) { $fetchArgs += '--refresh-sodir' }
if (-not $SkipDocs) { $fetchArgs += '--docs' }
if (-not $SkipModels) { $fetchArgs += '--models' }
if ($fetchArgs.Count -gt 0) {
    Write-Step '3. Downloads (data / Sodir / docs / model weights)'
    & $VenvPy (Join-Path $ScriptDir 'fh2026_fetch.py') --cache-root $CacheRoot @fetchArgs
    Add-Stage 'Downloads' ($LASTEXITCODE -eq 0) ("(" + ($fetchArgs -join ' ') + ")")
}

# ---------------------------------------------------------------- 4. Smoke test (offline)
if (-not $SkipSmoke) {
    Write-Step '4. Smoke test (runs offline against the cache)'
    $smokeArgs = @()
    $hasModels = Test-Path (Join-Path $env:HF_HOME 'hub\models--AutonLab--MOMENT-1-small')
    $hasData = Test-Path (Join-Path $CacheRoot 'data\force2020\penalty_matrix.npy')
    if (-not $hasModels) { $smokeArgs += '--skip-models'; Write-Host 'No model weights in cache yet: skipping model tests.' -ForegroundColor Yellow }
    if (-not $hasData) { $smokeArgs += '--skip-data'; Write-Host 'No FORCE data in cache yet: skipping data tests.' -ForegroundColor Yellow }
    & $VenvPy (Join-Path $ScriptDir 'fh2026_smoke_test.py') --cache-root $CacheRoot @smokeArgs
    Add-Stage 'Smoke test' ($LASTEXITCODE -eq 0) $(if ($smokeArgs) { '(partial: ' + ($smokeArgs -join ' ') + ')' } else { '' })
}

$failed = Show-Summary
Write-Host ''
Write-Host "Later, in a new PowerShell window:  Set-ExecutionPolicy -Scope Process Bypass; . '$envHelper'"
if ($failed -gt 0) { exit 1 } else { exit 0 }
