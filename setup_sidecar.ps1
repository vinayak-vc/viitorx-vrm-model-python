<#
.SYNOPSIS
    One-time setup for the Virtual Mirror Python tracking sidecar.

.DESCRIPTION
    Creates the virtualenv, installs the pinned dependencies, and VERIFIES that GPU inference is
    actually available before declaring success.

    That last step is the point of this script. The sidecar asks onnxruntime for DirectML and
    onnxruntime falls back to CPU silently if the provider cannot load. Correctness is unaffected,
    so the only symptom is that the whole app runs at roughly 5 fps instead of ~32 and every timing
    number is wrong. That failure has cost this project days. It is far cheaper to detect here, at
    install time, than to debug later as "the mirror feels laggy".

    Run this once per machine. It is safe to re-run; use -Force to rebuild the venv from scratch.

.PARAMETER Force
    Delete and recreate the virtualenv even if one already exists.

.PARAMETER PythonExe
    Path to a Python 3.10 interpreter. Autodetected when omitted.

.PARAMETER ModelPath
    Path to rtmw3d-x.onnx. When given, it is copied into models/ so the sidecar can find it.

.EXAMPLE
    .\setup_sidecar.ps1

.EXAMPLE
    .\setup_sidecar.ps1 -ModelPath "D:\models\rtmw3d-x.onnx" -Force

.NOTES
    Windows only. Requires Python 3.10 (the depthai wheels are cp310-only) and an OAK-D camera at
    runtime. Does not need administrator rights.
#>
[CmdletBinding()]
param(
    [switch]$Force,
    [string]$PythonExe = "",
    [string]$ModelPath = ""
)

$ErrorActionPreference = "Stop"

$SidecarRoot = $PSScriptRoot
$VenvDir     = Join-Path $SidecarRoot ".venv"
$VenvPython  = Join-Path $VenvDir "Scripts\python.exe"
$LockFile    = Join-Path $SidecarRoot "requirements.lock.txt"
$ModelDir    = Join-Path $SidecarRoot "models"
$ModelTarget = Join-Path $ModelDir "rtmw3d-x.onnx"

function Write-Step { param([string]$Message) Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Message) Write-Host "    OK  $Message" -ForegroundColor Green }
function Write-Warn { param([string]$Message) Write-Host "    !   $Message" -ForegroundColor Yellow }
function Fail       { param([string]$Message) Write-Host "`nFAILED: $Message" -ForegroundColor Red; exit 1 }

Write-Host "Virtual Mirror - sidecar setup" -ForegroundColor White
Write-Host "Sidecar root: $SidecarRoot"

# ---- 1. locate a Python 3.10 interpreter ------------------------------------------------------
# 3.10 exactly: depthai publishes cp310 wheels, so 3.11+ fails to resolve depthai at all and the
# error it produces ("no matching distribution") does not mention the version requirement.
Write-Step "Locating Python 3.10"
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $candidates = @()
    # The py launcher is the most reliable way to ask for a specific version on Windows.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $pyPath = (& py -3.10 -c "import sys; print(sys.executable)" 2>$null)
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($pyPath)) {
                $candidates += $pyPath.Trim()
            }
        } catch {
            # py launcher present but has no 3.10 registered; fall through to the path probes.
        }
    }
    $candidates += "C:\Program Files\Python310\python.exe"
    $candidates += "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { $PythonExe = $candidate; break }
    }
}

if ([string]::IsNullOrWhiteSpace($PythonExe) -or -not (Test-Path $PythonExe)) {
    Fail @"
Python 3.10 was not found.

Install it from https://www.python.org/downloads/release/python-3100/
(tick "Add python.exe to PATH"), then re-run this script. If it is installed somewhere
unusual, pass it explicitly:

    .\setup_sidecar.ps1 -PythonExe "C:\path\to\python.exe"
"@
}

$versionOutput = (& $PythonExe -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
if ($versionOutput -ne "3.10") {
    Fail "Found Python $versionOutput at $PythonExe, but 3.10 is required (depthai ships cp310 wheels only)."
}
Write-Ok "Python 3.10 at $PythonExe"

# ---- 2. create the virtualenv -----------------------------------------------------------------
Write-Step "Creating virtualenv"
if ((Test-Path $VenvDir) -and $Force) {
    Write-Warn "removing existing venv (-Force)"
    Remove-Item -Recurse -Force $VenvDir
}
if (Test-Path $VenvPython) {
    Write-Ok "venv already exists (use -Force to rebuild)"
} else {
    & $PythonExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Fail "python -m venv failed with exit code $LASTEXITCODE." }
    if (-not (Test-Path $VenvPython)) { Fail "venv was created but $VenvPython is missing." }
    Write-Ok "created $VenvDir"
}

# ---- 3. install pinned dependencies ------------------------------------------------------------
Write-Step "Installing dependencies"
if (-not (Test-Path $LockFile)) { Fail "requirements.lock.txt is missing at $LockFile." }

& $VenvPython -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Write-Warn "pip self-upgrade failed; continuing with the bundled pip." }

& $VenvPython -m pip install -r $LockFile
if ($LASTEXITCODE -ne 0) { Fail "pip install failed with exit code $LASTEXITCODE." }
Write-Ok "dependencies installed from requirements.lock.txt"

# ---- 4. verify the imports the supervisor probes for -------------------------------------------
Write-Step "Verifying imports"
& $VenvPython -c "import depthai, cv2, numpy, onnxruntime"
if ($LASTEXITCODE -ne 0) { Fail "importing depthai/cv2/numpy/onnxruntime failed. The venv is not usable." }
Write-Ok "depthai, cv2, numpy, onnxruntime all import"

# ---- 5. VERIFY GPU EXECUTION PROVIDER (the expensive-to-miss check) ----------------------------
Write-Step "Verifying DirectML GPU provider"
$providers = (& $VenvPython -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))").Trim()
Write-Host "    providers: $providers"
if ($providers -notmatch "DmlExecutionProvider") {
    Fail @"
onnxruntime has NO DirectML provider. Available: $providers

The sidecar will still run and produce correct poses, but on CPU: roughly 183 ms/frame
(~5 fps) instead of ~31 ms/frame (~32 fps). Nothing will report an error — the app will
just feel broken.

Usual cause: plain 'onnxruntime' got installed over 'onnxruntime-directml'. Fix with:

    .venv\Scripts\python.exe -m pip uninstall -y onnxruntime onnxruntime-directml
    .venv\Scripts\python.exe -m pip install -r requirements.lock.txt
"@
}
Write-Ok "DmlExecutionProvider present - GPU inference available"

# ---- 6. model file -----------------------------------------------------------------------------
Write-Step "Checking the pose model"
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
    if (-not (Test-Path $ModelPath)) { Fail "-ModelPath was given but $ModelPath does not exist." }
    Copy-Item -Path $ModelPath -Destination $ModelTarget -Force
    Write-Ok "copied model to $ModelTarget"
} elseif (Test-Path $ModelTarget) {
    Write-Ok "model already present at $ModelTarget"
} else {
    Write-Warn @"
rtmw3d-x.onnx (369 MB) is not installed.

It is not bundled with the build. Copy it into:
    $ModelDir
or re-run with:
    .\setup_sidecar.ps1 -ModelPath "path\to\rtmw3d-x.onnx"

The app will start without it, but tracking will not run and the HUD will say so.
"@
}

# ---- done ---------------------------------------------------------------------------------------
Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Interpreter: $VenvPython"
Write-Host "Unity starts the sidecar automatically. To run it by hand instead:"
Write-Host "    .venv\Scripts\python.exe sidecar_supervisor.py --model models\rtmw3d-x.onnx --portrait --portrait-dir ccw --subpixel-bits 3 --allow-port-listener"
