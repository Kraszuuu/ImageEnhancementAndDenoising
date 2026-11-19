param(
    [int]$Port = 8501,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$App = Join-Path $Root "src\lowlight\demo\app.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python not found at $Python. Create .venv and install requirements first."
}
if (-not (Test-Path -LiteralPath $App)) {
    throw "Demo app not found at $App."
}

$env:PYTHONPATH = Join-Path $Root "src"
$Arguments = @(
    "-m", "streamlit", "run", $App,
    "--server.address", "localhost",
    "--server.port", "$Port",
    "--server.runOnSave", "false"
)
if ($Headless) {
    $Arguments += "--server.headless", "true"
}

Push-Location $Root
try {
    & $Python @Arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
