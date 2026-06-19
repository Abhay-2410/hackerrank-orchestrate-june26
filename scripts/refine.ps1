# Full refinement workflow: sample eval → test predictions → package code.zip
# Add ANTHROPIC_API_KEY to .env first for best accuracy (optional).
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "Installing dependencies..."
pip install -r code/requirements.txt -q

Write-Host "`n=== Sample evaluation (20 claims) ==="
python code/evaluation/main.py

Write-Host "`n=== Test predictions (44 claims) ==="
python code/main.py

Write-Host "`n=== Re-generating evaluation report with test stats ==="
python code/evaluation/main.py

Write-Host "`n=== Packaging code.zip ==="
& (Join-Path $PSScriptRoot "package.ps1")

Write-Host "`nDone. Submit: code.zip, output.csv, and log.txt"
