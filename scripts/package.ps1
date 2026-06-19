# Package code/ into code.zip for submission (excludes __pycache__).
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$staging = Join-Path $env:TEMP "hackerrank_code_zip_$(Get-Random)"
$zipPath = Join-Path $repoRoot "code.zip"

Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $staging | Out-Null
Copy-Item -Recurse (Join-Path $repoRoot "code") (Join-Path $staging "code")

Get-ChildItem $staging -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force

if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path (Join-Path $staging "code") -DestinationPath $zipPath -Force
Remove-Item -Recurse -Force $staging

Write-Host "Created $zipPath"
