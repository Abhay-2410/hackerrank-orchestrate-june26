# Point git at YOUR fork so `git push` works.
# You cannot push to interviewstreet/hackerrank-orchestrate-june26 (403 — read-only for participants).
#
# Before running:
# 1. Open https://github.com/interviewstreet/hackerrank-orchestrate-june26
# 2. Click Fork → create under your account (e.g. Abhay-2410)
#
# Usage:
#   .\scripts\setup-git-remote.ps1
#   .\scripts\setup-git-remote.ps1 -GitHubUser YourUsername

param(
    [string]$GitHubUser = "Abhay-2410"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$upstream = "https://github.com/interviewstreet/hackerrank-orchestrate-june26.git"
$origin = "https://github.com/$GitHubUser/hackerrank-orchestrate-june26.git"

$remotes = git remote
if ($remotes -contains "upstream") {
    Write-Host "upstream already configured."
} elseif ($remotes -contains "origin") {
    $currentOrigin = (git remote get-url origin)
    if ($currentOrigin -eq $upstream) {
        git remote rename origin upstream
        Write-Host "Renamed origin -> upstream ($upstream)"
    }
}

if ($remotes -contains "origin") {
    git remote set-url origin $origin
    Write-Host "Set origin -> $origin"
} else {
    git remote add origin $origin
    Write-Host "Added origin -> $origin"
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Fork the repo on GitHub if you have not already:"
Write-Host "     https://github.com/interviewstreet/hackerrank-orchestrate-june26/fork"
Write-Host "  2. git push -u origin main"
Write-Host ""
Write-Host "Hackathon submission does NOT require GitHub push."
Write-Host "Submit code.zip, output.csv, and chat transcript on the HackerRank platform."
