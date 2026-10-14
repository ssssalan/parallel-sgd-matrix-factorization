# Publish this cleaned folder into the existing GitHub repository.
# Run from PowerShell after extracting the ZIP. Requires Git and GitHub authentication.

$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/ssssalan/parallel-sgd-matrix-factorization.git"
$Source = Split-Path -Parent $MyInvocation.MyCommand.Path
$Temp = Join-Path $env:TEMP "parallel-sgd-matrix-factorization-publish"

if (Test-Path $Temp) { Remove-Item $Temp -Recurse -Force }

git clone $RepoUrl $Temp

# Copy the cleaned public-ready files over the existing README/LICENSE.
Get-ChildItem -Path $Source -Force | Where-Object { $_.Name -notin @('.git') } | ForEach-Object {
    Copy-Item $_.FullName -Destination $Temp -Recurse -Force
}

Set-Location $Temp

git add -A
$changes = git status --porcelain
if (-not $changes) {
    Write-Host "No changes to publish."
    exit 0
}

git commit -m "Refactor parallel SGD project for public release"
git push origin main

Write-Host "Published to: https://github.com/ssssalan/parallel-sgd-matrix-factorization"
