param(
  [string]$Message = ""
)

if ($Message -eq "") {
  $Message = "Update " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
}

if (-not $env:HF_TOKEN -or $env:HF_TOKEN.Trim().Length -lt 10) {
  Write-Host "HF_TOKEN is missing. Set it with:"
  Write-Host '[Environment]::SetEnvironmentVariable("HF_TOKEN","YOUR_TOKEN","User")'
  exit 1
}

# Ensure we are in the repo root
$repoRoot = git rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Not a git repository. Run this inside your cloned Space folder."
  exit 1
}
Set-Location $repoRoot

# Stage everything
git add -A

# If nothing changed, exit cleanly
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
  Write-Host "No changes to commit."
  exit 0
}

# Commit
git commit -m $Message

# Push using token without storing it in git remote
$spaceUrl = "https://mnoorchenar:$($env:HF_TOKEN)@huggingface.co/spaces/mnoorchenar/scopus"
git push $spaceUrl HEAD:main

if ($LASTEXITCODE -eq 0) {
  Write-Host "Pushed to Hugging Face Space: mnoorchenar/scopus"
} else {
  Write-Host "Push failed."
  exit 1
}
