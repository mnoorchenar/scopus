param(
  [string]$Message = ""
)

if ($Message -eq "") {
  $Message = "Update " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
}

# Ensure we are in the repo root
$repoRoot = git rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Not a git repository. Run this inside your cloned repo folder."
  exit 1
}
Set-Location $repoRoot

# Block force push if a merge is in progress
if (Test-Path ".git\MERGE_HEAD") {
  Write-Host "Merge in progress. Resolve conflicts or run: git merge --abort"
  exit 1
}

# Check token
if (-not $env:HF_TOKEN -or $env:HF_TOKEN.Trim().Length -lt 10) {
  Write-Host "HF_TOKEN is missing. Set it with:"
  Write-Host '[Environment]::SetEnvironmentVariable("HF_TOKEN","YOUR_TOKEN","User")'
  exit 1
}

# Ensure GitHub remote exists
git remote get-url github 1>$null 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Missing GitHub remote named 'github'. Add it once with:"
  Write-Host "git remote add github git@github.com:mnoorchenar/scopus.git"
  exit 1
}

# Stage everything
git add -A

# Commit only if needed
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
  git commit -m $Message
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Commit failed."
    exit 1
  }
} else {
  Write-Host "No changes to commit."
}

# Push to Hugging Face (force)
$hfUrl = "https://mnoorchenar:$($env:HF_TOKEN)@huggingface.co/spaces/mnoorchenar/scopus"
git push $hfUrl HEAD:main --force
if ($LASTEXITCODE -ne 0) {
  Write-Host "Hugging Face push failed."
  exit 1
}
Write-Host "Force pushed to Hugging Face Space: mnoorchenar/scopus"

# Push to GitHub (force)
git push github HEAD:main --force
if ($LASTEXITCODE -ne 0) {
  Write-Host "GitHub push failed."
  exit 1
}
Write-Host "Force pushed to GitHub: mnoorchenar/scopus"
