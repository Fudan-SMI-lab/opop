# Sequentially run the remaining L3 experiments after the currently-running one exits.
# Timing jobs take the GPU exclusively, so runs must never overlap.
param(
  [int]$WaitForPid = 0,
  [string[]]$Tasks = @("level3:43", "level3:48"),
  [string]$LogDir = "D:\ClaudeCode\tmp"
)

$ErrorActionPreference = "Continue"
Set-Location "D:\Pyhon_projects\opop\v2"

if ($WaitForPid -gt 0) {
  Write-Output "[chain] waiting for pid $WaitForPid to exit..."
  while ($true) {
    $p = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    if ($null -eq $p) { break }
    Start-Sleep -Seconds 20
  }
  Write-Output "[chain] pid $WaitForPid gone; GPU free"
  Start-Sleep -Seconds 30   # let the WSL worker / file lock settle
}

foreach ($t in $Tasks) {
  $slug = $t -replace "[^A-Za-z0-9]", "-"
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $log = Join-Path $LogDir "chain-$slug-$stamp.log"
  Write-Output "[chain] === starting $t -> $log ==="
  & uv run kernel-opt --config configs/experiments_l3.yaml run --task $t *>&1 |
    Tee-Object -FilePath $log
  Write-Output "[chain] === finished $t (exit=$LASTEXITCODE) ==="
  Start-Sleep -Seconds 30
}
Write-Output "[chain] all tasks done"
