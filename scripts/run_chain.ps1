# Sequentially run experiments across BOTH arms, one at a time, in a given order.
#
# Generalises run_l3_chain.ps1, which hardcoded one config for every task. Each entry
# here carries its own config, so a gpt-5.6-sol task and a glm-5.3 task can share one
# queue: "run GLM first, then the remaining gpt tasks" is just an ordering.
#
# WHY STRICT SEQUENCING IS MANDATORY, not merely tidy:
# timing jobs take the GPU exclusively via GpuRwLock, whose lock file is
# `<runs_dir>/jobs/gpu.lock` (gpu/worker_client.py:115). The two arms have DIFFERENT
# runs_dir values, so they take DIFFERENT lock files and would NOT see each other --
# two concurrent runs would interleave timing on one GPU and silently corrupt every
# latency measurement in both. Nothing in the harness prevents this; only this queue does.
#
# Also verified by experiment before writing this: killing the PowerShell that owns a
# `... | Tee-Object` pipeline KILLS the child run (the child froze mid-write and exited).
# So -WaitForPid must wait for the run's own process, and a previous chain script must
# never be killed while its child is still running.
param(
  # Wait for this pid to exit before starting anything. Use the pid of the RUN process
  # (kernel-opt / python), not of a wrapper shell.
  [int]$WaitForPid = 0,
  # Each entry: @{ Task = "level3:21"; Config = "configs/experiments_l3.yaml"; Label = "gpt" }
  [hashtable[]]$Jobs = @(),
  [string]$LogDir = "D:\ClaudeCode\tmp"
)

$ErrorActionPreference = "Continue"
Set-Location "D:\Pyhon_projects\opop\v2"

if ($Jobs.Count -eq 0) {
  Write-Output "[chain] no jobs given; nothing to do"
  exit 0
}

Write-Output "[chain] queue:"
foreach ($j in $Jobs) {
  Write-Output ("[chain]   {0,-12} {1,-6} {2}" -f $j.Task, $j.Label, $j.Config)
}

if ($WaitForPid -gt 0) {
  Write-Output "[chain] waiting for pid $WaitForPid (the in-flight run) to exit..."
  while ($true) {
    $p = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    if ($null -eq $p) { break }
    Start-Sleep -Seconds 20
  }
  Write-Output "[chain] pid $WaitForPid gone; GPU free"
  Start-Sleep -Seconds 30   # let the WSL worker / file lock settle
}

foreach ($j in $Jobs) {
  $task = $j.Task
  $cfg = $j.Config
  $label = if ($j.Label) { $j.Label } else { "run" }
  if (-not (Test-Path $cfg)) {
    Write-Output "[chain] !! config missing, SKIPPING $task : $cfg"
    continue
  }
  $slug = $task -replace "[^A-Za-z0-9]", "-"
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $log = Join-Path $LogDir "chain-$label-$slug-$stamp.log"
  Write-Output "[chain] === starting $label $task ($cfg) -> $log ==="
  & uv run kernel-opt --config $cfg run --task $task *>&1 | Tee-Object -FilePath $log
  Write-Output "[chain] === finished $label $task (exit=$LASTEXITCODE) ==="
  Start-Sleep -Seconds 30
}
Write-Output "[chain] all jobs done"
