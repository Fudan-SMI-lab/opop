# Hand the queue over from the old chain to the GLM-first chain, safely.
#
# THE PROBLEM THIS SOLVES
# The chain currently armed (round_next.ps1, pid $OldChainPid) is queued 21 -> 43 -> 48 and
# will auto-start level3:43 within ~30 s of the in-flight run exiting. The new order must be
# glm:21 -> gpt:43 -> gpt:48, so the old chain has to be retired.
#
# It cannot simply be killed now. Verified by experiment (D:\ClaudeCode\tmp\pipetest):
# a PowerShell owning a `... | Tee-Object` pipeline is the parent of the run, and killing it
# froze the child mid-write and took it down. The in-flight run has ~4 h invested, so the
# only safe moment to retire the old chain is AFTER its current child has exited on its own.
#
# WHY THE HANDOVER MUST BE ATOMIC
# Both arms take an exclusive GPU lock at `<runs_dir>/jobs/gpu.lock`, and the two arms have
# DIFFERENT runs_dir -- so their locks are different files and would not see each other.
# Two overlapping runs would interleave timing on one GPU and corrupt every latency number
# in both. Therefore: wait for the run to exit, kill the old chain BEFORE it can spawn its
# next child, confirm no run process survives, only then start the new queue.
param(
  [int]$RunPid = 18516,        # the in-flight run's own python process
  [int]$OldChainPid = 10864,   # powershell running round_next.ps1 (queued 21->43->48)
  [string]$NewChain = "D:\ClaudeCode\tmp\round_next_glm_first.ps1",
  [string]$Log = "D:\ClaudeCode\tmp\handover.log"
)

$ErrorActionPreference = "Continue"
function Say($m) {
  $line = "[handover $(Get-Date -Format 'HH:mm:ss')] $m"
  Write-Output $line
  Add-Content -Path $Log -Value $line -Encoding utf8
}

Say "watching run pid $RunPid; will retire chain pid $OldChainPid then start $NewChain"

while ($true) {
  $p = Get-Process -Id $RunPid -ErrorAction SilentlyContinue
  if ($null -eq $p) { break }
  Start-Sleep -Seconds 15
}
Say "run pid $RunPid has exited"

# Retire the old chain immediately, before its 30 s inter-task sleep elapses.
$old = Get-Process -Id $OldChainPid -ErrorAction SilentlyContinue
if ($null -eq $old) {
  Say "old chain pid $OldChainPid already gone"
} else {
  Stop-Process -Id $OldChainPid -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 3
  $still = Get-Process -Id $OldChainPid -ErrorAction SilentlyContinue
  if ($null -eq $still) { Say "old chain pid $OldChainPid retired" }
  else { Say "WARNING old chain pid $OldChainPid still alive after Stop-Process" }
}

# The old chain may already have spawned level3:43 in the gap. Take down any run process
# that is NOT ours, so the GPU is genuinely free before the new queue starts.
#
# This runs strictly BEFORE the new chain is launched, so it cannot reach our own run.
# The `-notmatch 'Get-CimInstance'` guard excludes shells whose command line merely
# CONTAINS this pattern (a diagnostic query matching itself) -- observed live: a
# powershell.exe running exactly this filter showed up in its own results.
Start-Sleep -Seconds 5
$strays = Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -match 'kernel-opt' -and $_.CommandLine -match 'run --task' -and
  $_.CommandLine -notmatch 'Get-CimInstance' -and $_.ProcessId -ne $PID
}
foreach ($s in $strays) {
  Say "killing stray run pid $($s.ProcessId): $($s.CommandLine.Substring(0, [Math]::Min(120, $s.CommandLine.Length)))"
  Stop-Process -Id $s.ProcessId -Force -ErrorAction SilentlyContinue
}
if (-not $strays) { Say "no stray run processes; GPU is free" }

Start-Sleep -Seconds 20   # let the WSL worker and any file lock settle
Say "starting new queue: glm level3:21, then gpt level3:43, gpt level3:48"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $NewChain
Say "new chain exited (code=$LASTEXITCODE)"
