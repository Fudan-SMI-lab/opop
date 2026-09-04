# Push v2 to GitHub from WINDOWS, verifying by hash comparison.
#
# Why Windows and not WSL: as of 2026-09-05 WSL cannot reach github.com from this
# machine (curl/git time out after ~135s with "Failed to connect to port 443") while
# Windows git connects in under 2s. The reverse was true earlier in the project, so
# TRY BOTH before concluding the network is down -- scripts/push_v2.sh is the WSL
# equivalent and is still the right tool when WSL works.
#
# Only a matching ls-remote hash counts as proof. Push output does not: a real TLS
# failure can print a normal-looking "old..new HEAD -> v2", and ls-remote can exit 0
# having printed nothing.
#
# Usage:  powershell -File scripts/push_v2.ps1 -Pat <token>
param([Parameter(Mandatory=$true)][string]$Pat)

$ErrorActionPreference = "Continue"
Set-Location "D:\Pyhon_projects\opop\v2"
$env:GIT_TERMINAL_PROMPT = 0
$url = "https://ZihangZ:$Pat@github.com/Fudan-SMI-lab/opop.git"

$local = (& git rev-parse HEAD).Trim()
Write-Output "local HEAD = $local"

for ($i = 1; $i -le 3; $i++) {
  # 2>&1 on a native exe makes PowerShell wrap stderr lines as NativeCommandError and
  # sets $? false even on success, so push output is captured but its exit code is read
  # from $LASTEXITCODE instead.
  $pushOut = (& git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 `
                 push $url HEAD:v2 2>&1 | Out-String)
  $pushRc = $LASTEXITCODE

  $lsOut = (& git ls-remote $url refs/heads/v2 2>&1 | Out-String)
  $lsRc = $LASTEXITCODE
  if ($lsRc -ne 0) {
    Write-Output "attempt ${i}: push_rc=$pushRc but VERIFICATION FAILED TO RUN (ls-remote rc=$lsRc)"
    Write-Output ($lsOut -replace [regex]::Escape($Pat), "<redacted>")
    Write-Output "PUSH_UNVERIFIED: remote state unknown -- do NOT report this as pushed"
    exit 3
  }

  $m = [regex]::Match($lsOut, "(?m)^([0-9a-f]{40})\s")
  if (-not $m.Success) {
    # ls-remote exited 0 having printed no ref: not a comparison, not evidence.
    Write-Output "attempt ${i}: ls-remote exited 0 but returned no ref for v2 -- retrying"
    if ($i -eq 3) {
      Write-Output "PUSH_UNVERIFIED: could not read refs/heads/v2 after 3 attempts"
      exit 3
    }
    Start-Sleep -Seconds 5
    continue
  }

  $remote = $m.Groups[1].Value
  Write-Output "attempt ${i}: push_rc=$pushRc remote=$remote local=$local"
  if ($remote -eq $local) {
    Write-Output "PUSH_OK: remote matches local at $local"
    exit 0
  }
  Write-Output ($pushOut -replace [regex]::Escape($Pat), "<redacted>")
  Start-Sleep -Seconds 5
}

Write-Output "PUSH_FAILED: remote never matched local $local"
exit 1
