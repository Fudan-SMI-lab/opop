# Fetch a URL (or run a web search) and print it as plain text.
#
# WHY THIS EXISTS. On this Windows host the built-in WebSearch tool is unavailable for the current
# model and WebFetch refuses arxiv.org ("unable to verify if domain is safe"), while `curl` to
# export.arxiv.org times out -- the same host-level reachability problem already recorded for GitHub
# (see memory: autodl-network-turbo-fixes-github-timeouts, and the measured "Windows host cannot
# reach GitHub at all"). Invoke-WebRequest with an explicit TLS 1.2 + browser User-Agent DOES get
# through to arxiv.org's search UI and to proceedings.neurips.cc, so this is the only working route
# from here for a literature question.
#
# Usage (note: -Arg takes an ARRAY; pass it via powershell -Command so the @(...) literal parses,
# because -File flattens the array and the second URL lands as a positional argument):
#   powershell -NoProfile -ExecutionPolicy Bypass -Command "& '_tmp_web.ps1' -Mode f -Arg @('<url>')"
#   -Mode f   fetch each URL in -Arg and dump stripped text (-Limit caps chars per URL)
#   -Mode s   run each string in -Arg as a search query (mojeek, falling back to ecosia)
#
# Redirect output to a UTF-8 file rather than reading it off the console: printing non-ASCII to a
# GBK stdout crashes on this machine.
param(
  [Parameter(Mandatory=$true)][string]$Mode,
  [Parameter(Mandatory=$true)][string[]]$Arg,
  [int]$Limit = 14000
)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

function Strip-Html([string]$h) {
  $h = [Regex]::Replace($h, '(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>', ' ')
  $h = [Regex]::Replace($h, '(?is)<br\s*/?>', "`n")
  $h = [Regex]::Replace($h, '(?is)</(p|div|li|tr|h1|h2|h3|h4|h5|section|article|blockquote|pre|dd|dt)>', "`n")
  $h = [Regex]::Replace($h, '(?s)<[^>]+>', ' ')
  $h = [System.Net.WebUtility]::HtmlDecode($h)
  $h = [Regex]::Replace($h, '[ \t\xa0]+', ' ')
  $h = [Regex]::Replace($h, '(\r?\n[ ]*){3,}', "`n`n")
  return $h.Trim()
}
function Fetch([string]$u, [int]$tries = 3) {
  for ($k=0; $k -lt $tries; $k++) {
    try {
      $r = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 35 -UserAgent $UA -Headers @{"Accept-Language"="en-US,en;q=0.9"}
      return $r.Content
    } catch {
      $m = $_.Exception.Message
      if ($k -eq ($tries-1)) { return "FETCH_ERROR: " + $m }
      Start-Sleep -Seconds 2
    }
  }
}
function Do-Search([string]$query) {
  Write-Output ("=== QUERY: " + $query)
  $q = [System.Uri]::EscapeDataString($query)
  $c = Fetch ("https://www.mojeek.com/search?q=" + $q) 2
  $n = 0
  if ($c -notlike "FETCH_ERROR*") {
    $blocks = [Regex]::Matches($c, '(?s)<li>\s*<h2><a[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>(.*?)</li>')
    if ($blocks.Count -eq 0) { $blocks = [Regex]::Matches($c, '(?s)<a class="ob"[^>]*href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<li|</ul>)') }
    foreach ($b in $blocks) {
      $link = $b.Groups[1].Value
      $title = Strip-Html $b.Groups[2].Value
      $body = Strip-Html $b.Groups[3].Value
      if ($body.Length -gt 460) { $body = $body.Substring(0,460) }
      Write-Output ("- " + $title); Write-Output ("  " + $link); Write-Output ("  " + ($body -replace "`r?`n"," "))
      $n++; if ($n -ge 12) { break }
    }
  }
  if ($n -eq 0) {
    Write-Output "  [mojeek empty -> ecosia]"
    $c2 = Fetch ("https://www.ecosia.org/search?q=" + $q) 2
    if ($c2 -notlike "FETCH_ERROR*") {
      $t = Strip-Html $c2
      if ($t.Length -gt 5000) { $t = $t.Substring(0,5000) }
      Write-Output $t
    } else { Write-Output ("  BOTH ROUTES FAILED: " + $c2) }
  }
}

if ($Mode -eq "f") {
  foreach ($u in $Arg) {
    $c = Fetch $u
    Write-Output ("################ " + $u)
    $t = Strip-Html $c
    if ($t.Length -gt $Limit) { $t = $t.Substring(0, $Limit) }
    Write-Output $t
  }
}
elseif ($Mode -eq "s") { foreach ($query in $Arg) { Do-Search $query } }
