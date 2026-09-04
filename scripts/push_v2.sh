#!/bin/bash
# Push v2 to GitHub with verification; retry on the flaky-TLS "Everything up-to-date" lie.
cd /mnt/d/Pyhon_projects/opop/v2 || exit 1
URL="https://ZihangZ:${GH_PAT}@github.com/Fudan-SMI-lab/opop.git"
for i in 1 2 3; do
  git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 push "$URL" v2:v2 >/tmp/push.txt 2>&1
  R=$(git ls-remote "$URL" refs/heads/v2 2>/dev/null | cut -f1)
  L=$(git rev-parse v2)
  echo "attempt $i: remote=$R local=$L"
  if [ "$R" = "$L" ]; then echo PUSH_OK; exit 0; fi
  tail -2 /tmp/push.txt
  sleep 5
done
echo PUSH_FAILED
exit 1
