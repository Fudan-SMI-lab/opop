#!/usr/bin/env python3
"""What providers/models does this box's opencode config expose? Keys are never printed.

Purpose: confirm a box can actually reach zhipuai/glm-5.3 before committing it to a 12-hour
experiment. A missing provider surfaces as a dead agent call much later, which is the
expensive way to learn it.
"""
import json
import pathlib
import re
import sys

CANDIDATES = [
    "/root/.config/opencode/opencode.jsonc",
    "/root/.config/opencode/opencode.json",
    "/root/autodl-tmp/opop-workspace/opop/.opencode/opencode.jsonc",
    "/root/autodl-tmp/opop-workspace/opop/.opencode/opencode.json",
    "/root/autodl-tmp/opop-workspace/opop-glm/.opencode/opencode.jsonc",
    "/root/autodl-tmp/opop-workspace/opop-glm/.opencode/opencode.json",
]


def strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments, tolerating them inside the file but not inside strings."""
    out = []
    in_str = False
    esc = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = len(text) if j == -1 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i)
            i = len(text) if j == -1 else j + 2
            continue
        out.append(ch)
        i += 1
    s = "".join(out)
    # trailing commas
    return re.sub(r",(\s*[}\]])", r"\1", s)


for path in (sys.argv[1:] or CANDIDATES):
    p = pathlib.Path(path)
    if not p.exists():
        continue
    print(f"=== {p}")
    try:
        d = json.loads(strip_jsonc(p.read_text(encoding="utf-8", errors="replace")))
    except Exception as exc:  # noqa: BLE001
        print(f"    unparseable: {type(exc).__name__}: {exc}")
        continue
    prov = d.get("provider") or {}
    if not prov:
        print(f"    no provider block (top-level keys: {list(d)[:8]})")
    for name, v in prov.items():
        opts = v.get("options") or {}
        key = opts.get("apiKey")
        models = list((v.get("models") or {}))
        print(f"    {name}: baseURL={opts.get('baseURL')} "
              f"apiKey={'SET (len %d)' % len(key) if key else 'ABSENT'}")
        print(f"       models: {models[:8]}")
