"""2 个候选含 "pass" 却 0 个被拒 —— 找出原因。可能是 (a) 在注释里(被剥离),
(b) 是 'passed'/'bypass' 之类不满足词边界。这决定这道门到底多锋利:
如果真实候选只是恰好没写 try/pass,那门是"尚未咬到我们",而不是"无害"。
"""
import glob, pathlib, re
PASS_PATTERN = r"\bpass\b"
def strip_comments(code):
    out = []
    for line in code.split("\n"):
        if "#" in line: line = line[:line.index("#")]
        if "//" in line: line = line[:line.index("//")]
        out.append(line)
    return "\n".join(out)

roots = ["/root/autodl-tmp/opop-workspace/opop-glm/runs-l3/*/candidates/*/source.py",
         "/root/autodl-tmp/opop-workspace/opop-glm/runs-smoke42/*/candidates/*/source.py"]
for f in [x for r in roots for x in glob.glob(r)]:
    src = pathlib.Path(f).read_text(encoding="utf-8", errors="replace")
    if "pass" not in src:
        continue
    print(f"\n### {'/'.join(f.split('/')[-3:])}")
    for i, line in enumerate(src.split("\n"), 1):
        if "pass" in line:
            stripped = strip_comments(line)
            matches = bool(re.search(PASS_PATTERN, stripped))
            print(f"  L{i}: {line.strip()[:88]}")
            print(f"       after strip: {stripped.strip()[:70]!r}  -> regex hit: {matches}")
