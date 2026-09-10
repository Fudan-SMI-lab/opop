"""The clean before/after: was a COMPENSATED dot present anywhere pre-change?

Searched several ways, because "0 of N" is only worth stating if the search could have found a
hit. A compensated dot leaves fingerprints: high/low operand splits, a residual subtraction, or
3+ tl.dot calls accumulating into one acc. Any of them counts as a hit.
"""
import pathlib, re, sys, collections
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "../external_files/box2-runs/runs-l3")
PATTERNS = {
    "hi/lo dot operands":   r"tl\.dot\(\s*\w*[hl]\w*\s*,\s*\w*[hl]\w*",
    "residual subtraction": r"-\s*\w+\.to\(tl\.float32\)",
    "3+ dots in one expr":  r"tl\.dot\([^)]*\)\s*\+\s*tl\.dot\([^)]*\)\s*\+\s*tl\.dot",
    "split/hi/lo naming":   r"\b(split3|split_?hi|hi_?lo|_hi\b|_lo\b)",
    "DOT_MODE knob":        r"DOT_MODE",
}
n_cand = 0; hits = collections.Counter(); hit_files = collections.defaultdict(list)
dtype_knob = 0; choice_lists = collections.Counter()
for d in sorted(root.glob("run-l3-*")):
    for f in sorted(d.glob("candidates/*/*.py")):
        if f.name != "source.py": continue
        n_cand += 1
        src = f.read_text(encoding="utf-8", errors="replace")
        if "COMPUTE_DTYPE" in src: dtype_knob += 1
        for name, pat in PATTERNS.items():
            if re.search(pat, src):
                hits[name] += 1
                hit_files[name].append("%s/%s" % (d.name[-8:], f.parent.name[:13]))
print("candidate source.py files scanned: %d   (with a COMPUTE_DTYPE knob: %d)" % (n_cand, dtype_knob))
print("compensated-dot fingerprints found:")
for name in PATTERNS:
    n = hits[name]
    print("   %-22s %d / %d %s" % (name, n, n_cand, hit_files[name][:4] if n else ""))
