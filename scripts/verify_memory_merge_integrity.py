"""Independently verify the memory merges lost no measured numbers.

The subagent that performed the merges reported checking this itself. A checker reporting clean on its
own work carries no evidential weight (`a-clean-run-is-not-evidence-a-checker-works`), so this reads
the ORIGINAL files back out of git-independent evidence -- the `合并自:` back-links name what each
keep file absorbed, and every absorbed file's numbers must appear in its keep file.

The absorbed files are deleted, so their content cannot be re-read. What CAN be verified without them:

  1. every keep file that claims a merge has a `合并自:` line naming the absorbed slugs
  2. every absorbed slug named anywhere is genuinely gone from disk (not half-deleted)
  3. every `[[link]]` in every memory file either resolves to a file on disk OR to a slug that some
     keep file's `合并自:` line accounts for -- a link that resolves to neither is a dangling pointer
     to a fact that no longer has a home, which is exactly the loss this check is for
  4. MEMORY.md's own links all resolve, and every file on disk is indexed

That is the part checkable from the surviving state. It cannot re-derive the absorbed numbers, and
this script says so rather than implying a stronger guarantee than it delivers.
"""
import os
import re

M = r"D:\ClaudeCode\data\projects\D--Pyhon-projects-opop\memory"

files = {f for f in os.listdir(M) if f.endswith(".md") and f != "MEMORY.md"}
slugs = {f[:-3] for f in files}
bodies = {}
for f in files:
    bodies[f[:-3]] = open(os.path.join(M, f), encoding="utf-8").read()

# 1 + 2: what was absorbed, per the surviving back-links.
absorbed_by = {}
for slug, text in bodies.items():
    m = re.search(r"合并自[::]\s*(.+?)(?:\n\n|\Z)", text, re.DOTALL)
    if not m:
        continue
    named = re.findall(r"\[\[([^\]]+)\]\]", m.group(1))
    if named:
        absorbed_by[slug] = named

print(f"keep files declaring a merge: {len(absorbed_by)}")
total_absorbed = sum(len(v) for v in absorbed_by.values())
print(f"absorbed slugs named: {total_absorbed}")
half_deleted = [(k, a) for k, v in absorbed_by.items() for a in v if a in slugs]
if half_deleted:
    print(f"!! still on disk despite being marked absorbed: {half_deleted}")
else:
    print("all absorbed slugs are gone from disk (no half-merges)")

accounted = {a for v in absorbed_by.values() for a in v}

# 3: dangling wiki links. A link to an absorbed slug is FINE -- the back-link makes it findable.
# A link to a slug that is neither on disk nor absorbed is a pointer into nothing.
dangling = {}
for slug, text in bodies.items():
    body = text.split("---", 2)[-1]
    for link in set(re.findall(r"\[\[([^\]]+)\]\]", body)):
        if link in slugs or link in accounted:
            continue
        dangling.setdefault(link, []).append(slug)
print()
print(f"links resolving to neither a file nor an absorbed slug: {len(dangling)}")
for link, where in sorted(dangling.items())[:15]:
    print(f"  [[{link}]]  cited by: {', '.join(sorted(where)[:3])}")

# 4: the index.
idx = open(os.path.join(M, "MEMORY.md"), encoding="utf-8").read()
linked = set(re.findall(r"\]\(([^)]*\.md)\)", idx))
print()
print(f"MEMORY.md: {len(idx.splitlines())} lines, {len(linked)} links")
print(f"  indexed but absent from disk: {sorted(linked - files) or 'none'}")
print(f"  on disk but unindexed:        {sorted(f for f in files - linked) or 'none'}")
print()
print("NOT verified by this script: that each merged file reproduces every NUMBER from the file it")
print("absorbed. The absorbed files are deleted, so that claim is only re-checkable against git")
print("history or a backup -- it is not re-derivable from the surviving state.")
