"""Compact MEMORY.md's one-line-per-entry index without losing an entry.

WHY A SCRIPT AND NOT HAND EDITS. The index is 139 entries; hand-shortening invites dropping one
silently, and the failure would be invisible -- a missing pointer reads exactly like a memory that
was never written. So the rule here is mechanical: every line that starts with "- [" must still
start with "- [" afterwards, and the COUNT must be identical. The script asserts both.

It also fixes a real defect found while measuring: line 23 has TWO entries run together
("...正对照- [a constant reading is a broken probe]..."), so that second entry is currently
invisible to anything that reads one entry per line.
"""
import re
import sys

PATH = ("D:/ClaudeCode/data/projects/D--Pyhon-projects-opop/memory/MEMORY.md")

with open(PATH, encoding="utf-8") as fh:
    original = fh.read()

lines = original.split("\n")

# --- 1. split the two entries that share a line -------------------------------------------------
# The join happened where a hook-inserted section header swallowed the newline. Split on a "- ["
# that is preceded by a non-space character, which can only be a run-together.
fixed: list[str] = []
for line in lines:
    if line.startswith("- ") and line.count("](") > 1:
        # Break before every "- [" that is not at the start of the line.
        parts = re.split(r"(?<=\S)(?=- \[)", line)
        fixed.extend(parts)
    else:
        fixed.append(line)
lines = fixed

# --- 2. shorten the hook, and drop link TEXT that merely repeats the filename --------------------
#
# WHERE THE BYTES ACTUALLY ARE, measured before choosing a strategy: of 24.9 KB across 148 entries,
# the link text is 5.6 KB, the paths 6.6 KB and the hooks only 4.6 KB. So the file is dominated by
# LINKS, not prose -- and 79 of the 148 entries have a link text that is character-for-character the
# filename with dashes turned into spaces, i.e. the same string twice on one line.
#
# That measurement changed the plan. Squeezing hooks alone could not reach the budget, and pushing
# the hook cap down far enough to try left hooks severed mid-number ("ρ(头寸") -- destroying the one
# thing a hook is for. Shortening the redundant link TEXT costs nothing, because the path beside it
# still carries the full name.
MAX_HOOK = 80
# A title longer than this is shortened when it merely restates its own filename. Keeps enough words
# to stay recognisable in a list; the path is one character away for anything ambiguous.
MAX_TITLE_WORDS = 5

def shorten_title(line: str) -> str:
    """Trim a link TEXT that is just its own filename spelled with spaces.

    Only touches that redundant case: a title someone wrote by hand (e.g. "project state" pointing
    at `opop-v2-project-state.md`) carries information the path does not and is left alone.
    """
    def repl(m: re.Match) -> str:
        text, path = m.group(1), m.group(2)
        stem = path[:-3] if path.endswith(".md") else path
        if text.replace(" ", "-") != stem:
            return m.group(0)          # not redundant -- a real title
        words = text.split()
        if len(words) <= MAX_TITLE_WORDS:
            return m.group(0)
        return f"[{' '.join(words[:MAX_TITLE_WORDS])}…]({path})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+\.md)\)", repl, line)


def shorten(line: str) -> str:
    if not line.lstrip().startswith(("- [", "- **[")):
        return line
    line = shorten_title(line)
    if " — " not in line:
        return line
    head, hook = line.split(" — ", 1)
    if len(hook) <= MAX_HOOK:
        return line
    # Cut at a clause boundary so the hook stays a sentence rather than a truncation. Prefer the
    # last separator that still fits; a hook severed mid-number is worse than a longer one, so when
    # no boundary is available the hook is LEFT INTACT rather than hard-cut.
    floor = MAX_HOOK // 3
    best = None
    for sep in ("⇒", ";", ";", "、", ",", ","):
        idx = hook.rfind(sep, 0, MAX_HOOK)
        if idx > floor and (best is None or idx > best):
            best = idx
    if best is None:
        return line
    return f"{head} — {hook[:best].rstrip(' ;;、,,')}"

compacted = [shorten(ln) for ln in lines]

before_entries = sum(1 for ln in lines if ln.lstrip().startswith(("- [", "- **[")))
after_entries = sum(1 for ln in compacted if ln.lstrip().startswith(("- [", "- **[")))
out = "\n".join(compacted)

# --- 3. the guards. An index that lost an entry is worse than an index that is too long ----------
if after_entries != before_entries:
    print(f"REFUSED: entry count changed {before_entries} -> {after_entries}")
    sys.exit(1)
links_before = set(re.findall(r"\]\(([^)]+\.md)\)", original))
links_after = set(re.findall(r"\]\(([^)]+\.md)\)", out))
if links_before - links_after:
    print(f"REFUSED: these links would be lost: {sorted(links_before - links_after)}")
    sys.exit(1)

with open(PATH, "w", encoding="utf-8") as fh:
    fh.write(out)

print(f"entries: {before_entries} (unchanged)")
print(f"distinct links: {len(links_before)} -> {len(links_after)}")
print(f"bytes: {len(original.encode('utf-8'))} -> {len(out.encode('utf-8'))}")
