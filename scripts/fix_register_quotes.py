"""Repair the quote-placeholder corruption in GAP-REGISTER-AND-FIXES.md.

An earlier patch script wrote CJK-quoted prose using `" + Q + "` / `" + QQ + "` placeholders that were
meant to be substituted with the quote characters and never were, so 16 spots in the register read as
literal Python string-concatenation fragments.

`Q` opened a quote and `QQ` closed it, so they map to the CJK corner brackets used elsewhere in the
same document. Written to a file with newline="" rather than run as a heredoc, because heredocs with
CJK inner quotes have repeatedly hit SyntaxErrors under the GBK console on this host.
"""
import io

PATH = r"D:\Pyhon_projects\opop\v3\docs\GAP-REGISTER-AND-FIXES.md"

with io.open(PATH, encoding="utf-8", newline="") as f:
    text = f.read()

before_q = text.count('" + Q + "')
before_qq = text.count('" + QQ + "')

# Order matters: the QQ form contains the Q form as a substring is NOT true here
# ('" + Q + "' vs '" + QQ + "'), but replacing the longer one first is still the safe habit.
text = text.replace('" + QQ + "', "\u300d")   # closing 」
text = text.replace('" + Q + "', "\u300c")    # opening 「

with io.open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(text)

left = text.count(" + Q + ") + text.count(" + QQ + ")
print("replaced %d opening and %d closing placeholders; %d fragments remain"
      % (before_q, before_qq, left))
