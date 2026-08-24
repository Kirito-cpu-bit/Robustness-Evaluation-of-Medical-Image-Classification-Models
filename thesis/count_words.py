import re
from pathlib import Path

tex = Path(__file__).with_name("dissertation_report.tex").read_text(encoding="utf-8")
tex = re.sub(r"(?<!\\)%.*", "", tex)


def strip_latex(s: str) -> str:
    s = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", s, flags=re.S)
    s = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", s, flags=re.S)
    s = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", s, flags=re.S)
    s = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", s, flags=re.S)
    s = re.sub(r"\\begin\{titlepage\}.*?\\end\{titlepage\}", "", s, flags=re.S)
    for _ in range(6):
        s = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", s)
    s = re.sub(r"[{}\\$&~^_]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def count_words(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", s))


body = strip_latex(tex)
print(f"Main text (excl. title page, tables, bibliography): {count_words(body)} words")

m = re.search(r"\\begin\{abstract\}|\\chapter\*\{\\abstractname\}(.*?)\\newpage", tex, re.S)
if not m:
    m = re.search(r"\\chapter\*\{\\abstractname\}(.*?)\\newpage", tex, re.S)
abstract_block = m.group(1) if m else ""
print(f"Abstract + Keywords: {count_words(strip_latex(abstract_block))} words")

parts = re.split(r"\\chapter\{", tex)
print("\nPer chapter:")
for part in parts[1:]:
    title_end = part.find("}")
    title = part[:title_end]
    content = strip_latex(part[title_end + 1 :])
    print(f"  {title}: {count_words(content)} words")

bib = re.search(r"\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}", tex, re.S)
if bib:
    print(f"\nBibliography: {count_words(strip_latex(bib.group(1)))} words")

# Total including abstract but excluding bib/tables/title
main = strip_latex(re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", tex, flags=re.S))
print(f"\nTotal prose (abstract + Sections 1-3, excl. tables/bib/title): {count_words(main)} words")
