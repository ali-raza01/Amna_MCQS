#!/usr/bin/env python3
"""Extract MCQs from extracted_text.txt — OCR only, no native text."""

import re
from pathlib import Path

INPUT = Path("output_extract_v2/extracted_text.txt")
OUTPUT = Path("output_extract_v2/clean_questions_v3.txt")
SENTINEL = "\x00SPLIT\x00"

# ============ Patterns ============
ANSWER    = re.compile(r'(?i)^\s*(?:correct\s+)?answer\s*[:.\-–]')
RATIONALE = re.compile(r'(?i)^\s*rationale\s*[:.\-–]?')

HEADER = re.compile(
    r'(?i)^\s*[A-Za-z]{0,6}\s*'
    r'(?:Muscloskeletal|Musculoskeletal|Neuromuscular|Cardiovascular|'
    r'Therapeutic|Orthopedic|Paediatric|Pediatric|Sports|Quiz|'
    r'Electro|Knowledge|Figures|Endocrine|Endrocrine|Others|PPTE|Chapter|'
    r'Section|VIP)'
)


def is_option(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if re.match(r'^\d{1,2}\s*[\.\)]\s*\S', s):         # "1. text" / "1) text"
        return True
    if re.match(r'^\d{1,2}\s*-\s*[A-Za-z]', s):         # "1- text" (dash, letter after)
        return True
    if re.match(r'^[a-dA-D]\s*[\.\)]\s*\S', s):         # "a. text" / "A) text"
        return True
    if re.match(r'^[\.\)]\s*[A-Za-z]', s):              # ". text" / ") text"
        return True
    if re.match(r'^[A-Za-z]$', s):                      # single letter "A" / "L" / "K" / "Q"
        return True
    return False


def is_marker(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if re.match(r'^6(?=[\s\-–A-Za-z]|$)', s):           # "6", "6 -A", "6 measurement"
        return True
    if re.match(r'^[Cc]$', s):                          # OCR-mangled "6" → "C"
        return True
    if re.search(r'(?i)\bQuestion\s+\d+\b', s) and len(s) < 40:
        return True
    return False


def strip_rationale(text: str) -> str:
    """Drop 'Correct Answer:' + following rationale, exit on next marker or stem."""
    out = []
    in_rat = False
    for line in text.split('\n'):
        s = line.strip()
        if ANSWER.match(s) or RATIONALE.match(s):
            in_rat = True
            continue
        if in_rat:
            if is_marker(s):
                in_rat = False
                out.append(line)
                continue
            if len(s) > 15 and s.endswith(('?', ':')):
                in_rat = False
                out.append(line)
                continue
            continue
        out.append(line)
    return '\n'.join(out)


def mark_and_split(text: str):
    """Replace every question-boundary marker with a sentinel, then split."""
    # 6 at start of a line, followed by space / dash / letter / EOL
    text = re.sub(r'^[ \t]*6(?=[\s\-–A-Za-z]|$)', SENTINEL,
                  text, flags=re.MULTILINE)
    # Bare C on its own line (OCR misread of 6, seen on page 19)
    text = re.sub(r'^[ \t]*[Cc][ \t]*$', SENTINEL,
                  text, flags=re.MULTILINE)
    # Mid-line 6 glued to text: "movement. 6-A 67-year-old" (safe: needs dash + uppercase)
    text = re.sub(r'(?<=\s)6(?=[\-–][A-Z])', SENTINEL, text)
    # "Question N" line, possibly with an OCR prefix (t / r / , CR / etc.)
    text = re.sub(
        r'^[ \t]*(?:[^\w\n]{0,4}[ \t]*)?(?:[A-Za-z]{1,2}[ \t]+)?Question\s+\d+[ \t]*$',
        SENTINEL, text, flags=re.MULTILINE | re.IGNORECASE
    )
    return text.split(SENTINEL)


def parse_block(block: str):
    """Return {'stem': str, 'options': [str,...]} or None."""
    lines = [l.strip() for l in block.split('\n') if l.strip()]
    # Drop leading section headers
    while lines and HEADER.match(lines[0]) and len(lines[0]) < 60:
        lines.pop(0)
    if not lines:
        return None

    # First option-like line marks where options begin
    first_opt = next((i for i, l in enumerate(lines) if is_option(l)), None)
    if first_opt is None or first_opt == 0:
        return None

    stem_lines = lines[:first_opt]
    opt_lines  = lines[first_opt:]
    if len(opt_lines) < 2:
        return None

    stem = ' '.join(stem_lines)
    stem = re.sub(r'-\s+', '', stem)                 # join hyphenated wraps
    stem = re.sub(r'\s+', ' ', stem).strip()
    stem = re.sub(r'^[-–\d\s\.]+', '', stem).strip() # strip leading "32-" / "-A"
    if len(stem) < 5:
        return None

    options = []
    for opt in opt_lines:
        if is_option(opt):
            cleaned = re.sub(
                r'^(?:\d{1,2}|[a-dA-D]|[\.\)])\s*[\.\)\-:–]?\s*',
                '', opt
            ).strip()
            if cleaned:
                options.append(cleaned)
        elif options and re.match(r'^[a-z][a-z0-9\s]*$', opt) and len(opt) > 3:
            # lowercase continuation of previous option (wrapped line)
            options[-1] += ' ' + opt.strip()
        else:
            break  # end of options

    if len(options) < 2:
        return None
    return {'stem': stem, 'options': options}


def extract(path: Path):
    text = Path(path).read_text(encoding='utf-8')
    pages = re.compile(r'={10,}\n=== PAGE (\d+) ===\n={10,}').split(text)

    all_qs = []
    i = 1
    while i < len(pages):
        page_num = int(pages[i])
        content  = pages[i + 1] if i + 1 < len(pages) else ''
        i += 2

        # OCR TEXT ONLY — skip NATIVE TEXT entirely
        m = re.search(r'--- OCR TEXT ---\n(.*?)(?:\n--- EMBEDDED|\Z)',
                      content, re.DOTALL)
        if not m:
            continue

        cleaned = strip_rationale(m.group(1))
        for block in mark_and_split(cleaned):
            q = parse_block(block)
            if q:
                q['page'] = page_num
                all_qs.append(q)
    return all_qs


def write(qs, path: Path) -> int:
    lines = []
    for n, q in enumerate(qs, 1):
        lines.append(f'Question {n} (Page {q["page"]})')
        lines.append(q['stem'])
        for j, opt in enumerate(q['options'], 1):
            lines.append(f'{j}. {opt}')
            if j < len(q['options']):
                lines.append('')
        lines.append('')
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    return len(qs)


if __name__ == '__main__':
    qs = extract(INPUT)
    n  = write(qs, OUTPUT)
    print(f'✅ {n} questions → {OUTPUT}')