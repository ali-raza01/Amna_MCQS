#!/usr/bin/env python3
"""Clean extracted_text.txt → readable questions-only file."""

import re
from pathlib import Path

INPUT  = Path("output_extract_v2/extracted_text.txt")
OUTPUT = Path("output_extract_v2/clean_questions_v2.txt")

# --- header / section-label detection (allow small OCR noise prefix like "Vee ") ---
HEADER_LINE = re.compile(
    r'(?i)^\s*[A-Za-z]{0,6}\s*'
    r'(?:Muscloskeletal|Musculoskeletal|Neuromuscular|Cardiovascular|'
    r'Therapeutic|Orthopedic|Paediatric|Pediatric|Sports|Quiz\s*bank|'
    r'Electro|Knowledge|Figures|Endocrine|Endrocrine|Others|PPTE|Chapter|Section|VIP)'
)

ANSWER_LINE    = re.compile(r'(?i)^\s*correct\s+answer\s*[:.\-–]')
RATIONALE_LINE = re.compile(r'(?i)^\s*rationale\s*[:.\-–]?')

# marker: bare "6" line OR "Question N" possibly with OCR-noise prefix/suffix
MARKER = re.compile(
    r'\n\s*(?:6|(?:[a-zA-Z\(\)\[\]«tr]{0,5}\s*)?Question\s+\d+[^\n]{0,5}?)\s*\n',
    re.IGNORECASE
)

NUMBERED = re.compile(r'^\d+\s*[\.\)\-:]\s*\S')
LETTERED = re.compile(r'^[a-dA-D]\s*[\.\)\-:]\s*\S')
ROMAN    = re.compile(r'^[ivIV]{1,3}\s*[\.\)\-:]\s*\S')

def is_option(line):
    return bool(NUMBERED.match(line) or LETTERED.match(line) or ROMAN.match(line))


def is_header(line):
    return bool(HEADER_LINE.match(line))


# ---------- strip answers + rationale ----------
def strip_answers(text):
    out, in_answer = [], False
    for line in text.split('\n'):
        s = line.strip()
        if ANSWER_LINE.match(s) or RATIONALE_LINE.match(s):
            in_answer = True
            continue
        if in_answer:
            # exit if this line is a new question marker
            if MARKER.search('\n' + s + '\n'):
                in_answer = False
                out.append(line)
                continue
            # exit if this line looks like a new stem (long, ends with ? or :)
            if s.endswith((':', '?')) and len(s) > 15 and not is_option(s):
                in_answer = False
                out.append(line)
                continue
            continue
        out.append(line)
    return '\n'.join(out)


# ---------- classify a blank-line-separated sub-block ----------
def classify_and_split(lines):
    """Return {'stem': [...], 'opts': [...]} for a sub-block.

    Distinguishes real options from stem lines that start with a question
    number (e.g., '1- A patient presents …' is a stem, not option 1)."""
    if not lines:
        return {'stem': [], 'opts': []}

    stem_end = len(lines)
    for i, line in enumerate(lines):
        if not is_option(line):
            stem_end = i + 1
            continue
        # it's option-shaped — decide if it's actually a stem with number prefix
        if i + 1 < len(lines) and not is_option(lines[i + 1]):
            stem_end = i + 1   # next line isn't option → this was a stem
            continue
        stem_end = i
        break

    return {'stem': lines[:stem_end], 'opts': lines[stem_end:]}


# ---------- split page text into questions ----------
def split_into_questions(text):
    questions = []
    for part in MARKER.split(text):
        subs = re.split(r'\n\s*\n', part)
        current = {'stem': [], 'opts': []}
        for sub in subs:
            sub = sub.strip()
            if not sub:
                continue
            lines = [l.strip() for l in sub.split('\n') if l.strip()]
            if all(is_header(l) or len(l) < 3 for l in lines):
                continue
            split = classify_and_split(lines)
            if split['stem']:
                # a new stem → finalise whatever was in progress
                if current['opts']:
                    questions.append(current)
                    current = {'stem': [], 'opts': []}
                current['stem'].extend(split['stem'])
                if split['opts']:
                    current['opts'].extend(split['opts'])
            else:
                current['opts'].extend(split['opts'])
        if current['stem'] and current['opts']:
            questions.append(current)
    return questions


# ---------- build final question ----------
def build_question(block, page):
    stem = ' '.join(block['stem'])
    stem = re.sub(r'-\s+', '', stem)               # join hyphenated line breaks
    stem = re.sub(r'^\d+\s*[-–]\s*', '', stem)     # strip "32-Patient…" prefix
    stem = re.sub(r'\s+', ' ', stem).strip()
    if not stem or len(block['opts']) < 2:
        return None
    return {'stem': stem, 'options': block['opts'], 'page': page}


def clean_option_text(line):
    m = re.match(r'^(?:\d+|[a-dA-D]|[ivIV]{1,3})\s*[\.\)\-:]\s*(.+)$', line)
    txt = m.group(1) if m else line
    txt = re.sub(r'-\s+', '', txt)
    return re.sub(r'\s+', ' ', txt).strip()


# ---------- extract ----------
def extract(path):
    text = Path(path).read_text(encoding='utf-8')
    page_sections = re.split(r'={10,}\n=== PAGE (\d+) ===\n={10,}', text)
    out = []
    i = 1
    while i < len(page_sections):
        page = int(page_sections[i])
        content = page_sections[i + 1] if i + 1 < len(page_sections) else ''
        i += 2
        ocr = re.search(r'--- OCR TEXT ---\n(.*?)(?:\n--- EMBEDDED|\Z)',
                        content, re.DOTALL)
        if not ocr:
            continue
        cleaned = strip_answers(ocr.group(1))
        for block in split_into_questions(cleaned):
            q = build_question(block, page)
            if q:
                out.append(q)
    return out


# ---------- write ----------
def write(questions, path):
    lines = []
    for n, q in enumerate(questions, 1):
        lines.append(f'Question {n} (Page {q["page"]})')
        lines.append(q['stem'])
        for j, opt in enumerate(q['options'], 1):
            lines.append(f'{j}. {clean_option_text(opt)}')
            if j < len(q['options']):
                lines.append('')
        lines.append('')
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    return len(questions)


if __name__ == '__main__':
    qs = extract(INPUT)
    total = write(qs, OUTPUT)
    print(f'✅ {total} questions → {OUTPUT}')