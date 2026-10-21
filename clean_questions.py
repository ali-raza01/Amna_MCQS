#!/usr/bin/env python3
"""Clean extracted_text.txt → readable questions-only file."""

import re
from pathlib import Path

INPUT = Path("output_extract_v2/extracted_text.txt")
OUTPUT = Path("output_extract_v2/clean_questions.txt")

# --- Skip these block types entirely (headers, section labels) ---
HEADER_LINE = re.compile(
    r'(?i)^\s*(?:Muscloskeletal|Musculoskeletal|Neuromuscular|Cardiovascular|Therapeutic|'
    r'Orthopedic|Paediatric|Pediatric|Sports|Quiz\s*bank|Electro\s*&|Knowledge|'
    r'Figures|Endocrine|Endrocrine|Others\s*[-–]|PPTE|Chapter\s+\d|Section|VIP)'
)

# --- Answer / rationale markers to strip ---
ANSWER_LINE    = re.compile(r'(?i)^\s*correct\s+answer\s*[:.\-–]')
RATIONALE_LINE = re.compile(r'(?i)^\s*rationale\s*[:.\-–]?')

# --- Page / overlay markers used as natural question boundaries ---
PAGE_MARKER       = re.compile(r'^\s*6\s*$')
QUESTION_OVERLAY  = re.compile(r'(?i)^\s*[tr«]?\s*Question\s+\d+\s*$')

# --- Option patterns (numbered, lettered, roman) ---
NUMBERED = re.compile(r'^\d+\s*[\.\)\-:]\s*\S')
LETTERED = re.compile(r'^[a-dA-D]\s*[\.\)\-:]\s*\S')
ROMAN    = re.compile(r'^[ivIV]{1,3}\s*[\.\)\-:]\s*\S')

def is_option(line):
    return bool(NUMBERED.match(line) or LETTERED.match(line) or ROMAN.match(line))


def strip_answers(text):
    """Remove 'Correct Answer: X' lines and any rationale that follows."""
    out, in_answer = [], False
    for line in text.split('\n'):
        s = line.strip()
        if ANSWER_LINE.match(s) or RATIONALE_LINE.match(s):
            in_answer = True
            continue
        if in_answer:
            # Explicit exit markers
            if PAGE_MARKER.match(s) or QUESTION_OVERLAY.match(s):
                in_answer = False
                out.append(line)
                continue
            # A long line ending in ':' or '?' that isn't an option = new stem
            if s.endswith((':', '?')) and len(s) > 15 and not is_option(s):
                in_answer = False
                out.append(line)
                continue
            continue
        out.append(line)
    return '\n'.join(out)


def split_blocks(text):
    """Split page text into candidate question blocks."""
    parts = re.split(
        r'\n\s*(?:6|[tr«]\s*Question\s+\d+)\s*\n',
        text
    )
    blocks = []
    for part in parts:
        for sub in re.split(r'\n\s*\n', part):
            sub = sub.strip()
            if sub:
                blocks.append(sub)
    return blocks


def parse_block(block):
    lines = [l.strip() for l in block.split('\n') if l.strip()]
    if not lines:
        return None
    if all(HEADER_LINE.match(l) or len(l) < 2 for l in lines):
        return None

    has_lettered = any(LETTERED.match(l) for l in lines)

    stem_lines, options, in_options = [], [], False

    for line in lines:
        if HEADER_LINE.match(line) and not options:
            continue

        # Lettered option set (A/B/C/D or a/b/c/d) takes priority
        if has_lettered:
            if LETTERED.match(line):
                in_options = True
                options.append(line)
            elif in_options:
                if line[0].islower():
                    options[-1] += ' ' + line
                else:
                    break
            else:
                stem_lines.append(line)
        else:
            # Numeric option set
            if NUMBERED.match(line):
                if in_options:
                    # Restart detection: 1 after 4 → new question
                    cur = int(re.match(r'^(\d+)', line).group(1))
                    prev = int(re.match(r'^(\d+)', options[-1]).group(1))
                    if cur <= prev:
                        break
                in_options = True
                options.append(line)
            elif in_options:
                if line[0].islower():
                    options[-1] += ' ' + line
                else:
                    break
            else:
                stem_lines.append(line)

    if len(options) < 2 or not stem_lines:
        return None

    stem = ' '.join(stem_lines).strip()
    # Strip a leaked question number prefix like "32-Patient…"
    stem = re.sub(r'^\d+\s*[-–]\s*', '', stem).strip()

    return {'stem': stem, 'options': options}


def extract(extracted_path):
    text = Path(extracted_path).read_text(encoding='utf-8')
    page_sections = re.split(
        r'={10,}\n=== PAGE (\d+) ===\n={10,}',
        text
    )
    questions = []
    i = 1
    while i < len(page_sections):
        page_num = int(page_sections[i])
        content  = page_sections[i + 1] if i + 1 < len(page_sections) else ''
        i += 2

        ocr = re.search(
            r'--- OCR TEXT ---\n(.*?)(?:\n--- EMBEDDED|\Z)',
            content, re.DOTALL
        )
        if not ocr:
            continue

        cleaned = strip_answers(ocr.group(1))
        for block in split_blocks(cleaned):
            q = parse_block(block)
            if q:
                q['page'] = page_num
                questions.append(q)

    return questions


def clean_option(line):
    m = re.match(r'^(?:\d+|[a-dA-D]|[ivIV]{1,3})\s*[\.\)\-:]\s*(.+)$', line)
    return m.group(1).strip() if m else line.strip()


def write_output(questions, out_path):
    lines = []
    current_page = None
    q_num = 0

    for q in questions:
        if q['page'] != current_page:
            if lines:
                lines.append('')
            current_page = q['page']
            bar = '─' * 60
            lines += [bar, f'PAGE {current_page}', bar, '']

        q_num += 1
        lines.append(f'{q_num}. {q["stem"]}')
        for idx, opt in enumerate(q['options']):
            letter = chr(ord('a') + idx)
            lines.append(f'    {letter}) {clean_option(opt)}')
        lines.append('')

    Path(out_path).write_text('\n'.join(lines), encoding='utf-8')
    return q_num


if __name__ == '__main__':
    qs = extract(INPUT)
    total = write_output(qs, OUTPUT)
    print(f'✅ Extracted {total} clean questions → {OUTPUT}')