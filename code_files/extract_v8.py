import pymupdf
import pytesseract
from PIL import Image
import io
import re
import json
from pathlib import Path
from collections import Counter

# ============ CONFIG ============
PDF_PATH = "mcqs_1.pdf"
OUTPUT_DIR = Path("output_extract_v8")
START_PAGE = 16          # page 16 holds the "1.A Anatomy" section header
END_PAGE = None
DPI = 300
MIN_TEXT_LENGTH = 50     # fall back to OCR if text layer is too short

# ============ IMAGE WHITELIST (unchanged) ============
KEEP_IMAGES = {
    (17, 1),
    (21, 0), (22, 0), (23, 0), (24, 0),
    (31, 4), (32, 1), (32, 2), (33, 1), (36, 1), (40, 0),
    (54, 0), (57, 0), (57, 1), (58, 0), (59, 0), (59, 1),
    (60, 0), (61, 0), (61, 1), (64, 0), (75, 0), (77, 0),
    (96, 0), (109, 0), (118, 0), (178, 0), (194, 0), (198, 0),
    (202, 0), (203, 0), (219, 0), (307, 0), (363, 0), (384, 0),
    (433, 0),
    (454, 1), (454, 2), (454, 3), (454, 4), (454, 5), (454, 6),
    (455, 0), (455, 1), (455, 2), (455, 3), (455, 4), (455, 5),
    (456, 0), (456, 1), (456, 2), (456, 3), (456, 4), (456, 5),
    (457, 0), (457, 1), (457, 2), (457, 3), (457, 4), (457, 5),
}

# ============ PATTERNS ============
QUESTION_HEADER = re.compile(r'^\s*#*\*?\s*Question\s+(\d+)\s*$', re.IGNORECASE)

ANSWER_PATTERN = re.compile(
    r'(?i)^\s*#*\*?\s*(?:Correct\s+)?(?:Answer|Ans)\s*[:\-–]\s*'
    r'(\d+|[a-dA-D])\b\s*(?:\*?\s*)?(.*)$'
)

RATIONALE_PATTERN = re.compile(
    r'(?i)^\s*#*\*?\s*(?:rationale|explanation|justification)\s*[:\-–]?'
)

OPTION_PATTERN = re.compile(
    r'^\s*(\d+|[a-dA-D]|[ivxIVX]{1,4})\s*[\.\)\-]\s+(.+)$'
)

SECTION_KEYWORDS = [
    'anatomy', 'biomechanics', 'sports medicine', 'neuromuscular',
    'cardiovascular', 'pulmonary', 'paediatrics', 'pediatrics',
    'endocrine', 'burns', 'therapeutic exercises', 'electro', 'hydro',
    'musculoskeletal', 'others', 'vip', 'knowledge', 'figures',
    'quiz bank', 'orthopedic', 'chapter',
]


def is_option_line(line):
    return bool(OPTION_PATTERN.match(line))


def is_merged_options(line):
    """Detect '1. x 2. y 3. z' style lines."""
    return len(re.findall(r'\b\d+\s*[\.\)]\s+\S+', line)) >= 2


def split_merged_options(line):
    pattern = r'(\d+)\s*[\.\)]\s*([^0-9]+?)(?=\s*\d+\s*[\.\)]|$)'
    parts = re.findall(pattern, line)
    return ([f"{n}. {t.strip()}" for n, t in parts if t.strip()]
            if len(parts) >= 2 else [line])


def is_section_header(line):
    clean = line.strip('#*').strip().lower()
    if not clean or len(clean) > 100:
        return False
    return any(kw in clean for kw in SECTION_KEYWORDS)


def is_skip_page(text):
    """Skip index / number-only artifact pages (e.g. pages 10–15)."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return True
    if len(re.findall(r'\b\d+\.', ' '.join(lines))) > 100:
        return True
    number_only = sum(1 for l in lines if re.match(r'^\d+[\.\)]?$', l))
    if len(lines) > 5 and number_only / len(lines) > 0.7:
        return True
    return False


def extract_text(page):
    text = page.get_text("text")
    if len(text.strip()) >= MIN_TEXT_LENGTH:
        return text
    pix = page.get_pixmap(dpi=DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def _append_to_last_question(content_lines, last_q):
    """Merge page-break continuations into the previous question."""
    for _, line in content_lines:
        if not line:
            continue
        ans_m = ANSWER_PATTERN.match(line)
        if ans_m:
            last_q['answer'] = ans_m.group(1)
            rest = ans_m.group(2).strip() if ans_m.lastindex and ans_m.lastindex >= 2 else ''
            if rest:
                if re.search(r'(?i)(rationale|explanation)', rest):
                    r = re.search(r'(?i)(?:rationale|explanation)\s*[:\-–]?\s*(.*)', rest)
                    if r:
                        last_q['rationale'] = (last_q.get('rationale', '') + ' ' + r.group(1)).strip()
                else:
                    last_q['rationale'] = (last_q.get('rationale', '') + ' ' + rest).strip()
            continue
        if RATIONALE_PATTERN.match(line):
            rest = RATIONALE_PATTERN.sub('', line).strip()
            if rest:
                last_q['rationale'] = (last_q.get('rationale', '') + ' ' + rest).strip()
            continue
        if is_merged_options(line):
            last_q.setdefault('options', []).extend(split_merged_options(line))
            continue
        if is_option_line(line):
            last_q.setdefault('options', []).append(line)
            continue
        if last_q.get('options'):
            last_q['options'][-1] += ' ' + line
        elif last_q.get('rationale'):
            last_q['rationale'] += ' ' + line
        else:
            last_q['stem'] = (last_q.get('stem', '') + ' ' + line).strip()


def parse_questions_from_page(text, current_section, last_q_state):
    questions = []
    lines = [l.strip() for l in text.split('\n')]

    q_positions = [(i, int(QUESTION_HEADER.match(l).group(1)))
                   for i, l in enumerate(lines) if QUESTION_HEADER.match(l)]

    if not q_positions:
        if last_q_state:
            pre = [(i, l) for i, l in enumerate(lines) if l]
            if pre:
                _append_to_last_question(pre, last_q_state)
        return questions, current_section, last_q_state

    # Continuation before first header on this page
    first_q = q_positions[0][0]
    if last_q_state:
        pre = [(i, lines[i]) for i in range(first_q) if lines[i]]
        if pre:
            _append_to_last_question(pre, last_q_state)

    for idx, (start_i, q_num) in enumerate(q_positions):
        end_i = q_positions[idx + 1][0] if idx + 1 < len(q_positions) else len(lines)
        block = lines[start_i + 1:end_i]

        stem_parts, options, rationale_parts = [], [], []
        answer = None
        state = 'stem'

        for line in block:
            if not line:
                continue

            ans_m = ANSWER_PATTERN.match(line)
            if ans_m:
                answer = ans_m.group(1)
                state = 'answer'
                rest = ans_m.group(2).strip() if ans_m.lastindex and ans_m.lastindex >= 2 else ''
                if rest:
                    if re.search(r'(?i)(rationale|explanation)', rest):
                        r = re.search(r'(?i)(?:rationale|explanation)\s*[:\-–]?\s*(.*)', rest)
                        if r:
                            rationale_parts.append(r.group(1))
                            state = 'rationale'
                    else:
                        rationale_parts.append(rest)
                continue

            if RATIONALE_PATTERN.match(line):
                state = 'rationale'
                rest = RATIONALE_PATTERN.sub('', line).strip()
                if rest:
                    rationale_parts.append(rest)
                continue

            if is_merged_options(line) and state in ('stem', 'options'):
                options.extend(split_merged_options(line))
                state = 'options'
                continue

            if is_option_line(line) and state in ('stem', 'options'):
                options.append(line)
                state = 'options'
                continue

            if state == 'stem':
                stem_parts.append(line)
            elif state == 'options':
                if options:
                    options[-1] += ' ' + line
                else:
                    stem_parts.append(line)
            elif state == 'answer':
                rationale_parts.append(line)
                state = 'rationale'
            elif state == 'rationale':
                rationale_parts.append(line)

        clean_options = []
        for opt in options:
            m = OPTION_PATTERN.match(opt)
            clean_options.append(m.group(2).strip() if m else opt.strip())

        q_data = {
            'number': q_num,
            'section': current_section,
            'stem': ' '.join(stem_parts).strip(),
            'options': clean_options,
            'answer': answer,
            'rationale': ' '.join(rationale_parts).strip(),
        }
        questions.append(q_data)
        last_q_state = q_data

    return questions, current_section, last_q_state


# ============ MAIN ============
OUTPUT_DIR.mkdir(exist_ok=True)
IMG_DIR = OUTPUT_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)
for f in IMG_DIR.glob("*"):
    if f.is_file():
        f.unlink()

doc = pymupdf.open(PDF_PATH)
end = END_PAGE if END_PAGE else len(doc)

all_questions = []
current_section = "General"
last_q_state = None
debug_lines = []
saved_keys = set()

for page_idx in range(START_PAGE - 1, end):
    page = doc[page_idx]
    page_num = page_idx + 1

    text = extract_text(page)
    if is_skip_page(text):
        debug_lines.append(f"=== Page {page_num} === [SKIPPED]")
        continue

    for line in text.split('\n'):
        if is_section_header(line):
            current_section = line.strip('#*').strip()
            break

    page_qs, current_section, last_q_state = parse_questions_from_page(
        text, current_section, last_q_state
    )
    all_questions.extend(page_qs)

    debug_lines.append(f"=== Page {page_num} === [{current_section}]")
    for q in page_qs:
        debug_lines.append(f"Q{q['number']}: {q['stem'][:80]!r}")
        for opt in q['options']:
            debug_lines.append(f"   - {opt[:80]}")
        if q['answer']:
            debug_lines.append(f"   ANS: {q['answer']}")

    for i, info in enumerate(page.get_images(full=True)):
        if (page_num, i) not in KEEP_IMAGES:
            continue
        try:
            base = doc.extract_image(info[0])
        except Exception:
            continue
        fname = f"page_{page_num:03d}_img_{i}.{base['ext']}"
        (IMG_DIR / fname).write_bytes(base["image"])
        saved_keys.add((page_num, i))
        debug_lines.append(f"   img[{i}] KEPT {fname}")
    debug_lines.append("")

doc.close()

# ---- JSON ----
with open(OUTPUT_DIR / "questions.json", "w", encoding="utf-8") as f:
    json.dump(all_questions, f, indent=2, ensure_ascii=False)

# ---- Readable text ----
with open(OUTPUT_DIR / "questions.txt", "w", encoding="utf-8") as out:
    out.write("PROMETRIC PHYSICAL THERAPY EXAM - QUESTION BANK\n" + "=" * 70 + "\n\n")
    for q in all_questions:
        out.write(f"\n{'─' * 70}\n")
        out.write(f"[{q['section']}] Question {q['number']}:\n{q['stem']}\n")
        for j, opt in enumerate(q['options']):
            mark = " ✓" if q['answer'] and str(j + 1) == q['answer'] else ""
            out.write(f"  {j + 1}. {opt}{mark}\n")
        if q['answer']:
            out.write(f"→ Answer: {q['answer']}\n")
        if q['rationale']:
            out.write(f"  Rationale: {q['rationale']}\n")

# ---- Debug ----
with open(OUTPUT_DIR / "debug.txt", "w", encoding="utf-8") as out:
    out.write('\n'.join(debug_lines))

# ---- Summary ----
print("✅ Extraction complete")
print(f"   Questions : {len(all_questions):>4} → {OUTPUT_DIR / 'questions.txt'}")
print(f"   JSON      : {len(all_questions):>4} → {OUTPUT_DIR / 'questions.json'}")
print(f"   Figures   : {len(saved_keys):>4} → {IMG_DIR}/")
print(f"   Debug     : {OUTPUT_DIR / 'debug.txt'}")

missing = KEEP_IMAGES - saved_keys
if missing:
    print(f"\n⚠️  {len(missing)} requested image(s) NOT found:")
    for p, i in sorted(missing):
        print(f"     page {p} img_{i}")

print("\n📊 Questions per section:")
for sec, cnt in Counter(q['section'] for q in all_questions).most_common():
    print(f"   {sec}: {cnt}")

no_answer = [q for q in all_questions if not q['answer']]
if no_answer:
    print(f"\n⚠️  {len(no_answer)} question(s) without detected answer (first 15):")
    for q in no_answer[:15]:
        print(f"     [{q['section']}] Q{q['number']}: {q['stem'][:60]}")
    if len(no_answer) > 15:
        print(f"     ... and {len(no_answer) - 15} more")