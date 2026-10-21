import pymupdf
import pytesseract
from PIL import Image
import io
import re
import json
from pathlib import Path
from collections import defaultdict

PDF_PATH = "mcqs_1.pdf"
OUTPUT_DIR = Path("output_extract_v9")
START_PAGE = 16
END_PAGE = None
DPI = 300
MIN_TEXT_LENGTH = 50

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

# ============ FIXED PATTERNS ============
# More permissive header: handles "Question 12", "Question 12:", "Question 12.",
# "Q12", "## Question 12", "Question: 12" etc.
QUESTION_HEADER = re.compile(
    r'^\s*[#\*]*\s*(?:Question|Q\.?)\s*[:\-–]?\s*(\d{1,3})\s*[:\-–\.\)]?\s*$',
    re.IGNORECASE,
)

# FIXED: (?!\d) instead of \b — allows "4rationale:" to match
ANSWER_PATTERN = re.compile(
    r'(?i)^\s*[#\*]*\s*(?:Correct\s+)?(?:Answer|Ans)\s*[:\-–]\s*'
    r'(\d{1,3}|[a-dA-D])(?!\d)\s*(.*)$'
)

RATIONALE_PATTERN = re.compile(
    r'(?i)^\s*[#\*]*\s*(?:rationale|explanation|justification)\s*[:\-–]?\s*(.*)$'
)

OPTION_PATTERN = re.compile(r'^\s*(\d{1,2})\s*[\.\)\-]\s+(.+)$')

SECTION_KEYWORDS = {
    'anatomy', 'biomechanics', 'sports medicine', 'neuromuscular',
    'cardiovascular', 'pulmonary', 'paediatrics', 'pediatrics',
    'endocrine', 'burns', 'therapeutic exercise', 'electro',
    'hydrotherapy', 'musculoskeletal', 'quiz bank', 'orthopedic',
    'knowledge', 'figures', 'vip',
}


def is_section_header(line):
    clean = re.sub(r'^[#\*\d\.\s]+', '', line).strip().lower()
    if not clean or len(clean) > 100:
        return False
    return any(clean == kw or clean.startswith(kw + ' ') or clean.startswith(kw + ':')
               for kw in SECTION_KEYWORDS)


def is_skip_page(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return True
    if len(re.findall(r'\b\d+\.', ' '.join(lines))) > 100:
        return True
    number_only = sum(1 for l in lines if re.match(r'^\d+[\.\)]?$', l))
    if len(lines) > 5 and number_only / len(lines) > 0.7:
        return True
    return False


def is_merged_options(line):
    return len(re.findall(r'(?:^|\s)\d{1,2}\s*[\.\)]\s+\S+', line)) >= 2


def split_merged_options(line):
    results = []
    for m in re.finditer(r'(\d{1,2})\s*[\.\)]\s+([^0-9]+?)(?=\s+\d{1,2}\s*[\.\)]|$)', line):
        results.append((m.group(1), m.group(2).strip()))
    return results


def extract_text(page):
    text = page.get_text("text")
    if len(text.strip()) >= MIN_TEXT_LENGTH:
        return text
    pix = page.get_pixmap(dpi=DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def _consume_line(line, q):
    """Feed a single continuation line into an existing question dict.
    Returns True if consumed, False if it should be treated as new content."""
    if not line.strip():
        return True

    ans_m = ANSWER_PATTERN.match(line)
    if ans_m:
        q['answer'] = ans_m.group(1)
        trailing = ans_m.group(2).strip()
        if trailing:
            r_m = re.search(r'(?i)(?:rationale|explanation)\s*[:\-–]?\s*(.*)', trailing)
            if r_m:
                q['rationale'] = (q.get('rationale', '') + ' ' + r_m.group(1)).strip()
            else:
                q['rationale'] = (q.get('rationale', '') + ' ' + trailing).strip()
        return True

    rat_m = RATIONALE_PATTERN.match(line)
    if rat_m:
        if rat_m.group(1):
            q['rationale'] = (q.get('rationale', '') + ' ' + rat_m.group(1)).strip()
        return True

    if is_merged_options(line):
        for _, text in split_merged_options(line):
            q.setdefault('options', []).append(text)
        return True

    m = OPTION_PATTERN.match(line)
    if m:
        q.setdefault('options', []).append(m.group(2).strip())
        return True

    # Plain continuation text
    if q.get('answer') is not None:
        q['rationale'] = (q.get('rationale', '') + ' ' + line).strip()
    elif q.get('options'):
        q['options'][-1] += ' ' + line
    else:
        q['stem'] = (q.get('stem', '') + ' ' + line).strip()
    return True


def _would_start_orphan(line, last_q):
    """Detect a new (header-less) question starting with option '1.'."""
    if last_q is None:
        return False
    m = OPTION_PATTERN.match(line)
    if not m or m.group(1) != '1':
        return False
    # Previous question already completed (has answer) OR already has many options
    return last_q.get('answer') is not None or len(last_q.get('options', [])) >= 4


def parse_orphan_block(lines):
    """Parse a header-less block of content into a question (stem may be missing)."""
    orphan = {'number': None, 'stem': '', 'options': [], 'answer': None,
              'rationale': '', 'orphan': True}
    stem_parts = []
    state = 'stem'

    for line in lines:
        if not line.strip():
            continue
        ans_m = ANSWER_PATTERN.match(line)
        if ans_m:
            orphan['answer'] = ans_m.group(1)
            trailing = ans_m.group(2).strip()
            state = 'rationale'
            if trailing:
                r_m = re.search(r'(?i)(?:rationale|explanation)\s*[:\-–]?\s*(.*)', trailing)
                orphan['rationale'] += ' ' + (r_m.group(1) if r_m else trailing)
            continue
        rat_m = RATIONALE_PATTERN.match(line)
        if rat_m:
            state = 'rationale'
            if rat_m.group(1):
                orphan['rationale'] += ' ' + rat_m.group(1)
            continue
        if state == 'stem':
            if is_merged_options(line):
                for _, t in split_merged_options(line):
                    orphan['options'].append(t)
                state = 'options'
                continue
            m = OPTION_PATTERN.match(line)
            if m:
                orphan['options'].append(m.group(2).strip())
                state = 'options'
                continue
            stem_parts.append(line)
        elif state == 'options':
            if is_merged_options(line):
                for _, t in split_merged_options(line):
                    orphan['options'].append(t)
                continue
            m = OPTION_PATTERN.match(line)
            if m:
                orphan['options'].append(m.group(2).strip())
                continue
            if orphan['options']:
                orphan['options'][-1] += ' ' + line
            else:
                stem_parts.append(line)
        elif state == 'rationale':
            orphan['rationale'] += ' ' + line

    orphan['stem'] = ' '.join(stem_parts).strip()
    orphan['rationale'] = orphan['rationale'].strip()

    if not orphan['options'] and orphan['answer'] is None:
        return None
    return orphan


def parse_pre_header_lines(pre_lines, last_q, orphans_out):
    """Handle content before first 'Question N' header on a page."""
    if not pre_lines:
        return
    if last_q is None:
        orphan = parse_orphan_block(pre_lines)
        if orphan:
            orphans_out.append(orphan)
        return

    i = 0
    while i < len(pre_lines):
        line = pre_lines[i]
        if _would_start_orphan(line, last_q):
            orphan = parse_orphan_block(pre_lines[i:])
            if orphan:
                orphans_out.append(orphan)
            return
        _consume_line(line, last_q)
        i += 1


def parse_question_block(lines, q_num, section):
    q = {'number': q_num, 'section': section, 'stem': '', 'options': [],
         'answer': None, 'rationale': '', 'orphan': False}
    stem_parts = []
    state = 'stem'

    for line in lines:
        if not line.strip():
            continue
        ans_m = ANSWER_PATTERN.match(line)
        if ans_m:
            q['answer'] = ans_m.group(1)
            trailing = ans_m.group(2).strip()
            state = 'rationale'
            if trailing:
                r_m = re.search(r'(?i)(?:rationale|explanation)\s*[:\-–]?\s*(.*)', trailing)
                q['rationale'] += ' ' + (r_m.group(1) if r_m else trailing)
            continue
        rat_m = RATIONALE_PATTERN.match(line)
        if rat_m:
            state = 'rationale'
            if rat_m.group(1):
                q['rationale'] += ' ' + rat_m.group(1)
            continue
        if state == 'stem':
            if is_merged_options(line):
                for _, t in split_merged_options(line):
                    q['options'].append(t)
                state = 'options'
                continue
            m = OPTION_PATTERN.match(line)
            if m:
                q['options'].append(m.group(2).strip())
                state = 'options'
                continue
            stem_parts.append(line)
        elif state == 'options':
            if is_merged_options(line):
                for _, t in split_merged_options(line):
                    q['options'].append(t)
                continue
            m = OPTION_PATTERN.match(line)
            if m:
                q['options'].append(m.group(2).strip())
                continue
            if q['options']:
                q['options'][-1] += ' ' + line
            else:
                stem_parts.append(line)
        elif state == 'rationale':
            q['rationale'] += ' ' + line

    q['stem'] = ' '.join(stem_parts).strip()
    q['rationale'] = q['rationale'].strip()
    return q


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
last_q = None
saved_keys = set()
debug_lines = []
raw_text_pages = []

for page_idx in range(START_PAGE - 1, end):
    page = doc[page_idx]
    page_num = page_idx + 1
    text = extract_text(page)
    raw_text_pages.append(f"\n\n{'=' * 60}\nPAGE {page_num}\n{'=' * 60}\n{text}")

    if is_skip_page(text):
        debug_lines.append(f"=== Page {page_num} === [SKIPPED]")
        continue

    for line in text.split('\n'):
        if is_section_header(line):
            current_section = re.sub(r'^[#\*\d\.\s]+', '', line).strip()
            break

    lines = [l.strip() for l in text.split('\n')]
    header_positions = [(i, int(m.group(1)))
                        for i, l in enumerate(lines)
                        if (m := QUESTION_HEADER.match(l))]

    debug_lines.append(f"=== Page {page_num} === [{current_section}]")

    if not header_positions:
        content = [l for l in lines if l]
        if content:
            orphans = []
            parse_pre_header_lines(content, last_q, orphans)
            for o in orphans:
                o['section'] = current_section
                all_questions.append(o)
                last_q = o
                debug_lines.append(
                    f"  ORPHAN: opts={len(o['options'])} ans={o['answer']} "
                    f"stem={o['stem'][:50]!r}")
        debug_lines.append("")
    else:
        # Pre-header continuation / orphans
        pre = [l for l in lines[:header_positions[0][0]] if l]
        if pre:
            orphans = []
            parse_pre_header_lines(pre, last_q, orphans)
            for o in orphans:
                o['section'] = current_section
                all_questions.append(o)
                debug_lines.append(
                    f"  ORPHAN(pre): opts={len(o['options'])} ans={o['answer']}")
            if orphans:
                last_q = orphans[-1]

        # Parse each question segment
        for idx, (start_i, q_num) in enumerate(header_positions):
            end_i = header_positions[idx + 1][0] if idx + 1 < len(header_positions) else len(lines)
            block = [l for l in lines[start_i + 1:end_i] if l]
            q = parse_question_block(block, q_num, current_section)
            all_questions.append(q)
            last_q = q
            debug_lines.append(
                f"  Q{q_num}: opts={len(q['options'])} ans={q['answer']} "
                f"stem={q['stem'][:50]!r}")

        debug_lines.append("")

    # Images (whitelist only)
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
        debug_lines.append(f"  img[{i}] KEPT {fname}")

doc.close()

# ============ OUTPUT ============
with open(OUTPUT_DIR / "questions.json", "w", encoding="utf-8") as f:
    json.dump(all_questions, f, indent=2, ensure_ascii=False)

with open(OUTPUT_DIR / "questions.txt", "w", encoding="utf-8") as out:
    out.write("PROMETRIC PT EXAM — Question Bank\n" + "=" * 70 + "\n")
    for q in all_questions:
        out.write(f"\n{'─' * 70}\n")
        tag = " [ORPHAN — header missing in source]" if q.get('orphan') else ""
        out.write(f"[{q.get('section', 'General')}] Q{q['number']}{tag}\n")
        out.write(q['stem'] + '\n')
        for j, opt in enumerate(q['options'], 1):
            out.write(f"  {j}. {opt}\n")
        if q['answer']:
            out.write(f"→ Answer: {q['answer']}\n")
        if q['rationale']:
            out.write(f"  Rationale: {q['rationale']}\n")

with open(OUTPUT_DIR / "debug.txt", "w", encoding="utf-8") as out:
    out.write('\n'.join(debug_lines))

with open(OUTPUT_DIR / "raw_text.txt", "w", encoding="utf-8") as out:
    out.write('\n'.join(raw_text_pages))

# ============ SUMMARY & DIAGNOSTICS ============
print("✅ Done")
print(f"   Total questions : {len(all_questions)}")
print(f"   With answer     : {sum(1 for q in all_questions if q['answer'])}")
print(f"   Orphans         : {sum(1 for q in all_questions if q.get('orphan'))}")
print(f"   Images saved    : {len(saved_keys)} / {len(KEEP_IMAGES)}")

# Per-section gap report
by_section = defaultdict(list)
for q in all_questions:
    if not q.get('orphan') and q.get('number'):
        by_section[q.get('section', 'Unknown')].append(q['number'])

print("\n📊 Question number gaps by section:")
for sec, nums in by_section.items():
    nums_set = set(nums)
    if not nums_set:
        continue
    expected = set(range(min(nums_set), max(nums_set) + 1))
    missing = sorted(expected - nums_set)
    gap_str = f"  ⚠ missing: {missing[:15]}" if missing else "  ✓"
    print(f"   {sec}: {len(nums_set)} Qs ({min(nums_set)}-{max(nums_set)}){gap_str}")

orphans = [q for q in all_questions if q.get('orphan')]
if orphans:
    print(f"\n⚠  {len(orphans)} orphan question(s) (header missing in source):")
    for q in orphans:
        print(f"   [{q.get('section','?')}] opts={len(q['options'])} "
              f"ans={q['answer']} stem={q['stem'][:40]!r}")

missing_imgs = KEEP_IMAGES - saved_keys
if missing_imgs:
    print(f"\n⚠  {len(missing_imgs)} image(s) not found:")
    for p, i in sorted(missing_imgs):
        print(f"     page {p} img_{i}")