import pymupdf
import pytesseract
from PIL import Image
import io
import re
from pathlib import Path

# ============ CONFIG ============
PDF_PATH = "mcqs_1.pdf"       # adjust if your file is named differently
OUTPUT_DIR = Path("output")
START_PAGE = 17
END_PAGE = None
DPI = 300

# ============ IMAGE WHITELIST ============
# Only these (page, embedded-image-index) pairs are saved.
# The indices match what extract_v2.py reported (page_XXX_img_N).
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
ANSWER_LINE = re.compile(
    r'(?i)^\s*(?:correct\s+)?(?:answer|ans|rationale|explanation|solution|justification)\s*[:\-–]'
)
QUESTION_MARKER = re.compile(
    r'^\s*(?:Question\s+|Q\.?\s*)?(\d{1,2})[\s\.\):\-–]*$',
    re.IGNORECASE,
)
CHAPTER_LINE = re.compile(r'(?i)^\s*chapter\s+\d+\s*$')
SECTION_LINE = re.compile(
    r'(?i)^\s*(?:muscloskeletal|musculoskeletal|neuromuscular|cardiovascular|therapeutic|'
    r'orthopedic|paediatric|pediatric|sports|quiz\s*bank|electro\s*&\s*hydro|'
    r'knowledge|figures|endrocrine|endocrine|others)\b'
)
OPTION_RES = [
    re.compile(r'^\s*(\d+)\s*[\.\)]\s+(.+)$'),
    re.compile(r'^\s*([A-Da-d])\s*[\.\)]\s+(.+)$'),
    re.compile(r'^\s*([ivxIVX]{1,4})\s*[\.\)]\s+(.+)$'),
]

def is_option(s):
    return any(r.match(s) for r in OPTION_RES)

def looks_like_header(s):
    if CHAPTER_LINE.match(s):
        return True
    if SECTION_LINE.match(s) and len(s) < 80:
        return True
    if re.search(r'(?i)\b(review\s*&\s*study\s*guide)\b', s):
        return True
    return False

# ============ PARSER ============
def parse_page_ocr(text):
    questions = []
    stem, opts = [], []
    in_opts = False
    in_answer_mode = False
    answer_lines = 0
    first_content_seen = False

    def flush():
        nonlocal stem, opts, in_opts
        st = '\n'.join(x for x in stem if x.strip()).strip()
        if st or opts:
            questions.append({'stem': st, 'options': opts[:]})
        stem, opts, in_opts = [], [], False

    for raw in text.split('\n'):
        s = raw.strip()
        if not s:
            continue
        if not first_content_seen:
            if QUESTION_MARKER.match(s) or is_option(s) or s.endswith((':', '?', '؟')):
                first_content_seen = True
            else:
                continue
        if looks_like_header(s):
            continue
        if QUESTION_MARKER.match(s):
            flush()
            in_answer_mode = False
            answer_lines = 0
            continue
        if ANSWER_LINE.match(s):
            in_answer_mode = True
            answer_lines = 0
            continue
        if in_answer_mode:
            answer_lines += 1
            if answer_lines > 40 or s.endswith(('?', ':', '؟')):
                in_answer_mode = False
                answer_lines = 0
            else:
                continue
        if is_option(s):
            opts.append(s)
            in_opts = True
        else:
            if in_opts and s[0].islower():
                opts[-1] = opts[-1] + ' ' + s
            else:
                if in_opts:
                    flush()
                stem.append(s)
    flush()
    return questions

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
debug_lines = []
q_counter = 0
saved_keys = set()

for page_idx in range(START_PAGE - 1, end):
    page = doc[page_idx]
    page_num = page_idx + 1

    # Full-page OCR — same as extract_v2 (300 DPI, PSM 3 default)
    pix = page.get_pixmap(dpi=DPI)
    page_img = Image.open(io.BytesIO(pix.tobytes("png")))
    ocr_text = pytesseract.image_to_string(page_img)

    page_qs = parse_page_ocr(ocr_text)

    debug_lines.append(f"=== Page {page_num} ===")
    for i, q in enumerate(page_qs, 1):
        q_counter += 1
        q['number'] = q_counter
        q['page'] = page_num
        all_questions.append(q)
        debug_lines.append(f"Q{i} [global {q_counter}] stem: {q['stem'][:100]!r}")
        for opt in q['options']:
            debug_lines.append(f"   {opt}")

    # Save only whitelisted images
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
        debug_lines.append(f"img[{i}] KEPT {fname}")
    debug_lines.append("")

doc.close()

# ---- questions.txt ----
with open(OUTPUT_DIR / "questions.txt", "w", encoding="utf-8") as out:
    for q in all_questions:
        out.write(f"Question {q['number']} (Page {q['page']})\n")
        if q['stem']:
            out.write(q['stem'] + '\n')
        for j, opt in enumerate(q['options']):
            out.write(opt + '\n')
            if j < len(q['options']) - 1:
                out.write('\n')

# ---- debug.txt ----
with open(OUTPUT_DIR / "debug.txt", "w", encoding="utf-8") as out:
    out.write('\n'.join(debug_lines))

# ---- summary ----
print(f"✅ Done.")
print(f"   Questions : {len(all_questions):>3} → {OUTPUT_DIR/'questions.txt'}")
print(f"   Figures   : {len(saved_keys):>3} → {IMG_DIR}/")
print(f"   Debug     : {OUTPUT_DIR/'debug.txt'}")

missing = KEEP_IMAGES - saved_keys
if missing:
    print(f"\n⚠️  {len(missing)} requested image(s) NOT found:")
    for p, i in sorted(missing):
        print(f"     page {p} img_{i}")