import pymupdf
import pytesseract
from PIL import Image
import io, re, hashlib
from pathlib import Path

# ============ CONFIG ============
PDF_PATH = "mcqs_1.pdf"
OUTPUT_DIR = Path("output")
START_PAGE = 17
END_PAGE = None              # None = to end; set 20 for a quick test
DPI = 300
PSM = 4

# Image filtering
MIN_FIGURE_DIM = 80
MIN_FIGURE_CHARS = 100       # embedded image with MORE OCR chars than this → skip
FORCE_KEEP = {(17, 1)}       # (page_number, image_index) → always keep, ignore heuristic
FORCE_SKIP = set()           # (page_number, image_index) → always skip

# ============ PATTERNS ============
ANSWER_LINE = re.compile(
    r'(?i)^\s*(?:correct\s+)?(?:answer|ans|rationale|explanation|solution|justification)\s*[:\-–]'
)
QUESTION_MARKER = re.compile(r'^\s*\d{1,2}\s*$')    # lone 1-2 digit line = question separator
CHAPTER_LINE    = re.compile(r'(?i)^\s*chapter\s+\d+\s*$')
OPTION_RES = [
    re.compile(r'^\s*(\d+)\s*[\.\)]\s+(.+)$'),          # 1. text   1) text
    re.compile(r'^\s*([A-Da-d])\s*[\.\)]\s+(.+)$'),     # A. text   a) text
    re.compile(r'^\s*([ivxIVX]{1,4})\s*[\.\)]\s+(.+)$'),# i. text   iv) text
]

def is_option(s):
    return any(r.match(s) for r in OPTION_RES)

def looks_like_header(s):
    if CHAPTER_LINE.match(s):
        return True
    if re.search(r'(?i)\b(biomechanics|systems?)\b', s) and '-' in s and len(s) > 20:
        return True
    return False

# ============ PARSING ============
def parse_page_ocr(text):
    questions = []
    stem, opts = [], []
    in_opts = False
    in_answer_mode = False
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

        # Drop leading header(s) before first real content
        if not first_content_seen:
            if QUESTION_MARKER.match(s) or is_option(s) or s.endswith((':', '?', '؟')):
                first_content_seen = True
            else:
                continue

        if looks_like_header(s):
            continue

        # Question separator
        if QUESTION_MARKER.match(s):
            flush()
            in_answer_mode = False
            continue

        # Enter answer/rationale skip mode
        if ANSWER_LINE.match(s):
            in_answer_mode = True
            continue
        if in_answer_mode:
            continue

        if is_option(s):
            opts.append(s)
            in_opts = True
        else:
            if in_opts and s[0].islower():
                # Continuation of previous option (e.g. "consumption")
                opts[-1] = opts[-1] + ' ' + s
            else:
                if in_opts:
                    flush()
                stem.append(s)

    flush()
    return questions

# ============ IMAGES ============
def signature(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((16, 16))
        px = list(img.getdata())
        avg = sum(px) / len(px)
        bits = "".join("1" if p > avg else "0" for p in px)
        return "ah_" + hashlib.sha1(bits.encode()).hexdigest()[:16]
    except Exception:
        return "bh_" + hashlib.sha1(img_bytes).hexdigest()

def classify_image(img_bytes, dims):
    w, h = dims
    if w < MIN_FIGURE_DIM or h < MIN_FIGURE_DIM:
        return False, f"too small ({w}x{h})"
    try:
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        chars = len(pytesseract.image_to_string(img).strip())
    except Exception as e:
        return False, f"error: {e}"
    if chars < MIN_FIGURE_CHARS:
        return True, f"figure ({chars} chars, {w}x{h})"
    return False, f"text-image ({chars} chars, {w}x{h})"

# ============ MAIN ============
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

doc = pymupdf.open(PDF_PATH)
start_idx = START_PAGE - 1
end_idx = END_PAGE if END_PAGE else len(doc)

all_questions = []
debug = []
saved_sig = {}
q_counter = 0

for page_idx in range(start_idx, end_idx):
    page = doc[page_idx]
    page_num = page_idx + 1

    pix = page.get_pixmap(dpi=DPI)
    page_img = Image.open(io.BytesIO(pix.tobytes("png")))
    ocr_text = pytesseract.image_to_string(page_img, config=f"--psm {PSM}")

    page_qs = parse_page_ocr(ocr_text)
    for q in page_qs:
        q_counter += 1
        q['number'] = q_counter
        q['page'] = page_num
        all_questions.append(q)

    debug.append(f"\n=== PAGE {page_num} ===")
    debug.append(f"parsed {len(page_qs)} question(s)")
    for i, q in enumerate(page_qs, 1):
        debug.append(f"  Q{i} stem: {q['stem'][:80]!r}")
        for opt in q['options']:
            debug.append(f"     {opt}")

    for i, info in enumerate(page.get_images(full=True)):
        try:
            base = doc.extract_image(info[0])
        except Exception as e:
            debug.append(f"  img[{i}] extract error: {e}")
            continue
        dims = (base['width'], base['height'])
        sig = signature(base["image"])

        if (page_num, i) in FORCE_SKIP:
            debug.append(f"  img[{i}] SKIP (force)")
            continue
        force_keep = (page_num, i) in FORCE_KEEP
        if not force_keep:
            if sig in saved_sig:
                debug.append(f"  img[{i}] SKIP (dup of {saved_sig[sig]})")
                continue
            keep, reason = classify_image(base["image"], dims)
            if not keep:
                debug.append(f"  img[{i}] SKIP — {reason}")
                continue
        else:
            reason = f"FORCE KEEP ({dims[0]}x{dims[1]})"

        fname = f"page_{page_num:03d}_img_{i}.{base['ext']}"
        (FIG_DIR / fname).write_bytes(base["image"])
        saved_sig[sig] = fname
        debug.append(f"  img[{i}] KEPT {fname} — {reason}")

doc.close()

# Primary output: one question per block, blank line between options
with open(OUTPUT_DIR / "questions.txt", "w", encoding="utf-8") as out:
    for q in all_questions:
        out.write(f"Question {q['number']}\n")
        if q['stem']:
            out.write(q['stem'] + '\n')
        for j, opt in enumerate(q['options']):
            out.write(opt + '\n')
            if j < len(q['options']) - 1:
                out.write('\n')

with open(OUTPUT_DIR / "debug.txt", "w", encoding="utf-8") as out:
    out.write('\n'.join(debug))

print(f"✅ Done.")
print(f"   Questions : {OUTPUT_DIR/'questions.txt'}  ({len(all_questions)} total)")
print(f"   Debug     : {OUTPUT_DIR/'debug.txt'}")
print(f"   Figures   : {FIG_DIR}/")