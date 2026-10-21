#!/usr/bin/env python3
"""
Extract questions from mcqs_1.pdf with explicit page-range section assignment.

Fixes vs. v4:
  • Sections come from a hard-coded config (your page ranges), not header OCR
  • Per-section validation: expected count vs found, with missing/duplicate numbers
  • Global sequential numbering; chapter-local number kept as metadata
"""

import argparse, bisect, io, re, sys, time
from pathlib import Path

import fitz
from PIL import Image
import pytesseract

# ---------------- defaults ----------------
PDF_PATH      = Path("mcqs_1.pdf")
START_PAGE    = 16          # cover page of Anatomy — no questions, but included
DPI           = 300
TESS_PSM      = 6
PAGE_SENTINEL = "\n<<<PAGE_BREAK>>>\n"

OUTPUT_FILE   = Path("questions_clean.txt")
REPORT_FILE   = Path("sections_report.txt")
CACHE_FILE    = Path("debug_raw_pages.txt")
CLEANED_FILE  = Path("debug_parsed.txt")

# ---------------- SECTIONS (your page ranges) ----------------
# (start_page, end_page, label, expected_question_count)
SECTIONS = [
    ( 16,  50, "1.A Anatomy",                                    116),
    ( 51,  85, "1.B Biomechanics",                               100),
    ( 86, 158, "1.C Orthopedic",                                 277),
    (159, 180, "1.D Sports Medicine",                             64),
    (181, 239, "2 Neuromuscular System",                         248),
    (240, 297, "3 Cardiovascular and Pulmonary Systems",         189),
    (298, 320, "4.A Paediatric",                                  85),
    (321, 335, "4.B Endocrine",                                   41),
    (336, 340, "4.C Burn",                                        16),
    (341, 354, "5.A Therapeutic Exercises",                       34),
    (355, 401, "5.B Electro & Hydro therapy",                    149),
    (402, 435, "6.A Quiz Bank",                                   34),
    (436, 452, "6.B Knowledge",                                   81),
    (453, 457, "6.C Figures",                                     25),
]

def resolve_section(page: int) -> str:
    for s, e, label, _ in SECTIONS:
        if s <= page <= e:
            return label
    return f"Unknown (p.{page})"

def expected_count(label: str) -> int:
    for _, _, lbl, n in SECTIONS:
        if lbl == label:
            return n
    return 0

# ---------------- regex ----------------
QUESTION_HEADER = re.compile(r'^\s*#{0,6}\s*Question\s*(\d+)\b', re.MULTILINE | re.IGNORECASE)
END_MARKERS = [
    re.compile(r'\n\s*#{0,6}\s*Correct\s*Answer\b', re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*Rationale\b',        re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*Note\s+for\b',       re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*NOTE\b',             re.IGNORECASE),
]
STRICT_OPTION = re.compile(r'^\s*[1-4][.)\-:]\s+[A-Za-z]', re.MULTILINE)
INLINE_OPTION = re.compile(r'(?:^|\s)[1-4][.)\-:]\s+[A-Za-z]')

# ---------------- helpers ----------------
def count_options(t):
    return max(len(STRICT_OPTION.findall(t)), len(INLINE_OPTION.findall(t)))

def is_text_layer_usable(t):
    if not t: return False
    if len(re.findall(r'[A-Za-z]{3,}', t)) < 20: return False
    h = len(QUESTION_HEADER.findall(t)); o = count_options(t)
    return o >= (h * 2 if h else 2)

def strip_page_artifacts(t):
    t = re.sub(r'^[^\n]*PPTE\s+Review\s*&\s*Study\s+Guide[^\n]*\n?', '', t, count=1, flags=re.IGNORECASE)
    t = re.sub(r'^\s*\d{1,4}\s*\n', '', t, count=1)
    t = re.sub(r'^[^\w\n]{1,5}\s*(?=\n)', '', t, count=1)
    return t

def clean_page(t):
    t = re.sub(r'^.{0,6}\bQuestion\s+(\d+)\b', r'Question \1', t, flags=re.MULTILINE)
    t = re.sub(r'^[^\n]*PPTE\s+Review\s*&\s*Study\s+Guide[^\n]*$', '', t,
               flags=re.MULTILINE | re.IGNORECASE)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t

def cache_is_valid(cache_file: Path) -> bool:
    if not cache_file.exists() or cache_file.stat().st_size == 0:
        return False
    sample = cache_file.read_text(encoding="utf-8")[:4000]
    return (PAGE_SENTINEL.strip() in sample
            and re.search(r'Chapter\s+\d+|Musculoskeletal|Neuromuscular', sample) is not None)

# ---------------- extraction ----------------
def extract_pages(pdf_path, start, end, dpi, psm, force_ocr, cache_file):
    if not force_ocr and cache_is_valid(cache_file):
        print(f"Using cached raw pages from {cache_file}")
        raw_pages = cache_file.read_text(encoding="utf-8").split(PAGE_SENTINEL.strip())
    else:
        if cache_file.exists():
            print("⚠ cache missing/invalid → re-extracting")
        doc   = fitz.open(pdf_path)
        end   = min(end, len(doc))
        total = end - start + 1
        print(f"Extracting {total} pages @ {dpi} DPI (psm={psm}, force_ocr={force_ocr})")
        raw_pages, t0 = [], time.time()
        for i in range(start - 1, end):
            page = doc[i]
            tl = page.get_text("text")
            if force_ocr or not is_text_layer_usable(tl):
                pix = page.get_pixmap(dpi=dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, lang="eng",
                                                   config=f"--oem 1 --psm {psm}")
                src = "ocr"
            else:
                text, src = tl, "text"
            raw_pages.append(strip_page_artifacts(text))
            done = i - (start - 1) + 1
            eta  = (time.time() - t0) / done * (total - done)
            print(f"  p{i+1:>4} [{src:>4}] {len(text):>5} chars "
                  f"({done}/{total}, ETA {eta/60:.1f}min)")
        cache_file.write_text(PAGE_SENTINEL.join(raw_pages), encoding="utf-8")

    records = []
    for i, raw in enumerate(raw_pages):
        records.append({
            "page":    start + i,
            "text":    clean_page(raw),
            "section": resolve_section(start + i),
        })
    return records

# ---------------- parsing ----------------
def cut_question_text(chunk):
    earliest = len(chunk)
    for pat in END_MARKERS:
        m = pat.search(chunk)
        if m and m.start() < earliest:
            earliest = m.start()
    return chunk[:earliest]

def normalize(t):
    t = re.sub(r'^#{1,6}\s*', '', t, flags=re.MULTILINE)
    t = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', t)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{2,}', '\n', t)
    return t.strip()

def is_junk(body):
    if len(body) < 25: return True
    if len(re.findall(r'[A-Za-z]{3,}', body)) < 4: return True
    if re.search(r'PPTE\s+Review', body, re.IGNORECASE): return True
    return False

def parse_questions(pages):
    parts, page_meta, pos = [], [], 0
    for p in pages:
        page_meta.append((p["page"], pos, p["section"]))
        parts.append(p["text"])
        pos += len(p["text"]) + 2
    joined  = "\n\n".join(parts)
    offsets = [m[1] for m in page_meta]

    def page_at(off):
        i = bisect.bisect_right(offsets, off) - 1
        return page_meta[max(0, i)]

    matches, out = list(QUESTION_HEADER.finditer(joined)), []
    for i, m in enumerate(matches):
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(joined)
        body  = normalize(cut_question_text(joined[start:end]))
        if is_junk(body):
            continue
        p_start = page_at(start)
        p_end   = page_at(max(start, end - 1))
        out.append({
            "seq":        len(out) + 1,
            "orig_num":   int(m.group(1)),
            "start_page": p_start[0],
            "end_page":   p_end[0],
            "section":    p_start[2],
            "text":       body,
        })
    return out, joined

# ---------------- output & report ----------------
def write_output(questions, joined, extracted_range):
    # ---------- questions_clean.txt ----------
    lines = []
    for q in questions:
        pg = f"p.{q['start_page']}" if q["start_page"] == q["end_page"] \
             else f"p.{q['start_page']}-{q['end_page']}"
        lines.append(f"=== Q{q['seq']} | {pg} | {q['section']} ===")
        lines.append(q["text"])
        lines.append("")
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    CLEANED_FILE.write_text(joined, encoding="utf-8")

    # ---------- sections_report.txt ----------
    lo, hi = extracted_range
    by_section = {}
    for q in questions:
        by_section.setdefault(q["section"], []).append(q)

    rep = [
        "SECTION VALIDATION REPORT",
        "=" * 74,
        f"Extracted pages: {lo} – {hi}",
        "",
    ]
    total_found, total_expected = 0, 0
    for s, e, label, expected in SECTIONS:
        in_range = not (e < lo or s > hi)
        qs       = by_section.get(label, [])
        nums     = [q["orig_num"] for q in qs]
        seq_rng  = f"Q{qs[0]['seq']}–Q{qs[-1]['seq']}" if qs else "—"
        status   = "✓" if len(qs) == expected else ("!" if in_range else "·")

        rep.append(f"[{status}] {label}")
        rep.append(f"     Pages    : {s}–{e}")
        rep.append(f"     Expected : {expected}")
        rep.append(f"     Found    : {len(qs)}"
                   + (f"   (seq {seq_rng})" if qs else ""))

        if in_range:
            total_found    += len(qs)
            total_expected += expected

            if qs:
                present = set(nums)
                missing = sorted(set(range(1, expected + 1)) - present)
                dups    = sorted(n for n in present if nums.count(n) > 1)
                if missing:
                    preview = ', '.join(map(str, missing[:25]))
                    more    = f" … (+{len(missing)-25} more)" if len(missing) > 25 else ""
                    rep.append(f"     ⚠ Missing : {preview}{more}")
                if dups:
                    rep.append(f"     ⚠ Dups    : {', '.join(map(str, dups))}")
            else:
                rep.append(f"     ⚠ NO QUESTIONS EXTRACTED FROM THIS RANGE")
        else:
            rep.append(f"     (outside extracted page range)")
        rep.append("")

    rep.append("=" * 74)
    pct = 100 * total_found / total_expected if total_expected else 0
    rep.append(f"TOTAL  : {total_found} / {total_expected}  ({pct:.1f}%)")
    REPORT_FILE.write_text("\n".join(rep), encoding="utf-8")

    # ---------- console summary ----------
    print(f"\n✓ {len(questions)} questions → {OUTPUT_FILE}")
    print(f"✓ Section report      → {REPORT_FILE}")
    print(f"✓ Parsed text         → {CLEANED_FILE}\n")
    print(f"  {'Section':<42} {'Exp':>4} {'Got':>4}  Status")
    print(f"  {'-'*42} {'-'*4} {'-'*4}  {'-'*6}")
    for s, e, label, expected in SECTIONS:
        in_range = not (e < lo or s > hi)
        qs       = by_section.get(label, [])
        got      = len(qs)
        if not in_range:
            mark = "  —"
        elif got == expected:
            mark = "  ✓"
        elif got == 0:
            mark = "  !!"
        else:
            mark = f"  ! -{expected-got}"
        shown = f"{label[:40]:<42}"
        print(f"  {shown} {expected:>4} {got:>4}{mark}")

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf",            type=Path, default=PDF_PATH)
    p.add_argument("--start-page",     type=int,  default=START_PAGE)
    p.add_argument("--end-page",       type=int,  default=None)
    p.add_argument("--dpi",            type=int,  default=DPI)
    p.add_argument("--psm",            type=int,  default=TESS_PSM)
    p.add_argument("--force-ocr",      action="store_true")
    p.add_argument("--use-cache",      action="store_true")
    p.add_argument("--diagnose-seq",   type=int,  default=None)
    p.add_argument("--diagnose-page",  type=int,  default=None,
                   help="Print raw OCR for a given PDF page")
    return p.parse_args()

def main():
    a = parse_args()
    if not a.pdf.exists(): sys.exit(f"PDF not found: {a.pdf}")
    if a.use_cache and not cache_is_valid(CACHE_FILE):
        sys.exit(f"--use-cache requested but {CACHE_FILE} is missing or pre-v4 format")

    end = a.end_page or len(fitz.open(a.pdf))
    pages = extract_pages(a.pdf, a.start_page, end, a.dpi, a.psm,
                          a.force_ocr or not a.use_cache, CACHE_FILE)

    if a.diagnose_page is not None:
        p = next((p for p in pages if p["page"] == a.diagnose_page), None)
        if not p:
            sys.exit(f"Page {a.diagnose_page} not in extracted range")
        print(f"\n=== Raw OCR of page {a.diagnose_page} "
              f"[{p['section']}] ===\n{p['text']}")
        return

    qs, joined = parse_questions(pages)

    if a.diagnose_seq is not None:
        q = next((q for q in qs if q["seq"] == a.diagnose_seq), None)
        if not q: sys.exit(f"No seq={a.diagnose_seq}")
        print(f"\n=== Q{q['seq']} | orig {q['orig_num']} | "
              f"p.{q['start_page']} | {q['section']} ===\n{q['text']}")
        return

    write_output(qs, joined, (a.start_page, end))

if __name__ == "__main__":
    main()