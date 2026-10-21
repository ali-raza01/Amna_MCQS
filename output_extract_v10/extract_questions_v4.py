#!/usr/bin/env python3
"""
Extract questions from mcqs_1.pdf with:
  • Global sequential numbering (never dedups by OCR'd number — chapters restart at Q1)
  • Per-question section label (from page running header / sub-section title)
  • Per-question start & end page
  • A sections_report.txt with page ranges and numbering gaps
  • Permissive option detection (handles '1. foo' and '1- foo 2- bar')
"""

import argparse, bisect, io, re, sys, time
from pathlib import Path

import fitz
from PIL import Image
import pytesseract

# ---------------- defaults ----------------
PDF_PATH      = Path("mcqs_1.pdf")
START_PAGE    = 17
DPI           = 300
TESS_PSM      = 6
PAGE_SENTINEL = "\n<<<PAGE_BREAK>>>\n"

OUTPUT_FILE   = Path("questions_clean.txt")
REPORT_FILE   = Path("sections_report.txt")
CACHE_FILE    = Path("debug_raw_pages.txt")
CLEANED_FILE  = Path("debug_parsed.txt")

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

CHAPTER_RE       = re.compile(r'^\s*Chapter\s+(\d+)\s*(.*?)\s*$', re.MULTILINE | re.IGNORECASE)
SUBSECTION_RE    = re.compile(r'^\s*(\d+)\s*\.\s*([A-Z])\s*\.?\s+([A-Za-z][^\n]{0,60}?)\s*$', re.MULTILINE)
NAMED_SECTION_RE = re.compile(
    r'^\s*((?:Therapeutic\s+Modalities|VIP|Others)\s*-\s*[A-Za-z][^\n]{0,60}?)\s*$',
    re.MULTILINE)
SYSTEM_NAME_RE   = re.compile(
    r'^\s*((?:[A-Z][A-Za-z]+\s+)*[A-Z][A-Za-z]+)\s+Systems?\s*$', re.MULTILINE)


# ---------------- Section tracking ----------------
class SectionTracker:
    def __init__(self):
        self.chapter_key  = ""
        self.chapter_name = ""
        self.subsection   = ""

    def update(self, head: str) -> None:
        # "Chapter N [Name]"
        m = CHAPTER_RE.search(head)
        if m:
            ch, name = int(m.group(1)), m.group(2).strip()
            key = f"Ch{ch}"
            if key != self.chapter_key:
                self.chapter_key, self.chapter_name, self.subsection = key, name, ""
            elif name and not self.chapter_name:
                self.chapter_name = name
        # "<Word(s)> System" fallback (Musculoskeletal / Neuromuscular / …)
        if not self.chapter_name:
            m = SYSTEM_NAME_RE.search(head)
            if m:
                self.chapter_name = m.group(1).strip()
        # Numbered sub-section "1.A Anatomy", "1.D Sports Medicine"
        m = SUBSECTION_RE.search(head)
        if m:
            self.subsection = f"{m.group(1)}.{m.group(2)} {m.group(3).strip()}"
        # Named section "Others - X", "VIP - X", "Therapeutic Modalities - X"
        m = NAMED_SECTION_RE.search(head)
        if m:
            self.chapter_name = m.group(1).strip()

    def label(self) -> str:
        parts = []
        if self.chapter_key:
            parts.append((self.chapter_key + " " + self.chapter_name).strip())
        elif self.chapter_name:
            parts.append(self.chapter_name)
        if self.subsection:
            parts.append(self.subsection)
        return " | ".join(parts) if parts else "Unknown"


# ---------------- Extraction helpers ----------------
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
    # Line-based cleanup only — never touch page boundaries
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


# ---------------- Extract pages ----------------
def extract_pages(pdf_path, start, end, dpi, psm, force_ocr, cache_file):
    if not force_ocr and cache_is_valid(cache_file):
        print(f"Using cached raw pages from {cache_file}")
        raw_pages = cache_file.read_text(encoding="utf-8").split(PAGE_SENTINEL.strip())
    else:
        if cache_file.exists():
            print(f"⚠ cache missing/invalid (likely pre-v4) → re-extracting")
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

    # Now build per-page records with section labels
    tracker, records = SectionTracker(), []
    for i, raw in enumerate(raw_pages):
        head = '\n'.join(raw.split('\n')[:15])
        tracker.update(head)
        records.append({"page": start + i, "text": clean_page(raw),
                        "section": tracker.label()})
    return records


# ---------------- Parse ----------------
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
    joined_parts, page_meta, pos = [], [], 0
    for p in pages:
        page_meta.append((p["page"], pos, p["section"]))
        joined_parts.append(p["text"])
        pos += len(p["text"]) + 2
    joined  = "\n\n".join(joined_parts)
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


# ---------------- Report ----------------
def write_output(questions, joined):
    lines = []
    for q in questions:
        pg = f"p.{q['start_page']}" if q["start_page"] == q["end_page"] \
             else f"p.{q['start_page']}-{q['end_page']}"
        lines.append(f"=== Q{q['seq']} | {pg} | {q['section']} ===")
        lines.append(q["text"])
        lines.append("")
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    CLEANED_FILE.write_text(joined, encoding="utf-8")

    sections = {}
    for q in questions:
        sections.setdefault(q["section"], []).append(q)

    rep = ["SECTION REPORT", "=" * 60, ""]
    for sec, qs in sections.items():
        pgs  = sorted(set(q["start_page"] for q in qs))
        nums = sorted(q["orig_num"] for q in qs)
        miss = sorted(set(range(1, max(nums) + 1)) - set(nums))
        rep.append(f"Section: {sec}")
        rep.append(f"  Pages     : {min(pgs)} – {max(pgs)}")
        rep.append(f"  Questions : {len(qs)} (numbered {min(nums)}–{max(nums)})")
        rep.append(f"  Seq range : Q{qs[0]['seq']} – Q{qs[-1]['seq']}")
        if miss:
            rep.append(f"  ⚠ Missing : {', '.join(map(str, miss))}")
        rep.append("")
    REPORT_FILE.write_text("\n".join(rep), encoding="utf-8")

    print(f"\n✓ {len(questions)} questions → {OUTPUT_FILE}")
    print(f"✓ Section report      → {REPORT_FILE}")
    print(f"✓ Parsed text         → {CLEANED_FILE}\n")
    print(f"  seq  | orig |  pages   | opts | section")
    print(f"  -----+------+----------+------+----------------------------")
    for q in questions:
        n  = count_options(q["text"])
        pg = f"{q['start_page']}" if q["start_page"] == q["end_page"] \
             else f"{q['start_page']}-{q['end_page']}"
        print(f"  {q['seq']:>4} | {q['orig_num']:>4} | {pg:>8} | {n:>4}"
              f"{' ⚠' if n < 2 else '  '} | {q['section']}")


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
    return p.parse_args()

def main():
    a = parse_args()
    if not a.pdf.exists(): sys.exit(f"PDF not found: {a.pdf}")
    if a.use_cache and not cache_is_valid(CACHE_FILE):
        sys.exit(f"--use-cache requested but {CACHE_FILE} is missing or pre-v4 format")
    end     = a.end_page or len(fitz.open(a.pdf))
    pages   = extract_pages(a.pdf, a.start_page, end, a.dpi, a.psm,
                            a.force_ocr or not a.use_cache, CACHE_FILE)
    qs, joined = parse_questions(pages)

    if a.diagnose_seq:
        q = next((q for q in qs if q["seq"] == a.diagnose_seq), None)
        if not q: sys.exit(f"No seq={a.diagnose_seq}")
        print(f"\n=== Q{q['seq']} | orig {q['orig_num']} | "
              f"p.{q['start_page']} | {q['section']} ===\n{q['text']}")
        return
    write_output(qs, joined)

if __name__ == "__main__":
    main()