#!/usr/bin/env python3
"""
Extract question text (options only, no answers/rationales) from mcqs_1.pdf.

Key fix vs. v2:
  • Sequential global numbering (chapters restart at Q1; old dedup lost ~75 %).
  • Permissive option regex (handles "1- a 2- b 3- c" inline format).
  • Raw per-page OCR cache, sentinel-joined (--use-cache is instant).
  • Junk-body filter drops page-header remnants.

Requirements:
    uv add pymupdf pytesseract pillow
    (tesseract on PATH)
"""

import argparse
import io
import re
import sys
import time
from pathlib import Path

import fitz
from PIL import Image
import pytesseract

# ---------------------------------------------------------------- defaults
PDF_PATH       = Path("mcqs_1.pdf")
START_PAGE     = 17
END_PAGE       = None
DPI            = 300
TESS_PSM       = 6
OUTPUT_FILE    = Path("questions_clean.txt")
CACHE_FILE     = Path("debug_raw_pages.txt")   # NEW sentinel format
CLEANED_FILE   = Path("debug_parsed.txt")

PAGE_SENTINEL  = "\n<<<PAGE_BREAK>>>\n"

# ---------------------------------------------------------------- regex
QUESTION_HEADER = re.compile(
    r'^\s*#{0,6}\s*Question\s*(\d+)\b',
    re.MULTILINE | re.IGNORECASE,
)

END_MARKERS = [
    re.compile(r'\n\s*#{0,6}\s*Correct\s*Answer\b', re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*Rationale\b',        re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*Note\s+for\b',       re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*NOTE\b',             re.IGNORECASE),
]

# strict: option at line start, e.g. "1. foo"  (Anatomy style)
STRICT_OPTION = re.compile(r'^\s*[1-4][.)\-:]\s+[A-Za-z]', re.MULTILINE)
# permissive: option anywhere, e.g. "… 2- bar 3- baz" (Neuro style)
INLINE_OPTION = re.compile(r'(?:^|\s)[1-4][.)\-:]\s+[A-Za-z]')


# ---------------------------------------------------------------- helpers
def count_options(text: str) -> int:
    return max(len(STRICT_OPTION.findall(text)),
               len(INLINE_OPTION.findall(text)))


def is_text_layer_usable(text: str) -> bool:
    """Reject sparse text layers (e.g. page 17) so OCR kicks in."""
    if not text:
        return False
    if len(re.findall(r'[A-Za-z]{3,}', text)) < 20:
        return False
    headers = len(QUESTION_HEADER.findall(text))
    opts    = count_options(text)
    if headers == 0:
        return opts >= 2
    return opts >= headers * 2


def strip_page_artifacts(text: str) -> str:
    """Remove the running header, leading page numbers, and noise prefixes."""
    text = re.sub(r'^[^\n]*PPTE\s+Review\s*&\s*Study\s+Guide[^\n]*\n?',
                  '', text, count=1, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d{1,4}\s*\n', '', text, count=1)
    text = re.sub(r'^[^\w\n]{1,5}\s*(?=\n)', '', text, count=1)
    return text


def clean_full_text(text: str) -> str:
    """Global cleanup that must run AFTER pages are joined."""
    # OCR noise prefixes before 'Question N' ("t Q", "¢ Q", "at d Q", …)
    text = re.sub(r'^.{0,6}\bQuestion\s+(\d+)\b',
                  r'Question \1', text, flags=re.MULTILINE)
    # Running header anywhere on a line
    text = re.sub(r'^[^\n]*PPTE\s+Review\s*&\s*Study\s+Guide[^\n]*$',
                  '', text, flags=re.MULTILINE | re.IGNORECASE)
    # Chapter section titles
    text = re.sub(r'^[^\n]*Mus[ck]uloskeletal[^\n]*$',
                  '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\s*Chapter\s+\d+\s*$', '',
                  text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def cut_question_text(chunk: str) -> str:
    earliest = len(chunk)
    for pat in END_MARKERS:
        m = pat.search(chunk)
        if m and m.start() < earliest:
            earliest = m.start()
    return chunk[:earliest]


def normalize(text: str) -> str:
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


def is_junk(body: str) -> bool:
    """Drop bodies that are just page-header remnants."""
    if len(body) < 25:
        return True
    if len(re.findall(r'[A-Za-z]{3,}', body)) < 4:
        return True
    if re.search(r'PPTE\s+Review', body, re.IGNORECASE):
        return True
    return False


# ---------------------------------------------------------------- extraction
def extract_pages(pdf_path: Path, start: int, end: int,
                  dpi: int, psm: int, force_ocr: bool,
                  cache_file: Path) -> str:
    """Return raw per-page text separated by PAGE_SENTINEL.
    Cached in `cache_file` so re-parsing never needs OCR again."""
    if cache_file.exists() and cache_file.stat().st_size > 0:
        raw = cache_file.read_text(encoding="utf-8")
        if PAGE_SENTINEL.strip() in raw:
            print(f"Using cached raw pages from {cache_file}")
            return raw
        print(f"⚠  {cache_file} is old format → re-extracting")

    doc   = fitz.open(pdf_path)
    end   = min(end, len(doc))
    total = end - start + 1
    pages, t0 = [], time.time()
    print(f"Extracting {total} pages @ {dpi} DPI "
          f"(psm={psm}, force_ocr={force_ocr})")

    for i in range(start - 1, end):
        page = doc[i]
        tl   = page.get_text("text")
        if force_ocr or not is_text_layer_usable(tl):
            pix  = page.get_pixmap(dpi=dpi)
            img  = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(
                img, lang="eng", config=f"--oem 1 --psm {psm}")
            src = "ocr"
        else:
            text = tl
            src = "text"
        pages.append(text)

        done = i - (start - 1) + 1
        eta  = (time.time() - t0) / done * (total - done)
        print(f"  p{i+1:>4} [{src:>4}]  {len(text):>5} chars  "
              f"({done}/{total}, ETA {eta/60:.1f}min)")

    raw = PAGE_SENTINEL.join(pages)
    cache_file.write_text(raw, encoding="utf-8")
    return raw


def process_raw(raw: str) -> str:
    """Split sentinel-joined pages, strip each, join, clean globally."""
    if PAGE_SENTINEL.strip() in raw:
        pages    = raw.split(PAGE_SENTINEL)
        stripped = [strip_page_artifacts(p) for p in pages]
        full     = "\n\n".join(stripped)
    else:
        full = raw
    return clean_full_text(full)


# ---------------------------------------------------------------- parsing
def parse_questions(full_text: str) -> list[dict]:
    """
    Emit every question in document order.
    Never dedup by OCR'd number — chapters restart at Q1.
    """
    matches, out, seq = list(QUESTION_HEADER.finditer(full_text)), [], 0
    for i, m in enumerate(matches):
        orig = int(m.group(1))
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body  = normalize(cut_question_text(full_text[start:end]))
        if is_junk(body):
            continue
        seq += 1
        out.append({"seq": seq, "orig": orig, "text": body})
    return out


# ---------------------------------------------------------------- CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf",        type=Path, default=PDF_PATH)
    p.add_argument("--start-page", type=int,  default=START_PAGE)
    p.add_argument("--end-page",   type=int,  default=None)
    p.add_argument("--dpi",        type=int,  default=DPI)
    p.add_argument("--psm",        type=int,  default=TESS_PSM)
    p.add_argument("--force-ocr",  action="store_true")
    p.add_argument("--use-cache",  action="store_true",
                   help="Skip OCR, reuse debug_raw_pages.txt")
    p.add_argument("--diagnose",   type=int, default=None,
                   help="Print full body of question <seq> and exit")
    return p.parse_args()


def main():
    a = parse_args()
    if not a.pdf.exists():
        sys.exit(f"PDF not found: {a.pdf}")

    if a.use_cache and not CACHE_FILE.exists():
        sys.exit(f"--use-cache requested but {CACHE_FILE} doesn't exist")

    end = a.end_page or len(fitz.open(a.pdf))

    # 1. Get raw text (cache or OCR)
    raw = extract_pages(a.pdf, a.start_page, end, a.dpi, a.psm,
                        a.force_ocr, CACHE_FILE)

    # 2. Clean & parse
    full_text = process_raw(raw)
    CLEANED_FILE.write_text(full_text, encoding="utf-8")
    questions = parse_questions(full_text)

    # 3. Diagnose (single question)
    if a.diagnose is not None:
        q = next((q for q in questions if q["seq"] == a.diagnose), None)
        if not q:
            sys.exit(f"No question with seq={a.diagnose}")
        print(f"\n=== Q{q['seq']} (orig #{q['orig']}) ===\n{q['text']}")
        return

    # 4. Emit
    out = []
    for q in questions:
        out.append(f"=== Question {q['seq']} ===")
        out.append(q["text"])
        out.append("")
    OUTPUT_FILE.write_text("\n".join(out), encoding="utf-8")

    # 5. Report
    print(f"\n✓ {len(questions)} questions → {OUTPUT_FILE}")
    print(f"✓ Cleaned text        → {CLEANED_FILE}")
    print(f"✓ Raw pages (cached)  → {CACHE_FILE}\n")
    print(f"  seq | orig | chars | options")
    print(f"  ----+------+-------+--------")
    for q in questions:
        n    = count_options(q["text"])
        flag = " ⚠" if n < 2 else ""
        print(f"  {q['seq']:>3} | {q['orig']:>4} | {len(q['text']):>5} | {n}{flag}")

    low = [q for q in questions if count_options(q["text"]) < 2]
    if low:
        print(f"\n⚠ {len(low)} questions have <2 detected options — "
              f"inspect one with e.g. `--diagnose {low[0]['seq']}`")


if __name__ == "__main__":
    main()