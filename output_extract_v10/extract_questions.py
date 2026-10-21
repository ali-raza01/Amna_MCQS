#!/usr/bin/env python3
"""
Extract question text (options only, no answers/rationales) from mcqs_1.pdf pages 17+.

Requirements (with uv):
    uv add pymupdf pytesseract pillow
System: tesseract on PATH  (brew install tesseract)

Run:
    uv run extract_questions.py
    uv run extract_questions.py --start-page 17 --end-page 30 --dpi 200   # quick test

Outputs:
    questions_clean.txt   – extracted questions
    debug_full_text.txt   – raw OCR/parsed text (for tuning)
"""

import argparse
import io
import re
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

try:
    import pytesseract
except ImportError:
    sys.exit("Install pytesseract: uv add pytesseract")

# ---------------------------------------------------------------- defaults
PDF_PATH         = Path("mcqs_1.pdf")
START_PAGE       = 17
END_PAGE         = None       # None → to end of PDF
DPI              = 300
MIN_TEXT_CHARS   = 60         # below this, fall back to OCR
TESS_PSM         = 6          # 6 = "uniform block of text" (best for Q&A lists)
OUTPUT_FILE      = Path("questions_clean.txt")
CACHE_FILE       = Path("debug_full_text.txt")

# ---------------------------------------------------------------- regex
# Question header, tolerates "# Question 1", "Question 1", "Question1" etc.
QUESTION_HEADER = re.compile(
    r'^\s*#{0,6}\s*Question\s*(\d+)\b',
    re.IGNORECASE | re.MULTILINE,
)

# Anything from these markers onward is not part of the question.
END_MARKERS = [
    re.compile(r'\n\s*#{0,6}\s*Correct\s*Answer\s*[\s:.\-]', re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*Rationale\s*[\s:.\-]',     re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*Note\s+for\s+the\s+other',  re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*NOTE\s*[\s:.\-]',           re.IGNORECASE),
]


# ---------------------------------------------------------------- helpers
def is_text_layer_usable(text: str) -> bool:
    """Heuristic: real text layer vs. just a few numbers / bookmarks."""
    if not text:
        return False
    compact = re.sub(r'\s+', '', text)
    if len(compact) < MIN_TEXT_CHARS:
        return False
    return bool(re.search(r'[A-Za-z]{10,}', text))


def strip_page_artifacts(text: str) -> str:
    """Remove the running page number / header at the top of each page."""
    # "13\n" at start
    text = re.sub(r'^\s*\d{1,4}\s*\n', '', text, count=1)
    # "PPTE Review & Study Guide\n"
    text = re.sub(
        r'^\s*PPTE\s+Review\s*&\s*Study\s+Guide\s*\n',
        '', text, count=1, flags=re.IGNORECASE,
    )
    # Possible second leading number
    text = re.sub(r'^\s*\d{1,4}\s*\n', '', text, count=1)
    return text


def get_page_text(page: "fitz.Page", dpi: int, psm: int) -> tuple[str, str]:
    """Return (text, source_label) — tries text layer, else OCR."""
    text = page.get_text("text")
    if is_text_layer_usable(text):
        return text, "text"
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    ocr = pytesseract.image_to_string(
        img, lang="eng",
        config=f"--oem 1 --psm {psm}",
    )
    return ocr, "ocr"


def cut_question_text(chunk: str) -> str:
    """Trim at the earliest of Correct Answer / Rationale / Note."""
    earliest = len(chunk)
    for pat in END_MARKERS:
        m = pat.search(chunk)
        if m and m.start() < earliest:
            earliest = m.start()
    return chunk[:earliest]


def normalize(text: str) -> str:
    """Clean up markdown / whitespace artifacts from OCR."""
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)   # ## headings
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)          # **bold**
    text = re.sub(r'[ \t]+', ' ', text)                         # horizontal ws
    text = re.sub(r'\n{3,}', '\n\n', text)                      # blank lines
    return text.strip()


def parse_questions(full_text: str) -> list[dict]:
    """Split into questions by 'Question N' headers, then trim each."""
    matches = list(QUESTION_HEADER.finditer(full_text))
    out = []
    for i, m in enumerate(matches):
        num   = int(m.group(1))
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        chunk = cut_question_text(full_text[start:end])
        chunk = normalize(chunk)
        if len(chunk) >= 15:            # skip OCR noise
            out.append({"number": num, "text": chunk})
    return out


# ---------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf",        type=Path, default=PDF_PATH)
    p.add_argument("--start-page", type=int,  default=START_PAGE)
    p.add_argument("--end-page",   type=int,  default=None)
    p.add_argument("--dpi",        type=int,  default=DPI)
    p.add_argument("--psm",        type=int,  default=TESS_PSM)
    p.add_argument("--use-cache",  action="store_true",
                   help="Reuse debug_full_text.txt instead of re-OCR")
    return p.parse_args()


def main():
    a = parse_args()
    if not a.pdf.exists():
        sys.exit(f"PDF not found: {a.pdf}")

    # ------------------------------------------------------- cached path
    if a.use_cache and CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 0:
        print(f"Using cached text from {CACHE_FILE}")
        full_text = CACHE_FILE.read_text(encoding="utf-8")
    else:
        doc = fitz.open(a.pdf)
        end = a.end_page if a.end_page else len(doc)
        end = min(end, len(doc))
        total = end - (a.start_page - 1)
        print(f"Processing {total} pages ({a.start_page}–{end}) at {a.dpi} DPI "
              f"(psm={a.psm})\n")

        page_texts = []
        t0 = time.time()
        for idx in range(a.start_page - 1, end):
            page = doc[idx]
            text, source = get_page_text(page, a.dpi, a.psm)
            text = strip_page_artifacts(text)
            page_texts.append(text)

            done  = idx - (a.start_page - 1) + 1
            elapsed = time.time() - t0
            eta = (elapsed / done) * (total - done) if done else 0
            print(f"  Page {idx + 1:>4} [{source:>4}]  "
                  f"{len(text):>5} chars   ({done}/{total}, "
                  f"ETA {eta/60:.1f} min)")

        full_text = "\n\n".join(page_texts)
        CACHE_FILE.write_text(full_text, encoding="utf-8")

    # ------------------------------------------------------- parse & write
    questions = parse_questions(full_text)

    lines = []
    for q in questions:
        lines.append(f"=== Question {q['number']} ===")
        lines.append(q["text"])
        lines.append("")
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n✓ {len(questions)} questions  →  {OUTPUT_FILE}")
    print(f"✓ Raw text kept in         →  {CACHE_FILE}")
    if questions:
        print(f"\n--- Question {questions[0]['number']} preview ---")
        print(questions[0]["text"][:500])


if __name__ == "__main__":
    main()