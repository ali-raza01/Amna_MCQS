#!/usr/bin/env python3
"""
Extract question text (options only, no answers/rationales) from mcqs_1.pdf pages 17+.

Requirements (with uv):
    uv add pymupdf pytesseract pillow
System: tesseract on PATH  (brew install tesseract)

Run:
    uv run extract_questions.py --start-page 17 --end-page 20          # sanity check
    uv run extract_questions.py --force-ocr                            # bypass text layer
    uv run extract_questions.py --use-cache                            # re-parse only
"""

import argparse
import io
import re
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# ---------------------------------------------------------------- defaults
PDF_PATH       = Path("mcqs_1.pdf")
START_PAGE     = 17
END_PAGE       = None
DPI            = 300
MIN_TEXT_CHARS = 60
TESS_PSM       = 6
OUTPUT_FILE    = Path("questions_clean.txt")
CACHE_FILE     = Path("debug_full_text.txt")

# Question header — deliberately tolerant
QUESTION_HEADER = re.compile(r'^\s*#{0,6}\s*Question\s*(\d+)\b',
                             re.MULTILINE | re.IGNORECASE)

# Bounds of a question: anything from these markers onward is metadata
END_MARKERS = [
    re.compile(r'\n\s*#{0,6}\s*Correct\s*Answer\b', re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*Rationale\b',        re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*Note\s+for\b',       re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*NOTE\b',             re.IGNORECASE),
]


# ---------------------------------------------------------------- quality gate
def is_text_layer_usable(text: str) -> bool:
    """
    Trust the text layer only if it actually contains the multiple-choice
    structure. Otherwise the layer is likely a partial overlay and OCR is
    needed.
    """
    if not text:
        return False
    words = re.findall(r'[A-Za-z]{3,}', text)
    if len(words) < 20:
        return False

    q_headers = re.findall(r'^\s*#{0,3}\s*Question\s*\d+\b',
                           text, re.MULTILINE | re.IGNORECASE)
    # Option lines look like "1. weight lifting", "2. jogging", ...
    opts = re.findall(r'^\s*[1-4]\.\s+[A-Za-z]', text, re.MULTILINE)

    if q_headers:
        # Real questions have ≥3 options each; require ≥2 to be safe.
        return len(opts) >= len(q_headers) * 2
    return len(opts) >= 2


# ---------------------------------------------------------------- page cleanup
def strip_page_artifacts(text: str) -> str:
    """Remove the running header ('PPTE Review & Study Guide'), leading
    page numbers, and OCR-of-page-number garbage like 'y Wd'."""
    # Anywhere in the first line: '16', 'y Wd 16', etc. followed by header
    text = re.sub(
        r'^[^\n]*?PPTE\s+Review\s*&\s*Study\s+Guide[^\n]*\n?',
        '', text, count=1, flags=re.IGNORECASE,
    )
    # Leading bare page number (1–4 digits on its own line)
    text = re.sub(r'^\s*\d{1,4}\s*\n', '', text, count=1)
    # Leading short garbage line like 'y Wd' or '4'
    text = re.sub(r'^[^\w\n]{0,3}\s*(?=\n)', '', text, count=1)
    return text


def get_page_text(page: "fitz.Page", dpi: int, psm: int, force_ocr: bool):
    text = page.get_text("text")
    if not force_ocr and is_text_layer_usable(text):
        return text, "text"
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    ocr = pytesseract.image_to_string(img, lang="eng",
                                      config=f"--oem 1 --psm {psm}")
    return ocr, "ocr"


# ---------------------------------------------------------------- global clean
def clean_full_text(text: str) -> str:
    """
    Fix OCR-side issues that are hard to catch per-page:
      * stray prefix before a 'Question N' header ('t Q', '¢ Q', 'at d Q')
      * leftover PPTE headers mid-document
      * Musculoskeletal / Chapter running titles
    """
    # Strip up to 6 chars of noise immediately before 'Question N'.
    # Case-sensitive on 'Question' (the header always has a capital Q).
    text = re.sub(r'^.{0,6}\bQuestion\s+(\d+)\b',
                  r'Question \1', text, flags=re.MULTILINE)
    # Running header, anywhere
    text = re.sub(
        r'^[^\n]*PPTE\s+Review\s*&\s*Study\s+Guide[^\n]*$',
        '', text, flags=re.MULTILINE | re.IGNORECASE,
    )
    # 'Musculoskeletal Systems - Biomechanics' style chapter headers
    text = re.sub(
        r'^[^\n]*Mus[ck]uloskeletal[^\n]*$', '',
        text, flags=re.MULTILINE | re.IGNORECASE,
    )
    # Bare 'Chapter N' titles
    text = re.sub(r'^\s*Chapter\s+\d+\s*$', '',
                  text, flags=re.MULTILINE | re.IGNORECASE)
    # Collapse blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


# ---------------------------------------------------------------- per-question trim
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


# ---------------------------------------------------------------- parse + dedup
def parse_questions(full_text: str) -> list[dict]:
    matches = list(QUESTION_HEADER.finditer(full_text))
    by_num: dict[int, str] = {}
    for i, m in enumerate(matches):
        num   = int(m.group(1))
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body  = cut_question_text(full_text[start:end])
        body  = normalize(body)
        if len(body) < 15:
            continue
        # Dedup: if same question number appears twice (OCR sometimes reads
        # 'Question 12' as 'Question 1'), keep the longer body.
        if num not in by_num or len(body) > len(by_num[num]):
            by_num[num] = body
    return [{"number": n, "text": t} for n, t in sorted(by_num.items())]


# ---------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf",        type=Path, default=PDF_PATH)
    p.add_argument("--start-page", type=int,  default=START_PAGE)
    p.add_argument("--end-page",   type=int,  default=None)
    p.add_argument("--dpi",        type=int,  default=DPI)
    p.add_argument("--psm",        type=int,  default=TESS_PSM)
    p.add_argument("--force-ocr",  action="store_true",
                   help="Ignore text layer entirely; OCR every page.")
    p.add_argument("--use-cache",  action="store_true",
                   help="Reuse debug_full_text.txt instead of re-extracting.")
    return p.parse_args()


def main():
    a = parse_args()
    if not a.pdf.exists():
        sys.exit(f"PDF not found: {a.pdf}")

    # ---------- 1. Extract text
    if a.use_cache and CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 0:
        print(f"Using cached text from {CACHE_FILE}")
        full_text = CACHE_FILE.read_text(encoding="utf-8")
    else:
        doc = fitz.open(a.pdf)
        end = min(a.end_page or len(doc), len(doc))
        total = end - (a.start_page - 1)
        print(f"Processing {total} pages ({a.start_page}–{end}) "
              f"@ {a.dpi} DPI, psm={a.psm}"
              f"{' [FORCE OCR]' if a.force_ocr else ''}\n")

        page_texts, t0 = [], time.time()
        for idx in range(a.start_page - 1, end):
            text, src = get_page_text(doc[idx], a.dpi, a.psm, a.force_ocr)
            text = strip_page_artifacts(text)
            page_texts.append(text)

            done = idx - (a.start_page - 1) + 1
            eta = (time.time() - t0) / done * (total - done)
            print(f"  p{idx+1:>4} [{src:>4}]  {len(text):>5} chars   "
                  f"({done}/{total}, ETA {eta/60:.1f} min)")

        full_text = "\n\n".join(page_texts)
        CACHE_FILE.write_text(full_text, encoding="utf-8")

    # ---------- 2. Global cleanup (fixes OCR prefixes on 'Question N')
    full_text = clean_full_text(full_text)
    CACHE_FILE.write_text(full_text, encoding="utf-8")  # save cleaned version

    # ---------- 3. Parse
    questions = parse_questions(full_text)

    lines = []
    for q in questions:
        lines.append(f"=== Question {q['number']} ===")
        lines.append(q["text"])
        lines.append("")
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    # ---------- 4. Summary
    print(f"\n✓ {len(questions)} questions  →  {OUTPUT_FILE}")
    print(f"✓ Cleaned text  →  {CACHE_FILE}")
    print("\n  Q#  |  chars  |  options")
    print("  ----+---------+---------")
    for q in questions:
        n_opts = len(re.findall(r'^\s*[1-4]\.\s', q["text"], re.MULTILINE))
        flag = " ⚠" if n_opts < 2 else ""
        print(f"  {q['number']:>3} |  {len(q['text']):>5}  |  {n_opts}{flag}")


if __name__ == "__main__":
    main()