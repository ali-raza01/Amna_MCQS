#!/usr/bin/env python3
"""
annotate_answers.py — mark LLM answers on mcqs_1.pdf, section by section.

Per-section layout assumed:

    output_extract_v10/
    ├── mcqs_1.pdf
    ├── annotate_answers.py
    ├── 1.A Anatomy/
    │   ├── anatomy_answers.txt      (or README.md)
    │   ├── debug_raw_pages.txt      (cache from extract_questions.py)
    │   └── .header_index.json       (auto-built on first run, ~1-2 min)
    ├── 1.B Biomechanics/
    │   ├── README.md
    │   ├── debug_raw_pages.txt
    │   └── .header_index.json
    └── ...

Usage (from output_extract_v10/):

    # all sections
    uv run annotate_answers.py --answers-dir . --out annotated_full.pdf

    # one section
    uv run annotate_answers.py --section "1.A Anatomy" --out annotated_anatomy.pdf

    # dry-run
    uv run annotate_answers.py --answers-dir . --out x.pdf --dry-run
"""
from __future__ import annotations

import argparse
import bisect
import io
import json
import re
import sys
import time
from pathlib import Path

import fitz
from PIL import Image
import pytesseract


# ════════════════════════════════════════════════════════════════════════
# SECTIONS
# ════════════════════════════════════════════════════════════════════════
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

PAGE_SENTINEL   = "\n<<<PAGE_BREAK>>>\n"
QUESTION_HEADER = re.compile(r'^\s*#{0,6}\s*Question\s*(\d+)\b',
                             re.MULTILINE | re.IGNORECASE)
END_MARKERS = [
    re.compile(r'\n\s*#{0,6}\s*Correct\s*Answer\b', re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*Rationale\b',        re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*Note\s+for\b',       re.IGNORECASE),
    re.compile(r'\n\s*#{0,6}\s*NOTE\b',             re.IGNORECASE),
]


def slug(s):
    s = s.lower()
    s = re.sub(r"[.\s]+", "_", s)
    s = re.sub(r"[^\w]+", "_", s)
    return s.strip("_")


def section_range(label):
    for s, e, lbl, n in SECTIONS:
        if lbl == label:
            return s, e, n
    return None


def resolve_section(page):
    for s, e, lbl, _ in SECTIONS:
        if s <= page <= e:
            return lbl
    return None


# ════════════════════════════════════════════════════════════════════════
# Load questions from per-section cache
# ════════════════════════════════════════════════════════════════════════
def load_questions_from_cache(cache_path, start_page):
    """
    Parse questions from one section's debug_raw_pages.txt.
    `start_page` = the PDF page this cache file begins at
                  (from SECTIONS table).
    """
    raw = cache_path.read_text(encoding="utf-8")
    pages = raw.split(PAGE_SENTINEL.strip())
    print(f"  cache has {len(pages)} pages, starting at PDF page {start_page}")

    parts, meta, pos = [], [], 0
    for i, p in enumerate(pages):
        page_num = start_page + i
        parts.append(p)
        meta.append((page_num, pos, resolve_section(page_num)))
        pos += len(p) + 2
    joined = "\n\n".join(parts)
    offsets = [m[1] for m in meta]

    def page_at(off):
        i = bisect.bisect_right(offsets, off) - 1
        return meta[max(0, i)]

    matches, questions = list(QUESTION_HEADER.finditer(joined)), []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(joined)
        chunk = joined[start:end]
        for pat in END_MARKERS:
            em = pat.search(chunk)
            if em:
                chunk = chunk[:em.start()]
                break
        body = chunk[m.end() - start:].strip()
        if len(body) < 15:
            continue
        p_start = page_at(start)
        questions.append({
            "seq":        len(questions) + 1,
            "orig_num":   int(m.group(1)),
            "page_start": p_start[0],
            "section":    p_start[2],
            "body":       body,
        })
    return questions


# ════════════════════════════════════════════════════════════════════════
# Answer-file parsing
# ════════════════════════════════════════════════════════════════════════
def parse_answers(text):
    """{orig_num: raw_answer}  —  markdown format (### Q + Chosen answer)."""
    answers = {}

    md_headers = list(re.finditer(r'###\s*Q(\d+)\b', text, re.IGNORECASE))
    for i, h in enumerate(md_headers):
        n = int(h.group(1))
        start = h.end()
        end = md_headers[i + 1].start() if i + 1 < len(md_headers) else len(text)
        chunk = text[start:end]

        m = re.search(
            r'\*\*\s*Chosen\s+answer\s*:?\s*\*\*\s*'
            r'(.+?)(?=\n\s*\*\*[A-Z]|\n\s*\n|\Z)',
            chunk, re.IGNORECASE | re.DOTALL,
        )
        if not m:
            m = re.search(
                r'Chosen\s+answer\s*:?\s*(.+?)(?:\n\s*\n|\n\s*\*\*|\Z)',
                chunk, re.IGNORECASE | re.DOTALL,
            )
        if m:
            answers[n] = m.group(1).strip()

    if answers:
        return answers

    # Fallbacks (older formats)
    for m in re.finditer(
        r"===\s*Question\s+(\d+)[^=\n]*===[\s\S]*?"
        r"(?:Correct\s+Answer|Answer)\s*[:\-]\s*([^\n]+)",
        text, re.IGNORECASE,
    ):
        answers[int(m.group(1))] = m.group(2).strip()

    for m in re.finditer(r"(?:Question|Q)\s*(\d+)\s*[:\-=]\s*([^\n]+)",
                         text, re.IGNORECASE):
        n = int(m.group(1))
        if n not in answers:
            answers[n] = m.group(2).strip()

    for m in re.finditer(r"^\s*(\d+)\s*[.\)\-:]\s*([^\n]+)$",
                         text, re.MULTILINE):
        n = int(m.group(1))
        if n not in answers:
            answers[n] = m.group(2).strip()

    return answers


def normalize_answer(raw):
    """
    "**b — hip extension**" → (2, "hip extension")
    "**2**"                 → (2, "")
    "**Extension of hip**"  → (None, "Extension of hip")
    """
    raw = raw.strip().strip("*_`").strip()
    if not raw:
        return (None, "")

    raw = re.sub(r"^(?:the\s+)?answer\s*(?:is)?\s*[:\-—–]?\s*", "",
                 raw, flags=re.IGNORECASE)
    raw = re.sub(r"^(?:option|opt)\.?\s*", "", raw, flags=re.IGNORECASE)
    raw = raw.strip()
    if not raw:
        return (None, "")

    letter_clean = re.sub(r"[.,;:!?()\[\]{}'\"`*_\s]+", "", raw).upper()
    if len(letter_clean) == 1 and letter_clean in "ABCD":
        return (ord(letter_clean) - ord("A") + 1, "")

    m = re.match(r"^([a-dA-D])(?:[.)\-—–:]|\s)+\s*(.*)$", raw)
    if m:
        return (ord(m.group(1).upper()) - ord("A") + 1,
                re.sub(r"\s+", " ", m.group(2)).strip())

    m = re.match(r"^([1-4])(?:[.)\-—–:]|\s)+\s*(.*)$", raw)
    if m:
        return (int(m.group(1)),
                re.sub(r"\s+", " ", m.group(2)).strip())

    m = re.match(r"^([1-4])\b", raw)
    if m:
        rest = re.sub(r"^[.\)\-—–:]\s*", "", raw[m.end():].strip())
        return (int(m.group(1)), rest)

    return (None, raw)


def extract_options(body):
    opts = {}
    for m in re.finditer(r"^\s*([1-4])[.\)\-:]\s+(.+?)(?=\n|$)",
                         body, re.MULTILINE):
        opts[int(m.group(1))] = m.group(2).strip()
    if len(opts) >= 2:
        return opts
    for m in re.finditer(
        r"(?:^|\s)([1-4])[.\)\-:]\s+(.+?)(?=\s+[1-4][.\)\-:]|$)", body
    ):
        opts[int(m.group(1))] = m.group(2).strip()
    return opts


def resolve_answer_text(opt_num, extra, options):
    if opt_num is not None and opt_num in options:
        return options[opt_num]
    if extra:
        el = extra.lower().strip()
        for o in options.values():
            if o.lower().strip() == el or el in o.lower() or o.lower() in el:
                return o
    return extra


# ════════════════════════════════════════════════════════════════════════
# Per-section header index (cached inside each section folder)
# ════════════════════════════════════════════════════════════════════════
def build_header_index(pdf_path, start_page, end_page, dpi=300):
    doc = fitz.open(pdf_path)
    end_page = min(end_page, len(doc))
    total = end_page - start_page + 1
    print(f"Building header index for pages {start_page}–{end_page} "
          f"({total} pages @ {dpi} DPI) …")
    index, t0 = {}, time.time()

    for k, page_num in enumerate(range(start_page, end_page + 1), 1):
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT)
        scale, words = 72.0 / dpi, data["text"]
        headers = {}

        for i, w in enumerate(words):
            wl = w.lower().strip(".,:;#")
            if wl == "question":
                for j in range(i + 1, min(i + 4, len(words))):
                    wj = words[j].strip(".,:;#")
                    if wj.isdigit():
                        n = int(wj)
                        if n not in headers:
                            x  = data["left"][i] * scale
                            y  = data["top"][i]  * scale
                            x2 = (data["left"][j] +
                                  data["width"][j]) * scale
                            h  = max(data["height"][i],
                                     data["height"][j]) * scale
                            headers[n] = (x, y, x2, y + h)
                        break
            elif wl.startswith("question") and len(w) > 8:
                rest = w[8:].strip(".,:;#")
                if rest.isdigit():
                    n = int(rest)
                    if n not in headers:
                        x  = data["left"][i] * scale
                        y  = data["top"][i]  * scale
                        x2 = (data["left"][i] +
                              data["width"][i]) * scale
                        h  = data["height"][i] * scale
                        headers[n] = (x, y, x2, y + h)

        if headers:
            index[page_num] = headers
        if k % 10 == 0 or k == total:
            eta = (time.time() - t0) / k * (total - k)
            print(f"  {k}/{total} pages, "
                  f"{sum(len(v) for v in index.values())} headers, "
                  f"ETA {eta/60:.1f} min")

    doc.close()
    return index


def load_or_build_index(pdf_path, start_page, end_page, dpi, rebuild, cache_dir):
    cache = cache_dir / ".header_index.json"
    if not rebuild and cache.exists():
        print(f"Loading header index from {cache}")
        data = json.loads(cache.read_text())
        return {int(k): {int(n): tuple(r) for n, r in v.items()}
                for k, v in data.items()}
    index = build_header_index(pdf_path, start_page, end_page, dpi)
    cache.write_text(json.dumps(
        {str(k): {str(n): list(r) for n, r in v.items()}
         for k, v in index.items()}, indent=2))
    print(f"  → cached to {cache}")
    return index


# ════════════════════════════════════════════════════════════════════════
# Section discovery inside a base directory
# ════════════════════════════════════════════════════════════════════════
def find_section_folder(section_label, base_dir):
    """Locate the subdirectory matching a section label."""
    slug_full = slug(section_label)              # "1_a_anatomy"
    slug_short = slug(section_label.split()[-1]) # "anatomy"
    for sub in sorted(base_dir.iterdir()):
        if not sub.is_dir():
            continue
        sub_slug = slug(sub.name)
        if (sub_slug == slug_full
                or sub_slug.startswith(slug_full)
                or slug_short in sub_slug):
            return sub
    return None


def find_answer_file_in(section_dir):
    """
    Locate the answer file inside a section folder.
    Prefers README.md, then *_answers.txt, then any .md/.txt
    (ignoring debug_raw_pages.txt and other known artifacts).
    """
    ignore = {"debug_raw_pages", "questions_clean", "debug_parsed",
              "annotation_report", "sections_report", "read"}

    # Preferred order
    for name in ("README", "readme", "answers",
                 slug(section_dir.name) + "_answers"):
        for ext in (".md", ".txt"):
            f = section_dir / (name + ext)
            if f.exists():
                return f

    # Fuzzy
    for ext in (".md", ".txt"):
        for f in sorted(section_dir.glob("*" + ext)):
            if slug(f.stem) in ignore:
                continue
            if "answers" in f.stem.lower():
                return f

    # Any non-artifact .md/.txt
    for ext in (".md", ".txt"):
        for f in sorted(section_dir.glob("*" + ext)):
            if slug(f.stem) in ignore:
                continue
            return f

    return None


# ════════════════════════════════════════════════════════════════════════
# Annotation
# ════════════════════════════════════════════════════════════════════════
def annotate_section(doc, section_label, questions, answers, index, report):
    section_qs = [q for q in questions if q["section"] == section_label]
    matched, no_answer, no_location = 0, [], []
    used_y = {}

    for q in section_qs:
        n = q["orig_num"]
        if n not in answers:
            no_answer.append(n)
            continue

        opt_num, extra = normalize_answer(answers[n])
        options  = extract_options(q["body"])
        ans_text = resolve_answer_text(opt_num, extra, options)

        page_num  = q["page_start"]
        rect_data = index.get(page_num, {}).get(n)

        if not rect_data:
            page = doc[page_num - 1]
            for variant in (f"Question {n}", f"Question{n}"):
                for r in page.search_for(variant):
                    try:
                        txt = page.get_textbox(r).strip()
                    except Exception:
                        txt = ""
                    if re.match(rf"^Question\s*{n}\b(?!\d)", txt,
                                re.IGNORECASE):
                        rect_data = (r.x0, r.y0, r.x1, r.y1)
                        break
                if rect_data:
                    break

        if not rect_data:
            no_location.append((n, page_num))
            continue

        page = doc[page_num - 1]
        rect = fitz.Rect(*rect_data)

        try:
            page.add_highlight_annot(rect)
        except Exception:
            pass

        tag_x = page.rect.width - 54
        tag_y = rect.y0
        ys = used_y.setdefault(page_num, [])
        while any(abs(tag_y - y) < 14 for y in ys):
            tag_y += 14
        ys.append(tag_y)

        tag_label = f"ANS {opt_num}" if opt_num else "ANS"
        try:
            page.add_freetext_annot(
                fitz.Rect(tag_x, tag_y, tag_x + 46, tag_y + 12),
                tag_label, fontsize=9,
                text_color=(1, 0, 0),
                fill_color=(1, 1, 0.85),
                border_color=(1, 0, 0),
                border_width=0.5,
            )
        except Exception:
            pass

        note = f"Q{n} → {tag_label}"
        if extra and ans_text and extra.lower() != ans_text.lower():
            note += f"\nLLM: {extra}"
            note += f"\nOption: {ans_text}"
        elif ans_text:
            note += f"\n{ans_text}"
        if opt_num is not None and options and opt_num not in options:
            note += f"\n⚠ option {opt_num} not found in question"
        try:
            page.add_text_annot(rect.tl, note)
        except Exception:
            pass

        matched += 1

    total = len(section_qs)
    pct = 100 * matched / total if total else 0
    _, _, exp = section_range(section_label) or (0, 0, None)
    report.append(f"[{section_label}]   {matched}/{total}  ({pct:.0f}%)"
                  + (f"   expected {exp}" if exp else ""))
    if no_answer:
        p = ", ".join(map(str, no_answer[:20]))
        more = f"  …(+{len(no_answer)-20})" if len(no_answer) > 20 else ""
        report.append(f"  ⚠ no answer entry: Q{p}{more}")
    if no_location:
        items = ", ".join(f"Q{n}@p{pg}" for n, pg in no_location[:10])
        report.append(f"  ⚠ header not located: {items}")
    report.append("")
    return matched, total


# ════════════════════════════════════════════════════════════════════════
# Full section pipeline
# ════════════════════════════════════════════════════════════════════════
def process_section(section_label, section_dir, pdf_path, doc,
                    dpi, rebuild_index, report):
    print(f"\n→ {section_label}   ({section_dir})")

    rng = section_range(section_label)
    if not rng:
        print("  ⚠ unknown section label")
        return 0, 0
    start_page, end_page, expected = rng

    cache = section_dir / "debug_raw_pages.txt"
    if not cache.exists():
        print(f"  ⚠ missing cache: {cache}")
        return 0, 0

    ans_file = find_answer_file_in(section_dir)
    if not ans_file:
        print(f"  ⚠ no answer file found in {section_dir}")
        return 0, 0
    print(f"  answers : {ans_file.name}")

    questions = load_questions_from_cache(cache, start_page)
    print(f"  questions parsed: {len(questions)} (expected {expected})")

    answers = parse_answers(ans_file.read_text(encoding="utf-8"))
    print(f"  answers parsed  : {len(answers)}")

    index = load_or_build_index(pdf_path, start_page, end_page,
                                dpi, rebuild_index, section_dir)
    pages_indexed = len([p for p in index if start_page <= p <= end_page])
    print(f"  header index    : {pages_indexed} pages in range")

    return annotate_section(doc, section_label, questions,
                            answers, index, report)


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf",           default="mcqs_1.pdf")
    ap.add_argument("--out",           default="mcqs_1_annotated.pdf")
    ap.add_argument("--report",        default="annotation_report.txt")
    ap.add_argument("--section",       default=None,
                    help='Single section, e.g. "1.A Anatomy"')
    ap.add_argument("--section-dir",   default=None,
                    help="Section folder (overrides auto-discovery)")
    ap.add_argument("--answers-dir",   default=None,
                    help="Base directory containing all section folders")
    ap.add_argument("--dpi",           type=int, default=300)
    ap.add_argument("--rebuild-index", action="store_true",
                    help="Force re-OCR of the section's header index")
    ap.add_argument("--dry-run",       action="store_true")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        sys.exit(f"PDF not found: {pdf}")

    # Build list of (section_label, section_dir) to process
    jobs = []
    if args.section:
        base = Path(args.section_dir or ".")
        sdir = Path(args.section_dir) if args.section_dir \
               else find_section_folder(args.section, base)
        if not sdir or not sdir.is_dir():
            sys.exit(f"Section folder not found for: {args.section}")
        jobs.append((args.section, sdir))
    elif args.answers_dir:
        base = Path(args.answers_dir)
        if not base.is_dir():
            sys.exit(f"Not a directory: {base}")
        for _, _, label, _ in SECTIONS:
            sdir = find_section_folder(label, base)
            if sdir:
                jobs.append((label, sdir))
            else:
                print(f"  ⚠ no folder for {label}")
        if not jobs:
            sys.exit("No section folders found")
    else:
        sys.exit("Use --section or --answers-dir")

    print(f"Processing {len(jobs)} section(s)")

    doc = fitz.open(pdf)
    report = ["ANSWER ANNOTATION REPORT", "=" * 60, ""]
    gm = gt = 0
    for label, sdir in jobs:
        m, t = process_section(label, sdir, pdf, doc,
                               args.dpi, args.rebuild_index, report)
        gm += m
        gt += t
        print(f"  → {m}/{t} annotated")

    if not args.dry_run:
        doc.save(args.out, garbage=4, deflate=True)
        print(f"\n✓ annotated PDF saved → {args.out}")
    else:
        print("\n(dry run — PDF not saved)")
    doc.close()

    report.append("=" * 60)
    report.append(f"TOTAL: {gm}/{gt} annotated")
    Path(args.report).write_text("\n".join(report), encoding="utf-8")
    print(f"✓ report → {args.report}")
    print(f"✓ {gm}/{gt} questions annotated")


if __name__ == "__main__":
    main()