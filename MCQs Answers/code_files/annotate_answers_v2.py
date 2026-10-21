#!/usr/bin/env python3
"""
annotate_answers.py — mark answers on mcqs_1.pdf, section by section.

Key change vs. previous versions:
    Question BODIES come from <section>/debug_raw_pages.txt (fast, cached).
    Question POSITIONS come from fresh OCR of the PDF pages listed in SECTIONS.
    Page offsets in the cache are irrelevant — no more silent zero-annotation.
"""
from __future__ import annotations

import argparse
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


# ════════════════════════════════════════════════════════════════════════
# Question bodies from cache (page offsets ignored — we only want text)
# ════════════════════════════════════════════════════════════════════════
def load_question_bodies(cache_path):
    raw = cache_path.read_text(encoding="utf-8")
    text = raw.replace(PAGE_SENTINEL.strip(), "\n\n")

    matches, out = list(QUESTION_HEADER.finditer(text)), []
    for i, m in enumerate(matches):
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end]
        for pat in END_MARKERS:
            em = pat.search(chunk)
            if em:
                chunk = chunk[:em.start()]
                break
        body = chunk[m.end() - start:].strip()
        if len(body) < 15:
            continue
        out.append({"orig_num": int(m.group(1)), "body": body})
    return out


# ════════════════════════════════════════════════════════════════════════
# Question positions from fresh OCR of the real PDF pages
# ════════════════════════════════════════════════════════════════════════
def build_position_index(pdf_path, start_page, end_page, dpi=300):
    doc = fitz.open(pdf_path)
    end_page = min(end_page, len(doc))
    total = end_page - start_page + 1
    print(f"  OCR pages {start_page}–{end_page} ({total} pages, {dpi} DPI) …")
    index, t0 = {}, time.time()

    for k, pnum in enumerate(range(start_page, end_page + 1), 1):
        page = doc[pnum - 1]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        scale, words = 72.0 / dpi, data["text"]

        for i, w in enumerate(words):
            wl = w.lower().strip(".,:;#")
            if wl == "question":
                for j in range(i + 1, min(i + 4, len(words))):
                    wj = words[j].strip(".,:;#")
                    if wj.isdigit():
                        n = int(wj)
                        if n not in index:
                            x  = data["left"][i] * scale
                            y  = data["top"][i]  * scale
                            x2 = (data["left"][j] + data["width"][j]) * scale
                            h  = max(data["height"][i],
                                     data["height"][j]) * scale
                            index[n] = (pnum, x, y, x2, y + h)
                        break
            elif wl.startswith("question") and len(w) > 8:
                rest = w[8:].strip(".,:;#")
                if rest.isdigit():
                    n = int(rest)
                    if n not in index:
                        x  = data["left"][i] * scale
                        y  = data["top"][i]  * scale
                        x2 = (data["left"][i] + data["width"][i]) * scale
                        h  = data["height"][i] * scale
                        index[n] = (pnum, x, y, x2, y + h)

        if k % 10 == 0 or k == total:
            eta = (time.time() - t0) / k * (total - k)
            print(f"    {k}/{total}  {len(index)} headers  ETA {eta/60:.1f}min")

    doc.close()
    return index


def load_or_build_position_index(pdf_path, start_page, end_page,
                                 dpi, rebuild, cache_file):
    if not rebuild and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            idx = {int(k): tuple(v) for k, v in data.items()}
            print(f"  Loading positions from {cache_file.name} "
                  f"({len(idx)} headers)")
            return idx
        except Exception as e:
            print(f"  ⚠ failed to load {cache_file.name}: {e} → rebuilding")

    idx = build_position_index(pdf_path, start_page, end_page, dpi)
    cache_file.write_text(json.dumps({str(k): list(v) for k, v in idx.items()},
                                     indent=2))
    print(f"  → {len(idx)} headers cached to {cache_file.name}")
    return idx


# ════════════════════════════════════════════════════════════════════════
# Answer-file parsing
# ════════════════════════════════════════════════════════════════════════
def parse_answers(text):
    answers = {}
    md = list(re.finditer(r'###\s*Q(\d+)\b', text, re.IGNORECASE))
    for i, h in enumerate(md):
        n = int(h.group(1))
        end = md[i + 1].start() if i + 1 < len(md) else len(text)
        chunk = text[h.end():end]
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

    for m in re.finditer(
        r"===\s*Question\s+(\d+)[^=\n]*===[\s\S]*?"
        r"(?:Correct\s+Answer|Answer)\s*[:\-]\s*([^\n]+)",
        text, re.IGNORECASE,
    ):
        answers.setdefault(int(m.group(1)), m.group(2).strip())
    for m in re.finditer(r"(?:Question|Q)\s*(\d+)\s*[:\-=]\s*([^\n]+)",
                         text, re.IGNORECASE):
        answers.setdefault(int(m.group(1)), m.group(2).strip())
    for m in re.finditer(r"^\s*(\d+)\s*[.\)\-:]\s*([^\n]+)$",
                         text, re.MULTILINE):
        answers.setdefault(int(m.group(1)), m.group(2).strip())
    return answers


def normalize_answer(raw):
    raw = raw.strip().strip("*_`").strip()
    if not raw:
        return (None, "")
    raw = re.sub(r"^(?:the\s+)?answer\s*(?:is)?\s*[:\-—–]?\s*", "",
                 raw, flags=re.IGNORECASE)
    raw = re.sub(r"^(?:option|opt)\.?\s*", "", raw, flags=re.IGNORECASE).strip()
    if not raw:
        return (None, "")
    letters = re.sub(r"[^A-Za-z]", "", raw).upper()
    if len(letters) == 1 and letters in "ABCD":
        return (ord(letters) - ord("A") + 1, "")
    m = re.match(r"^([a-dA-D])(?:[.)\-—–:]|\s)+\s*(.*)$", raw)
    if m:
        return (ord(m.group(1).upper()) - ord("A") + 1,
                re.sub(r"\s+", " ", m.group(2)).strip())
    m = re.match(r"^([1-4])(?:[.)\-—–:]|\s)+\s*(.*)$", raw)
    if m:
        return (int(m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip())
    m = re.match(r"^([1-4])\b", raw)
    if m:
        return (int(m.group(1)),
                re.sub(r"^[.\)\-—–:]\s*", "", raw[m.end():].strip()))
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
# Section folder / answer file discovery
# ════════════════════════════════════════════════════════════════════════
def find_section_folder(label, base):
    sf = slug(label)
    sh = slug(label.split()[-1])
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        ss = slug(sub.name)
        if ss == sf or ss.startswith(sf) or sh in ss:
            return sub
    return None


def find_answer_file_in(folder):
    ignore = {"debug_raw_pages", "questions_clean", "debug_parsed",
              "annotation_report", "sections_report", "read"}
    preferred = [slug(folder.name) + "_answers",
                 "answers", "README", "readme"]
    for name in preferred:
        for ext in (".md", ".txt"):
            f = folder / (name + ext)
            if f.exists():
                return f
    for ext in (".md", ".txt"):
        for f in sorted(folder.glob("*" + ext)):
            if slug(f.stem) not in ignore:
                return f
    return None


# ════════════════════════════════════════════════════════════════════════
# Annotation
# ════════════════════════════════════════════════════════════════════════
def annotate_section(doc, questions, answers, positions, report, label):
    matched = 0
    no_answer, no_position = [], []
    used_y = {}

    for q in questions:
        n = q["orig_num"]
        if n not in answers:
            no_answer.append(n)
            continue
        if n not in positions:
            no_position.append(n)
            continue

        pnum, x, y, x2, y2 = positions[n]
        page = doc[pnum - 1]
        rect = fitz.Rect(x, y, x2, y2)

        opt_num, extra = normalize_answer(answers[n])
        options = extract_options(q["body"])
        ans_text = resolve_answer_text(opt_num, extra, options)

        # Highlight
        try:
            page.add_highlight_annot(rect)
        except Exception:
            pass

        # Right-margin tag
        tag_x = page.rect.width - 54
        tag_y = y
        ys = used_y.setdefault(pnum, [])
        while any(abs(tag_y - yy) < 14 for yy in ys):
            tag_y += 14
        ys.append(tag_y)

        tag = f"ANS {opt_num}" if opt_num else "ANS"
        try:
            page.add_freetext_annot(
                fitz.Rect(tag_x, tag_y, tag_x + 46, tag_y + 12),
                tag, fontsize=9,
                text_color=(1, 0, 0),
                fill_color=(1, 1, 0.85),
                border_color=(1, 0, 0), border_width=0.5,
            )
        except Exception:
            pass

        # Sticky note
        note = f"Q{n} → {tag}"
        if extra and ans_text and extra.lower() != ans_text.lower():
            note += f"\nLLM: {extra}\nOption: {ans_text}"
        elif ans_text:
            note += f"\n{ans_text}"
        try:
            page.add_text_annot(rect.tl, note)
        except Exception:
            pass

        matched += 1

    total = len(questions)
    pct = 100 * matched / total if total else 0
    report.append(f"[{label}]  {matched}/{total}  ({pct:.0f}%)")
    if no_answer:
        p = ", ".join(map(str, no_answer[:20]))
        more = f"  …(+{len(no_answer)-20})" if len(no_answer) > 20 else ""
        report.append(f"  ⚠ no answer entry: Q{p}{more}")
    if no_position:
        p = ", ".join(map(str, no_position[:20]))
        more = f"  …(+{len(no_position)-20})" if len(no_position) > 20 else ""
        report.append(f"  ⚠ OCR couldn't locate: Q{p}{more}")
    report.append("")
    return matched, total


# ════════════════════════════════════════════════════════════════════════
def process_section(label, folder, pdf_path, doc, dpi, rebuild, report):
    print(f"\n→ {label}   ({folder.name})")
    rng = section_range(label)
    if not rng:
        print("  ⚠ unknown section")
        return 0, 0
    start_page, end_page, expected = rng

    cache = folder / "debug_raw_pages.txt"
    if not cache.exists():
        print(f"  ⚠ missing {cache}")
        return 0, 0

    questions = load_question_bodies(cache)
    print(f"  questions from cache: {len(questions)} (expected {expected})")

    ans_file = find_answer_file_in(folder)
    if not ans_file:
        print("  ⚠ no answer file")
        return 0, 0
    print(f"  answer file: {ans_file.name}")
    answers = parse_answers(ans_file.read_text(encoding="utf-8"))
    print(f"  answers parsed: {len(answers)}")

    idx_file = folder / ".header_index.json"
    positions = load_or_build_position_index(
        pdf_path, start_page, end_page, dpi, rebuild, idx_file)
    print(f"  positions found: {len(positions)}")

    return annotate_section(doc, questions, answers, positions, report, label)


# ════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf",           default="mcqs_1.pdf")
    ap.add_argument("--out",           default="mcqs_1_annotated.pdf")
    ap.add_argument("--report",        default="annotation_report.txt")
    ap.add_argument("--section",       default=None)
    ap.add_argument("--section-dir",   default=None)
    ap.add_argument("--answers-dir",   default=None)
    ap.add_argument("--dpi",           type=int, default=300)
    ap.add_argument("--rebuild-index", action="store_true",
                    help="Force re-OCR of the section's header positions")
    ap.add_argument("--dry-run",       action="store_true")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        sys.exit(f"PDF not found: {pdf}")

    jobs = []
    if args.section:
        base = Path(args.section_dir or ".")
        sdir = Path(args.section_dir) if args.section_dir \
               else find_section_folder(args.section, base)
        if not sdir or not sdir.is_dir():
            sys.exit(f"Section folder not found: {args.section}")
        jobs.append((args.section, sdir))
    elif args.answers_dir:
        base = Path(args.answers_dir)
        if not base.is_dir():
            sys.exit(f"Not a directory: {base}")
        for _, _, lbl, _ in SECTIONS:
            sub = find_section_folder(lbl, base)
            if sub:
                jobs.append((lbl, sub))
            else:
                print(f"  ⚠ no folder for {lbl}")
        if not jobs:
            sys.exit("No section folders found")
    else:
        sys.exit("Use --section or --answers-dir")

    print(f"Processing {len(jobs)} section(s)")
    doc = fitz.open(pdf)
    report = ["ANSWER ANNOTATION REPORT", "=" * 60, ""]
    gm = gt = 0
    for label, folder in jobs:
        m, t = process_section(label, folder, pdf, doc,
                               args.dpi, args.rebuild_index, report)
        gm += m
        gt += t
        print(f"  → {m}/{t} annotated")

    if not args.dry_run:
        doc.save(args.out, garbage=4, deflate=True)
        print(f"\n✓ saved → {args.out}")
    else:
        print("\n(dry run)")
    doc.close()

    report.append("=" * 60)
    report.append(f"TOTAL: {gm}/{gt}")
    Path(args.report).write_text("\n".join(report), encoding="utf-8")
    print(f"✓ report → {args.report}")
    print(f"✓ {gm}/{gt} annotated")


if __name__ == "__main__":
    main()