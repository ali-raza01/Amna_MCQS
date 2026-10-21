import pymupdf
import pytesseract
from PIL import Image
import io
import re
from pathlib import Path

# --- CONFIG ---
PDF_PATH = "mcqs_1.pdf"
OUTPUT_DIR = Path("output")
START_PAGE = 17
END_PAGE = None          # None = to end; set e.g. 20 for a quick test
DPI = 300
PSM = 4                  # good for MCQ papers (single column, variable text)

# Figure filter: an embedded image whose OCR yields fewer characters
# than this threshold is treated as a genuine figure and saved.
MIN_FIGURE_CHARS = 100
MIN_FIGURE_DIM = 80      # skip tiny icons/bullets/watermarks

# --- PATTERNS ---
# Matches lines like: "Correct Answer: B", "Answer - C", "Rationale: ...",
# "Explanation: ...", "Solution: ...", "Justification: ..."
ANSWER_LINE = re.compile(
    r'(?i)^\s*(?:correct\s+)?(?:answer|ans|rationale|explanation|solution|justification)\s*[:\-–]'
)
# Matches the start of a new question: "17.", "Q17.", "Q.17.", "Question 17:"
QUESTION_START = re.compile(
    r'(?i)^\s*(?:question\s*)?(?:q\s*\.?\s*)?\d+\s*[\.\)\:\-–]\s'
)

def strip_answers_and_rationale(text: str) -> str:
    """Remove 'Correct Answer: ...' and any following rationale block
    until the next question begins."""
    out, in_explanation = [], False
    for line in text.split('\n'):
        s = line.strip()
        if ANSWER_LINE.match(s):
            in_explanation = True
            continue
        if QUESTION_START.match(s):
            in_explanation = False
        if not in_explanation:
            out.append(line)
    return '\n'.join(out)

def classify_image(img_bytes: bytes):
    """Return (keep_as_figure: bool, reason: str)."""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        if img.width < MIN_FIGURE_DIM or img.height < MIN_FIGURE_DIM:
            return False, f"too small ({img.width}x{img.height})"
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        txt = pytesseract.image_to_string(img).strip()
        chars = len(txt)
        if chars < MIN_FIGURE_CHARS:
            return True, f"figure ({chars} OCR chars, {img.width}x{img.height})"
        return False, f"text-image ({chars} OCR chars) - skipped"
    except Exception as e:
        return False, f"error: {e}"

# --- RUN ---
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

doc = pymupdf.open(PDF_PATH)
end = END_PAGE if END_PAGE else len(doc)

with open(OUTPUT_DIR / "cleaned_text.txt", "w", encoding="utf-8") as out:
    for page_idx in range(START_PAGE - 1, end):
        page = doc[page_idx]
        page_num = page_idx + 1
        out.write(f"\n{'='*72}\n=== PAGE {page_num} ===\n{'='*72}\n")

        # 1) Native text — preserves question numbers / headings
        native = page.get_text("text").strip()
        out.write("\n--- NATIVE TEXT ---\n")
        out.write(native + "\n" if native else "[none]\n")

        # 2) Full-page OCR, cleaned of answers/rationale
        pix = page.get_pixmap(dpi=DPI)
        page_img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_raw = pytesseract.image_to_string(page_img, config=f"--psm {PSM}")
        ocr_clean = strip_answers_and_rationale(ocr_raw)
        out.write("\n--- OCR TEXT (answers/rationale removed) ---\n")
        out.write(ocr_clean + "\n")

        # 3) Extract only figure-like embedded images
        out.write("\n--- FIGURES ---\n")
        kept = 0
        for i, info in enumerate(page.get_images(full=True)):
            xref = info[0]
            try:
                base = doc.extract_image(xref)
            except Exception as e:
                out.write(f"  [{i}] extract error: {e}\n")
                continue
            keep, reason = classify_image(base["image"])
            if keep:
                fname = f"page_{page_num:03d}_fig_{kept}.{base['ext']}"
                (FIGURES_DIR / fname).write_bytes(base["image"])
                out.write(f"  [{i}] KEPT {fname} — {reason}\n")
                kept += 1
            else:
                out.write(f"  [{i}] skipped — {reason}\n")
        if kept == 0:
            out.write("  [no figures on this page]\n")

doc.close()
print(f"✅ Done.")
print(f"   Text  : {OUTPUT_DIR/'cleaned_text.txt'}")
print(f"   Figures: {FIGURES_DIR}/")