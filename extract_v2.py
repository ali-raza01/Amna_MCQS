import pymupdf
import pytesseract
from PIL import Image
from pathlib import Path

# --- CONFIG ---
PDF_PATH = "mcqs_1.pdf"
OUTPUT_DIR = Path("output")
START_PAGE = 17        # 1-indexed, inclusive
END_PAGE = None        # None = go to end; set e.g. 20 for a test run

OUTPUT_DIR.mkdir(exist_ok=True)
IMAGES_DIR = OUTPUT_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

doc = pymupdf.open(PDF_PATH)
total = len(doc)
end = END_PAGE if END_PAGE else total

with open(OUTPUT_DIR / "extracted_text.txt", "w", encoding="utf-8") as out:
    for page_idx in range(START_PAGE - 1, end):
        page = doc[page_idx]
        page_num = page_idx + 1
        out.write(f"\n{'='*72}\n=== PAGE {page_num} ===\n{'='*72}\n")

        # 1) Native text (may be sparse or heading-only)
        native = page.get_text("text").strip()
        out.write(f"\n--- NATIVE TEXT ({len(native)} chars) ---\n")
        out.write(native + "\n" if native else "[none]\n")

        # 2) Always render the full page and OCR it
        pix = page.get_pixmap(dpi=300)   # 300 DPI is a good OCR trade-off
        page_img = IMAGES_DIR / f"page_{page_num:03d}_full.png"
        pix.save(str(page_img))
        out.write(f"\n--- FULL PAGE IMAGE: {page_img.name} ---\n")

        try:
            ocr = pytesseract.image_to_string(Image.open(page_img))
            out.write(f"\n--- OCR TEXT ---\n{ocr}\n")
        except Exception as e:
            out.write(f"\n[OCR ERROR: {e}]\n")

        # 3) Extract every embedded image on the page
        img_list = page.get_images(full=True)
        out.write(f"\n--- EMBEDDED IMAGES ({len(img_list)}) ---\n")
        for i, info in enumerate(img_list):
            xref = info[0]
            try:
                base = doc.extract_image(xref)
                ext = base["ext"]
                img_file = IMAGES_DIR / f"page_{page_num:03d}_img_{i}.{ext}"
                img_file.write_bytes(base["image"])
                out.write(f"  [{i}] {img_file.name} ({base['width']}x{base['height']})\n")
            except Exception as e:
                out.write(f"  [{i}] ERROR: {e}\n")

doc.close()
print(f"✅ Done. Output in '{OUTPUT_DIR}/' — text in extracted_text.txt, images in images/")