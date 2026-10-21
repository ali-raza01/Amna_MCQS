import pymupdf

pdf_path = "mcqs_1.pdf"
output_path = "extracted_mixed_text.txt"

# Open the PDF document
doc = pymupdf.open(pdf_path)

# Open the output file for writing
with open(output_path, "w", encoding="utf-8") as out:
    # Iterate through each page
    for page_num, page in enumerate(doc, start=1):
        # Use to_markdown() or get_text() — PyMuPDF automatically
        # triggers OCR on any page that lacks selectable text.
        text = page.get_text("text", flags=pymupdf.TEXT_DEHYPHENATE)

        # Write a page header for clarity
        out.write(f"\n--- Page {page_num} ---\n")
        out.write(text)
        out.write("\n")

doc.close()
print(f"✅ Extraction complete. Text saved to '{output_path}'.")