uv run extract_questions_v5.py --start-page 453 --end-page 457 --force-ocr

uv run annotate_answers.py --section "1.A Anatomy" --out test_anatomy.pdf

uv run annotate_answers.py --section "1.A Anatomy" --rebuild-index --dpi 400 --out test.pdf