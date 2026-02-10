#!/usr/bin/env python3
"""Run this in your repo root: python3 diagnose_pdfs.py"""

import os

try:
    from pypdf import PdfReader
except ImportError:
    print("Installing pypdf...")
    os.system("pip install pypdf")
    from pypdf import PdfReader

data_dir = "data"
if not os.path.exists(data_dir):
    print("ERROR: No 'data' directory found. Run this from your repo root.")
    exit(1)

print("=" * 70)
print("PDF CORPUS DIAGNOSTIC")
print("=" * 70)

total_files = 0
total_pages = 0
total_chars = 0
problem_files = []

for root, _, fnames in os.walk(data_dir):
    for fn in sorted(fnames):
        if not fn.lower().endswith(".pdf"):
            continue
        total_files += 1
        fpath = os.path.join(root, fn)
        size_kb = os.path.getsize(fpath) / 1024

        try:
            reader = PdfReader(fpath)
            pages = len(reader.pages)
            total_pages += pages

            text = ""
            for p in reader.pages:
                t = p.extract_text() or ""
                text += t

            chars = len(text.strip())
            words = len(text.split())
            total_chars += chars

            status = "OK" if chars > 100 else "** NO TEXT **"
            if chars <= 100:
                problem_files.append((fn, "No extractable text - likely scanned image"))

            print(f"\n  {fn}")
            print(f"    Size: {size_kb:.0f} KB | Pages: {pages} | Words: {words:,} | Chars: {chars:,} | {status}")

            if chars > 0:
                preview = text.strip()[:200].replace("\n", " ")
                print(f"    Preview: {preview}...")

        except Exception as e:
            problem_files.append((fn, str(e)))
            print(f"\n  {fn}")
            print(f"    ** ERROR: {e} **")

print("\n" + "=" * 70)
print(f"SUMMARY: {total_files} files, {total_pages} pages, {total_chars:,} total characters")
print("=" * 70)

if problem_files:
    print("\nPROBLEM FILES:")
    for fn, reason in problem_files:
        print(f"  ** {fn}: {reason}")
    print("\nFiles with no extractable text are INVISIBLE to the RAG.")
    print("Fix: Re-export as text-based PDF, or OCR with pytesseract.")
else:
    print("\nAll files have extractable text. If the RAG still can't find content,")
    print("the issue is in retrieval (query mismatch) or the system prompt.")
