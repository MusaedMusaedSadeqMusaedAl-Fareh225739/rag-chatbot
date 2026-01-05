# scripts/pdf_to_txt.py
"""
PDF to Text Preprocessing Script
Handles various PDF types: scanned, two-column, mixed content

Features:
- OCR fallback for scanned pages
- Two-column layout detection
- Grayscale to B/W conversion for better OCR
- Noise line filtering
"""

import os
import argparse
from pathlib import Path

try:
    import pdfplumber
    import pytesseract
    from PIL import Image, ImageOps
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pdfplumber pytesseract Pillow")
    exit(1)


def preprocess_image_to_bw(image):
    """Convert image to high-contrast black & white for better OCR."""
    gray = ImageOps.grayscale(image)
    return gray.point(lambda x: 0 if x < 150 else 255, '1')


def extract_standard_page(page, ocr_config):
    """
    Single-block text extraction with OCR fallback.
    
    Args:
        page: pdfplumber page object
        ocr_config: Tesseract configuration string
    
    Returns:
        Extracted text string
    """
    # Try embedded text first
    text = page.extract_text() or ""
    
    # OCR fallback if no embedded text
    if not text.strip():
        img = page.to_image(resolution=300).original.convert("RGB")
        bw = preprocess_image_to_bw(img)
        text = pytesseract.image_to_string(bw, config=ocr_config)
    
    return text


def extract_two_column_page(page, ocr_config, noise_patterns=None):
    """
    Two-column text extraction for documents like A-Z guides.
    
    Args:
        page: pdfplumber page object
        ocr_config: Tesseract configuration string
        noise_patterns: List of strings to filter out
    
    Returns:
        Extracted and cleaned text string
    """
    if noise_patterns is None:
        noise_patterns = []
    
    w, h = page.width, page.height
    mid = w / 2

    def get_col_text(x0, x1):
        """Extract text from a column region."""
        sl = page.within_bbox((x0, 0, x1, h))
        txt = sl.extract_text() or ""
        
        if not txt.strip():
            img = sl.to_image(resolution=300).original.convert("RGB")
            bw = preprocess_image_to_bw(img)
            txt = pytesseract.image_to_string(bw, config=ocr_config)
        
        return txt

    left_txt = get_col_text(0, mid)
    right_txt = get_col_text(mid, w)

    # Filter noise lines
    clean = []
    for line in (left_txt + "\n" + right_txt).splitlines():
        stripped = line.strip()
        if any(pat in stripped for pat in noise_patterns):
            continue
        clean.append(line)
    
    return "\n".join(clean)


def extract_text_from_pdf(pdf_path, two_column_patterns=None, noise_patterns=None):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        two_column_patterns: Filename patterns that indicate two-column layout
        noise_patterns: Strings to filter from two-column extractions
    
    Returns:
        Full extracted text
    """
    if two_column_patterns is None:
        two_column_patterns = []
    if noise_patterns is None:
        noise_patterns = []
    
    ocr_config = r'--oem 3 --psm 6'
    pages_txt = []
    
    fname = os.path.basename(pdf_path)
    is_two_column = any(pat in fname for pat in two_column_patterns)
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            print(f"  Processing page {i+1}/{len(pdf.pages)}...", end="\r")
            
            if is_two_column:
                txt = extract_two_column_page(page, ocr_config, noise_patterns)
            else:
                txt = extract_standard_page(page, ocr_config)
            
            pages_txt.append(txt)
    
    print()  # New line after progress
    return "\n\n".join(pages_txt)


def process_all_pdfs(input_folder, output_folder, two_column_patterns=None, noise_patterns=None):
    """
    Process all PDFs in a folder and save as .txt files.
    
    Args:
        input_folder: Folder containing PDF files
        output_folder: Folder to save extracted text files
        two_column_patterns: Filename patterns for two-column detection
        noise_patterns: Strings to filter from output
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_path.glob("*.pdf")) + list(input_path.glob("*.PDF"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        return
    
    print(f"Found {len(pdf_files)} PDF files\n")
    
    for pdf_file in pdf_files:
        out_file = output_path / (pdf_file.stem + ".txt")
        
        print(f"Processing: {pdf_file.name}")
        
        try:
            txt = extract_text_from_pdf(
                str(pdf_file), 
                two_column_patterns or [],
                noise_patterns or []
            )
            
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(txt)
            
            print(f"  Saved: {out_file.name}\n")
            
        except Exception as e:
            print(f"  Error: {e}\n")
    
    print("Done")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF documents to text files with OCR support"
    )
    parser.add_argument(
        "input", 
        help="Input folder containing PDF files"
    )
    parser.add_argument(
        "output", 
        help="Output folder for text files"
    )
    parser.add_argument(
        "--two-column", 
        nargs="*", 
        default=[],
        help="Filename patterns that indicate two-column layout (e.g., 'A-Z' 'Guide')"
    )
    parser.add_argument(
        "--noise", 
        nargs="*", 
        default=[],
        help="Strings to filter out from extracted text"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("PDF to Text Converter")
    print("=" * 50)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()
    
    process_all_pdfs(
        args.input, 
        args.output, 
        args.two_column, 
        args.noise
    )


if __name__ == "__main__":
    main()
