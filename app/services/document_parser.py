import io
import os
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, Any
from pypdf import PdfReader
from app.config import MAX_UPLOAD_SIZE_BYTES, ALLOWED_DOCUMENT_EXTENSIONS

def extract_text_from_pdf(file_bytes: bytes) -> Dict[str, Any]:
    """Extracts text content and metadata from PDF files using pypdf."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        num_pages = len(reader.pages)
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text.strip())
        
        full_text = "\n\n".join(extracted_pages).strip()
        if not full_text:
            return {
                "success": False,
                "error": "The PDF file appears to contain only scanned images or non-selectable text. Please use the Screenshot/Image OCR tab."
            }
            
        return {
            "success": True,
            "text": full_text,
            "page_count": num_pages,
            "word_count": len(full_text.split())
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse PDF document: {str(e)}"
        }

def extract_text_from_docx(file_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts text from DOCX files by reading word/document.xml directly from the ZIP archive.
    Fast, reliable, and requires zero external binary dependencies.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx_zip:
            xml_content = docx_zip.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            # XML namespace for WordprocessingML
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            paragraphs = []
            for p in tree.iterfind('.//w:p', ns):
                texts = [node.text for node in p.iterfind('.//w:t', ns) if node.text]
                if texts:
                    paragraphs.append(''.join(texts))
                    
            full_text = '\n\n'.join(paragraphs).strip()
            if not full_text:
                return {
                    "success": False,
                    "error": "The DOCX file did not contain any readable text paragraphs."
                }
                
            return {
                "success": True,
                "text": full_text,
                "word_count": len(full_text.split())
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse DOCX document: {str(e)}"
        }

def extract_text_from_txt(file_bytes: bytes) -> Dict[str, Any]:
    """Decodes plain text files with UTF-8 or Latin-1 fallback."""
    try:
        try:
            text = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text = file_bytes.decode('latin-1')
            
        text = text.strip()
        if not text:
            return {
                "success": False,
                "error": "The text file is empty."
            }
            
        return {
            "success": True,
            "text": text,
            "word_count": len(text.split())
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read text file: {str(e)}"
        }

def parse_document(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Validates document format, extracts text, and returns structured extraction metadata."""
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        return {
            "success": False,
            "error": f"File size exceeds maximum permitted limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
        }

    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_DOCUMENT_EXTENSIONS:
        return {
            "success": False,
            "error": f"Unsupported document format '{ext}'. Allowed extensions are: {', '.join(ALLOWED_DOCUMENT_EXTENSIONS)}"
        }

    if ext == ".pdf":
        result = extract_text_from_pdf(file_bytes)
    elif ext == ".docx":
        result = extract_text_from_docx(file_bytes)
    elif ext == ".txt":
        result = extract_text_from_txt(file_bytes)
    else:
        return {"success": False, "error": "Unknown document type."}

    if result.get("success"):
        return {
            "success": True,
            "filename": filename,
            "file_type": ext.lstrip('.').upper(),
            "extracted_text": result["text"],
            "word_count": result.get("word_count", len(result["text"].split())),
            "page_count": result.get("page_count", 1)
        }
    else:
        return {
            "success": False,
            "filename": filename,
            "error": result.get("error", "Unknown document processing error.")
        }
