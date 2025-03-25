import os
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_document(file_path: str, filename: str) -> List[str]:
    """
    Process a document file (PDF, DOCX, CSV) and extract text chunks.
    
    Args:
        file_path: Path to the document file
        filename: Original filename
        
    Returns:
        List of text chunks extracted from the document
    """
    try:
        # Extract file extension
        _, ext = os.path.splitext(filename)
        ext = ext.lower().strip('.')
        
        if ext == 'pdf':
            return _process_pdf(file_path)
        elif ext == 'docx':
            return _process_docx(file_path)
        elif ext == 'csv':
            return _process_csv(file_path)
        else:
            logger.error(f"Unsupported document format: {ext}")
            return []
    
    except Exception as e:
        logger.error(f"Error processing document {filename}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def _process_pdf(file_path: str) -> List[str]:
    """Extract text from a PDF file."""
    try:
        import PyPDF2
        
        chunks = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text and text.strip():
                    # Simple chunking by page
                    chunks.append(f"Page {page_num + 1}: {text}")
        
        return chunks
    
    except ImportError:
        logger.error("PyPDF2 is required for processing PDF files")
        return ["Error: PyPDF2 library not found. Unable to process PDF."]
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return []

def _process_docx(file_path: str) -> List[str]:
    """Extract text from a DOCX file."""
    try:
        from docx import Document
        
        chunks = []
        doc = Document(file_path)
        
        # Process paragraphs
        current_chunk = ""
        for para in doc.paragraphs:
            if para.text and para.text.strip():
                # Append paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n"
                current_chunk += para.text
                
                # If chunk is getting large, add it and start a new one
                if len(current_chunk) > 1000:
                    chunks.append(current_chunk)
                    current_chunk = ""
        
        # Add any remaining text
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    except ImportError:
        logger.error("python-docx is required for processing DOCX files")
        return ["Error: python-docx library not found. Unable to process DOCX."]
    except Exception as e:
        logger.error(f"Error processing DOCX: {str(e)}")
        return []

def _process_csv(file_path: str) -> List[str]:
    """Extract text from a CSV file."""
    try:
        import csv
        import pandas as pd
        
        # First try with pandas for better handling
        try:
            df = pd.read_csv(file_path)
            # Convert each row to a string representation
            chunks = [df.iloc[i:i+10].to_string(index=False) for i in range(0, len(df), 10)]
            return chunks
        except:
            # Fallback to basic CSV reading
            chunks = []
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                headers = next(csv_reader, None)
                
                if headers:
                    current_rows = [",".join(headers)]
                    row_count = 0
                    
                    for row in csv_reader:
                        current_rows.append(",".join(row))
                        row_count += 1
                        
                        # Group rows into chunks
                        if row_count >= 10:
                            chunks.append("\n".join(current_rows))
                            current_rows = [",".join(headers)]
                            row_count = 0
                    
                    # Add any remaining rows
                    if len(current_rows) > 1:
                        chunks.append("\n".join(current_rows))
            
            return chunks
    
    except ImportError:
        logger.error("Pandas is recommended for processing CSV files")
        return ["Error: Unable to process CSV file properly."]
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        return []
