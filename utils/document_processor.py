import os
import logging
import io
import PyPDF2
import docx
import pandas as pd
from PIL import Image
import pytesseract
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_document(file_path, file_name):
    """
    Process document based on file type and extract text content.
    
    Args:
        file_path (str): Path to the document file
        file_name (str): Original filename
    
    Returns:
        list: List of text chunks extracted from document
    """
    try:
        extension = file_name.split('.')[-1].lower()
        
        if extension == 'pdf':
            return process_pdf(file_path)
        elif extension == 'docx':
            return process_docx(file_path)
        elif extension == 'csv':
            return process_csv(file_path)
        else:
            raise ValueError(f"Unsupported document type: {extension}")
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

def process_pdf(file_path):
    """
    Extract text from PDF file and split into chunks.
    
    Args:
        file_path (str): Path to the PDF file
    
    Returns:
        list: List of text chunks extracted from PDF
    """
    chunks = []
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            # First pass: extract all text
            all_text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text + "\n\n"
            
            # Check if we got meaningful text
            if len(all_text.strip()) < 100 and num_pages > 0:
                logger.info("PDF text extraction yielded limited text. Trying OCR...")
                # If we didn't get much text, the PDF might be scanned images
                # Use Tesseract OCR with PDF2Image
                try:
                    import pdf2image
                    images = pdf2image.convert_from_path(file_path)
                    for i, image in enumerate(images):
                        page_text = pytesseract.image_to_string(image)
                        all_text += f"Page {i+1}:\n{page_text}\n\n"
                except ImportError:
                    logger.warning("pdf2image not available for OCR fallback")
            
            # Split into chunks of about 1000 words each
            chunks = split_text_into_chunks(all_text)
            
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise
    
    return chunks

def process_docx(file_path):
    """
    Extract text from DOCX file and split into chunks.
    
    Args:
        file_path (str): Path to the DOCX file
    
    Returns:
        list: List of text chunks extracted from DOCX
    """
    try:
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        tables_text = []
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = ' '.join([cell.text for cell in row.cells if cell.text.strip()])
                if row_text:
                    tables_text.append(row_text)
        
        # Combine and split into chunks
        all_text = "\n".join(paragraphs) + "\n" + "\n".join(tables_text)
        chunks = split_text_into_chunks(all_text)
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error processing DOCX: {str(e)}")
        raise

def process_csv(file_path):
    """
    Extract content from CSV file and convert to text description.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        list: List of text chunks describing CSV content
    """
    try:
        df = pd.read_csv(file_path)
        
        # Create a text description of the CSV data
        overview = f"CSV file with {len(df)} rows and {len(df.columns)} columns.\n"
        overview += f"Column names: {', '.join(df.columns.tolist())}\n\n"
        
        # Add data sample (first few rows)
        sample_size = min(5, len(df))
        sample_rows = []
        for i in range(sample_size):
            row = df.iloc[i]
            row_str = ", ".join([f"{col}: {row[col]}" for col in df.columns])
            sample_rows.append(f"Row {i+1}: {row_str}")
        
        overview += "Sample data:\n" + "\n".join(sample_rows) + "\n\n"
        
        # Add summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            overview += "Summary statistics for numeric columns:\n"
            for col in numeric_cols:
                stats = df[col].describe()
                overview += f"{col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, std={stats['std']:.2f}\n"
        
        # Process data in chunks if large
        chunks = [overview]
        
        # If the CSV has many rows, process additional chunks
        if len(df) > 10:
            chunk_size = 50
            for i in range(0, len(df), chunk_size):
                end = min(i + chunk_size, len(df))
                chunk_df = df.iloc[i:end]
                
                chunk_text = f"CSV data rows {i+1} to {end}:\n"
                for j, row in chunk_df.iterrows():
                    row_str = ", ".join([f"{col}: {row[col]}" for col in df.columns])
                    chunk_text += f"Row {j+1}: {row_str}\n"
                
                chunks.append(chunk_text)
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        raise

def split_text_into_chunks(text, words_per_chunk=1000):
    """
    Split text into chunks of approximately specified number of words.
    
    Args:
        text (str): Text to split
        words_per_chunk (int): Target number of words per chunk
    
    Returns:
        list: List of text chunks
    """
    # Clean the text - remove excessive whitespace and line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    
    # If text is small enough, return as single chunk
    if len(words) <= words_per_chunk:
        return [text]
    
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i + words_per_chunk]
        chunks.append(' '.join(chunk_words))
    
    return chunks
