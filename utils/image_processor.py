import os
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_image(file_path: str) -> List[str]:
    """
    Process an image file and extract text or descriptions.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        List of text chunks extracted from the image
    """
    try:
        # Attempt to extract text from image using OCR
        ocr_text = _extract_text_ocr(file_path)
        
        # If OCR fails or returns no text, generate a description
        if not ocr_text:
            description = _generate_image_description(file_path)
            if description:
                return [description]
            else:
                return ["No text content could be extracted from this image."]
        
        return [ocr_text]
    
    except Exception as e:
        logger.error(f"Error processing image {file_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def _extract_text_ocr(file_path: str) -> str:
    """Extract text from an image using OCR."""
    try:
        # Try Tesseract OCR if available
        import pytesseract
        from PIL import Image
        
        # Open the image
        img = Image.open(file_path)
        
        # Perform OCR
        text = pytesseract.image_to_string(img)
        
        if text and text.strip():
            return text.strip()
        return ""
    
    except ImportError:
        logger.warning("pytesseract not available, OCR functionality limited")
        return ""
    except Exception as e:
        logger.error(f"OCR processing error: {str(e)}")
        return ""

def _generate_image_description(file_path: str) -> str:
    """Generate a description of the image content using Anthropic's Claude Vision."""
    try:
        # Import Anthropic's client
        from utils.anthropic_client import analyze_image
        
        # Generate description using Anthropic Claude
        description = analyze_image(file_path, "Describe this image in detail, including any visible text.")
        
        return description
    
    except ImportError:
        logger.warning("Anthropic package not available, image description unavailable")
        return "Image description unavailable: required libraries missing."
    except Exception as e:
        logger.error(f"Image description error: {str(e)}")
        return f"Image description unavailable: {str(e)}"
