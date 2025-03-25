import os
import logging
import base64
from PIL import Image
import io
import pytesseract
import anthropic
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Anthropic client
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = Anthropic(
    api_key=ANTHROPIC_API_KEY,
)

def process_image(file_path):
    """
    Process image file to extract text and visual information.
    
    Args:
        file_path (str): Path to the image file
    
    Returns:
        list: List of text chunks containing extracted information
    """
    try:
        # Validate and resize image
        try:
            with Image.open(file_path) as img:
                # Convert to RGB if needed
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Resize if either dimension is greater than 1000px
                if img.width > 1000 or img.height > 1000:
                    ratio = min(1000/img.width, 1000/img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save optimized image
                img.save(file_path, 'JPEG', quality=85, optimize=True)
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return ["Error: Unable to process image format"]

        # Get OCR text with timeout
        try:
            ocr_text = extract_text_with_ocr(file_path)
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            ocr_text = ""

        # Get vision description with timeout
        try:
            vision_description = analyze_image_with_claude(file_path)
        except Exception as e:
            logger.error(f"Vision API error: {str(e)}")
            vision_description = ""
        
        results = []
        if ocr_text and ocr_text.strip():
            results.append(f"Text extracted from image:\n{ocr_text}")
        
        if vision_description:
            results.append(f"Image description:\n{vision_description}")
            
        if not results:
            results.append("No meaningful content could be extracted from this image.")
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def extract_text_with_ocr(file_path):
    """
    Extract text from image using OCR.
    
    Args:
        file_path (str): Path to the image file
    
    Returns:
        str: Extracted text
    """
    try:
        # Open the image
        image = Image.open(file_path)
        
        # Extract text using Tesseract OCR
        text = pytesseract.image_to_string(image)
        
        return text
    
    except Exception as e:
        logger.error(f"OCR extraction error: {str(e)}")
        return ""

def analyze_image_with_claude(file_path):
    """
    Analyze image content using Claude's Vision capabilities.
    
    Args:
        file_path (str): Path to the image file
    
    Returns:
        str: Description of the image
    """
    try:
        # Read the image file
        with open(file_path, "rb") as img_file:
            image_data = img_file.read()
        
        # Set up the Claude Vision request
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022", # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image in detail. Describe what you see, including objects, people, scenes, text, and any other relevant information. Be thorough but concise."
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.b64encode(image_data).decode('utf-8')
                            }
                        }
                    ]
                }
            ]
        )
        
        # Extract the description
        description = response.content[0].text
        
        return description
    
    except Exception as e:
        logger.error(f"Claude Vision analysis error: {str(e)}")
        return ""
