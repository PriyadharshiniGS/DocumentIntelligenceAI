import os
import logging
import base64
from PIL import Image
import io
import pytesseract
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

def process_image(file_path):
    """
    Process image file to extract text and visual information.
    
    Args:
        file_path (str): Path to the image file
    
    Returns:
        list: List of text chunks containing extracted information
    """
    try:
        # First, try OCR with Tesseract to extract any text
        ocr_text = extract_text_with_ocr(file_path)
        
        # Then, use OpenAI's GPT-4 Vision to analyze the image
        vision_description = analyze_image_with_gpt4(file_path)
        
        # Combine results
        results = []
        
        if ocr_text.strip():
            results.append(f"Text extracted from image:\n{ocr_text}")
        
        if vision_description:
            results.append(f"Image description:\n{vision_description}")
        
        # If no results were obtained, return a fallback message
        if not results:
            results.append("No text or meaningful content could be extracted from this image.")
        
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

def analyze_image_with_gpt4(file_path):
    """
    Analyze image content using GPT-4 Vision.
    
    Args:
        file_path (str): Path to the image file
    
    Returns:
        str: Description of the image
    """
    try:
        # Convert image to base64
        with open(file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Call OpenAI API to analyze the image
        response = openai.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image in detail. Describe what you see, including objects, people, scenes, text, and any other relevant information. Be thorough but concise."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        # Extract the description
        description = response.choices[0].message.content
        
        return description
    
    except Exception as e:
        logger.error(f"GPT-4 Vision analysis error: {str(e)}")
        return ""
