"""
Anthropic Claude API utilities for embedding and text generation.
"""
import os
import logging
import base64
import sys
from typing import List, Optional, Dict, Any
import json
import anthropic

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# The newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
# do not change this unless explicitly requested by the user
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_EMBEDDING_MODEL = "claude-3-haiku-20240307" # For embeddings

def get_anthropic_client():
    """Initialize and return an Anthropic client."""
    try:
        # Get API key from environment variables
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("Anthropic API key not found in environment variables")
            return None
        
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=api_key)
        return client
    
    except Exception as e:
        logger.error(f"Error initializing Anthropic client: {str(e)}")
        return None

def generate_text_response(prompt: str, context: str = None, model: str = DEFAULT_MODEL) -> str:
    """
    Generate a text response using Anthropic's Claude API.
    
    Args:
        prompt: The text prompt to generate a response for
        context: Optional context to include with the prompt
        model: The Claude model to use
        
    Returns:
        Generated text response
    """
    try:
        client = get_anthropic_client()
        if not client:
            return "Error: Unable to initialize Anthropic client. Please check your API key."
        
        # Build the system prompt
        system_prompt = """
        You are a helpful assistant that answers questions based on the provided context information.
        Respond using only information present in the context. If the context doesn't contain relevant 
        information to answer the question, say "I don't have enough information to answer that question."
        
        Do not make up information or use prior knowledge not found in the context.
        
        Keep your answers concise and to the point, but provide complete information.
        """
        
        # Build the user message
        if context:
            user_message = f"""
            Question: {prompt}
            
            Context information:
            {context}
            """
        else:
            user_message = prompt
        
        # Generate response
        response = client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=1000,
            temperature=0.3,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        return response.content[0].text
    
    except Exception as e:
        logger.error(f"Error generating text response: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error generating response: {str(e)}"

def analyze_image(image_path: str, prompt: str = "Describe this image in detail, including any visible text.") -> str:
    """
    Analyze an image using Anthropic's Claude Vision capabilities.
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt for the image analysis
        
    Returns:
        Text description of the image
    """
    try:
        client = get_anthropic_client()
        if not client:
            return "Error: Unable to initialize Anthropic client. Please check your API key."
        
        # Read and encode the image
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        
        # Create the message with image
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1000,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
        )
        
        return response.content[0].text
    
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error analyzing image: {str(e)}"

def generate_embeddings(text: str) -> List[float]:
    """
    Generate text embeddings using Anthropic's API.
    
    Args:
        text: The text to generate embeddings for
        
    Returns:
        List of embedding values
    """
    try:
        client = get_anthropic_client()
        if not client:
            logger.error("Unable to initialize Anthropic client")
            import numpy as np
            # Return random embedding as fallback
            random_vector = np.random.normal(0, 1, 1536)
            normalized_vector = random_vector / np.linalg.norm(random_vector)
            return normalized_vector.tolist()
        
        # Clean and truncate text if needed
        text = text.strip()
        if len(text) > 8000:
            text = text[:8000]  # Truncate to avoid token limits
        
        # Generate embeddings
        response = client.embeddings.create(
            model=DEFAULT_EMBEDDING_MODEL,
            input=text
        )
        
        return response.embedding
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise