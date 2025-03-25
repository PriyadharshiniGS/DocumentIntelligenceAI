import os
import logging
import numpy as np
import anthropic
from anthropic import Anthropic
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Anthropic client
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = Anthropic(
    api_key=ANTHROPIC_API_KEY,
)

def get_embeddings(text):
    """
    Generate embeddings for the input text using Anthropic Claude API.
    
    Args:
        text (str): Input text for which to generate embeddings
    
    Returns:
        numpy.ndarray: Embedding vector
    """
    try:
        # Clean and prepare text
        if not text or not text.strip():
            raise ValueError("Empty text provided for embedding")
        
        text = text.replace("\n", " ").strip()
        
        # Generate embeddings using Anthropic's embeddings API
        response = client.embeddings.create(
            model="claude-3-5-sonnet-20241022-embeddings",
            input=text
        )
        
        # Extract embedding from response
        embedding = response.embedding
        
        return np.array(embedding, dtype=np.float32)
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise
