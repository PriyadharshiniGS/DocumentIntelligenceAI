import os
import logging
import numpy as np
import requests
import json
import openai

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get API keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize OpenAI client for embeddings
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_embeddings(text):
    """
    Generate embeddings for the input text using OpenAI API.
    Since Anthropic embeddings are not as easily accessible, 
    we'll use OpenAI's embeddings API instead.
    
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
        
        # Use OpenAI embeddings (more stable API for embeddings)
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        
        # Extract embedding
        embedding = response.data[0].embedding
        
        return np.array(embedding, dtype=np.float32)
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise
