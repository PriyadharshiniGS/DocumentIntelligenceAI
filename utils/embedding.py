import os
import logging
import numpy as np
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

def get_embeddings(text):
    """
    Generate embeddings for the input text using OpenAI API.
    
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
        
        # Generate embeddings using OpenAI's embeddings API
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        # Extract embedding from response
        embedding = response.data[0].embedding
        
        return np.array(embedding, dtype=np.float32)
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise
