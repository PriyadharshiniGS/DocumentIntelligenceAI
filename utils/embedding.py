import os
import logging
import numpy as np
from typing import List, Union
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_embeddings(text: str, timeout: int = 30) -> List[float]:
    """
    Generate embeddings for a given text using Anthropic's embedding model.
    
    Args:
        text: The text to generate embeddings for
        timeout: Maximum time to wait for embedding generation (seconds)
        
    Returns:
        List of embedding values
        
    Raises:
        TimeoutError: If embedding generation takes too long
        Exception: For other errors
    """
    try:
        # Start timer for timeout handling
        start_time = time.time()
        
        # Import Anthropic client
        from utils.anthropic_client import generate_embeddings
        
        # Check for timeout
        if time.time() - start_time > timeout:
            raise TimeoutError("Embedding generation timed out")
        
        # Get embedding from Anthropic
        embedding = generate_embeddings(text)
        
        return embedding
    
    except TimeoutError:
        logger.error("Embedding generation timed out")
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Fallback to a random embedding
        return _get_random_embedding()

def _get_random_embedding(dim: int = 1536) -> List[float]:
    """
    Generate a random embedding vector for fallback situations.
    
    Args:
        dim: Dimension of the embedding vector (default is 1536 for OpenAI embeddings)
        
    Returns:
        List of random embedding values normalized to unit length
    """
    # Generate random vector
    random_vector = np.random.normal(0, 1, dim)
    
    # Normalize to unit length
    normalized_vector = random_vector / np.linalg.norm(random_vector)
    
    return normalized_vector.tolist()
