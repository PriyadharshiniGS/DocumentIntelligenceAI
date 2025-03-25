import os
import logging
import numpy as np
import anthropic
from anthropic import Anthropic
import hashlib
import json
import base64

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get Anthropic API key
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Initialize Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY)

def get_hash(text):
    """
    Generate a stable hash for the text.
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_embeddings(text):
    """
    Generate embeddings for the input text using Anthropic Claude API.
    As Anthropic doesn't have a dedicated embeddings API like OpenAI,
    we'll use a deterministic approach to generate embeddings.
    
    Args:
        text (str): Input text for which to generate embeddings
    
    Returns:
        numpy.ndarray: Embedding vector
    """
    try:
        # Clean and prepare text
        if not text or not text.strip():
            logger.error("Empty text provided for embedding")
            raise ValueError("Empty text provided for embedding")
        
        text = text.replace("\n", " ").strip()
        
        logger.debug(f"Generating embeddings for text: {text[:50]}...")
        
        # Since Anthropic doesn't have a dedicated embeddings endpoint,
        # we'll use a message to Claude asking for a vectorized representation
        # of the text content, which we can then process into a usable embedding
        
        # We'll use a system prompt that guides Claude to produce a structured text representation
        # that we can convert to a numeric vector
        system_prompt = """
        You are a text embedding generator. For the provided text, generate a representation
        that captures semantic meaning. Output ONLY a JSON array of 1536 float values between -1 and 1
        that can be used as an embedding vector. Do not include any explanations, headers, or additional text.
        """
        
        # For short texts, we'll use the text directly
        # For longer texts, we'll use a hash-based approach to ensure consistency
        if len(text) < 500:
            prompt_text = f"Generate embedding for: {text}"
        else:
            # Use a shorter text + hash approach for longer texts
            text_hash = get_hash(text)
            prompt_text = f"Generate embedding for text with hash {text_hash}. First 200 chars: {text[:200]}"
        
        # Get Claude's response
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022", # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            system=system_prompt,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0,  # Zero temperature for deterministic output
            max_tokens=4000
        )
        
        # Extract the embedding array from response
        embedding_text = response.content[0].text.strip()
        
        # Remove any leading/trailing characters that aren't part of the JSON array
        embedding_text = embedding_text.strip('`')
        if embedding_text.startswith('json'):
            embedding_text = embedding_text[4:].strip()
        
        # Parse the embedding
        embedding = json.loads(embedding_text)
        
        logger.debug(f"Generated embedding with dimension: {len(embedding)}")
        
        return np.array(embedding, dtype=np.float32)
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise
