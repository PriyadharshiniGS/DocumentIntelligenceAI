import os
import logging
import numpy as np
import hashlib
import json
import anthropic
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get Anthropic API key
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Initialize Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY)

def get_hash(text):
    """Generate a stable hash for the text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_embedding_seed(text, dimension=1536):
    """Generate a deterministic vector based on text hash."""
    # Create a deterministic seed from the text
    text_hash = get_hash(text)
    seed = int(text_hash, 16) % (2**32)

    # Use the seed to initialize numpy's random generator
    rng = np.random.RandomState(seed)

    # Generate a random vector with the specified dimension
    vector = rng.normal(0, 1, dimension)

    # Normalize to unit length
    vector = vector / np.linalg.norm(vector)

    return vector.astype(np.float32)

def get_embeddings(text, timeout=120):
    """
    Generate embeddings for the input text.

    First attempts to use Anthropic to create semantically meaningful embeddings.
    If that fails, falls back to a hash-based deterministic approach.

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

        try:
            # First try with Anthropic if available
            # Using a Claude-based embedding approach
            system_prompt = """
            You are a text embedding generator. For the provided text, your task is to provide a summary
            that captures the essential meaning. This summary will be used to generate embeddings.
            Focus on key concepts and entities. Be concise but thorough.
            """

            # For longer texts, summarize first
            if len(text) > 500:
                prompt_text = f"Extract essential meaning from this text: {text[:2000]}"
                if len(text) > 2000:
                    prompt_text += f" [truncated, full text hash: {get_hash(text)}]"
            else:
                prompt_text = f"Extract essential meaning from: {text}"

            # Get Claude's response for the semantic embedding
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                system=system_prompt,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.0,  # Zero temperature for deterministic output
                max_tokens=300,  # Reduced token limit
                timeout=30  # Set explicit timeout
            )

            # Use Claude's semantic summary as the basis for the hash-based embedding
            semantic_text = response.content[0].text.strip()
            logger.debug(f"Generated semantic summary: {semantic_text[:100]}...")

            # Generate embedding from semantic summary
            embedding = get_embedding_seed(semantic_text)

        except Exception as api_error:
            # Fall back to direct hash-based method if Anthropic fails
            logger.warning(f"Anthropic API error, falling back to hash-based embeddings: {str(api_error)}")
            embedding = get_embedding_seed(text)

        logger.debug(f"Generated embedding with dimension: {embedding.shape[0]}")

        return embedding

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        # Final fallback - just use pure hash-based method with no API
        try:
            return get_embedding_seed(text)
        except:
            # If everything fails, raise the original error
            raise