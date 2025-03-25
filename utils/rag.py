import os
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_response(query: str, context: str) -> str:
    """
    Generate a response to a query using the provided context.
    
    Args:
        query: The user's question
        context: Context information retrieved from the vector store
        
    Returns:
        Generated response text
    """
    try:
        # Import Anthropic client utility
        from utils.anthropic_client import generate_text_response
        
        # Generate response using Anthropic
        response = generate_text_response(query, context)
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return "I'm sorry, but I encountered an error while generating a response. Please try again."
