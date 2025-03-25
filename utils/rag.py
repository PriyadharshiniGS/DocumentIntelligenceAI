import os
import logging
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

def generate_response(query, context):
    """
    Generate a response based on user query and retrieved context using RAG.
    
    Args:
        query (str): User's question
        context (str): Retrieved document context
    
    Returns:
        str: Generated response
    """
    try:
        # Create system message and prompt for Claude
        system_message = """
        You are a helpful assistant that answers questions based only on the provided context.
        If the context doesn't contain the answer, acknowledge that you don't have enough information rather than making up an answer.
        Always provide specific details from the context to support your answer.
        Be clear, direct, and informative in your responses.
        """
        
        # Create full message with context
        user_message = f"""
        Context:
        {context}
        
        Question: {query}
        """
        
        # Call Anthropic Claude API for completion
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022", # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            system=system_message,
            messages=[
                {"role": "user", "content": user_message}
            ],
            temperature=0.5,  # Lower temperature for more factual responses
            max_tokens=500
        )
        
        # Extract and return the generated response
        answer = response.content[0].text
        
        logger.debug(f"Generated response for query: {query[:50]}...")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise
