import os
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

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
        # Create a system message that instructs the model to answer based on provided context
        system_message = """
        You are a helpful assistant that answers questions based only on the provided context.
        If the context doesn't contain the answer, acknowledge that you don't have enough information rather than making up an answer.
        Always provide specific details from the context to support your answer.
        Be clear, direct, and informative in your responses.
        """
        
        # Create user message with context
        user_message = f"""
        Context:
        {context}
        
        Question: {query}
        """
        
        # Call OpenAI API for completion
        response = openai.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.5,  # Lower temperature for more factual responses
            max_tokens=500
        )
        
        # Extract and return the generated response
        answer = response.choices[0].message.content
        
        logger.debug(f"Generated response for query: {query[:50]}...")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise
