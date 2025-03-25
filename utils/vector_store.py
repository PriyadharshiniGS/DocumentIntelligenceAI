import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    In-memory vector store for storing and retrieving document embeddings.
    
    This class provides functionality to:
    - Add embeddings with associated metadata
    - Search for similar embeddings using cosine similarity
    - Clear the store
    """
    
    def __init__(self):
        """Initialize an empty vector store."""
        self.embeddings = []
        self.metadata = []
        logger.debug("Initialized empty vector store")
    
    def add_embedding(self, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """
        Add an embedding vector with associated metadata to the store.
        
        Args:
            embedding: List of embedding values
            metadata: Dictionary of metadata associated with the embedding
        """
        try:
            # Convert to numpy array for efficient computation
            embedding_array = np.array(embedding)
            
            # Store embedding and metadata
            self.embeddings.append(embedding_array)
            self.metadata.append(metadata)
            
            logger.debug(f"Added embedding with metadata: {metadata.get('file_name', 'unknown')}")
        except Exception as e:
            logger.error(f"Error adding embedding: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the k most similar embeddings to the query embedding.
        
        Args:
            query_embedding: The query embedding to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries containing similarity scores and metadata
        """
        try:
            # Handle empty store
            if not self.embeddings:
                logger.debug("Search performed on empty vector store")
                return []
            
            # Convert query to numpy array
            query_array = np.array(query_embedding)
            
            # Calculate cosine similarity
            similarities = self._calculate_similarities(query_array)
            
            # Get indices of top k results
            top_indices = similarities.argsort()[-k:][::-1]
            
            # Prepare results
            results = []
            for idx in top_indices:
                results.append({
                    'similarity': float(similarities[idx]),
                    'metadata': self.metadata[idx]
                })
            
            logger.debug(f"Search returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def clear(self) -> None:
        """Clear all embeddings and metadata from the store."""
        try:
            self.embeddings = []
            self.metadata = []
            logger.debug("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
    
    def _calculate_similarities(self, query_array: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarities between query and all stored embeddings.
        
        Args:
            query_array: Numpy array of query embedding
            
        Returns:
            Numpy array of similarity scores
        """
        # Convert list of embeddings to 2D array for vectorized operations
        embeddings_array = np.array(self.embeddings)
        
        # Calculate dot product
        dot_product = np.dot(embeddings_array, query_array)
        
        # Calculate norms
        query_norm = np.linalg.norm(query_array)
        embedding_norms = np.linalg.norm(embeddings_array, axis=1)
        
        # Calculate cosine similarity
        similarities = dot_product / (embedding_norms * query_norm)
        
        return similarities
