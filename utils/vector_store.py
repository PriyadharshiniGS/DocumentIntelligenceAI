import os
import logging
import numpy as np
import faiss
import pickle
import tempfile
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store for document embeddings using FAISS.
    """
    
    def __init__(self, dimension=1536):  # Default dimension for OpenAI embeddings
        """
        Initialize the vector store.
        
        Args:
            dimension (int): Dimension of the embedding vectors
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance index
        self.metadata = []  # Store metadata alongside vectors
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.temp_dir, "faiss_index.pkl")
        self.metadata_path = os.path.join(self.temp_dir, "metadata.pkl")
        
        logger.debug(f"Initialized vector store with dimension {dimension}")
    
    def add_embedding(self, embedding, metadata=None):
        """
        Add an embedding vector to the index.
        
        Args:
            embedding (numpy.ndarray): Embedding vector
            metadata (dict): Metadata associated with the embedding
        """
        try:
            # Ensure embedding is the right shape
            if embedding.shape != (self.dimension,):
                raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {embedding.shape[0]}")
            
            # Add to FAISS index
            self.index.add(np.array([embedding], dtype=np.float32))
            
            # Store metadata
            if metadata is None:
                metadata = {}
            self.metadata.append(metadata)
            
            # Save updated index and metadata
            self._save_state()
            
            logger.debug(f"Added embedding with metadata: {metadata.get('file_name', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error adding embedding: {str(e)}")
            raise
    
    def search(self, query_embedding, k=5):
        """
        Search for similar embeddings.
        
        Args:
            query_embedding (numpy.ndarray): Query embedding vector
            k (int): Number of results to return
        
        Returns:
            list: List of dictionaries with search results and metadata
        """
        try:
            # Ensure we don't try to retrieve more items than we have
            k = min(k, self.index.ntotal)
            
            # If index is empty, return empty results
            if self.index.ntotal == 0:
                return []
            
            # Ensure query_embedding is the right shape
            if query_embedding.shape != (self.dimension,):
                raise ValueError(f"Query embedding dimension mismatch: expected {self.dimension}, got {query_embedding.shape[0]}")
            
            # Search the index
            distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
            
            # Prepare results
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx != -1:  # -1 indicates no match
                    results.append({
                        'score': float(distances[0][i]),
                        'metadata': self.metadata[idx]
                    })
            
            logger.debug(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            raise
    
    def clear(self):
        """Clear the vector store."""
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            self._save_state()
            logger.debug("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise
    
    def _save_state(self):
        """Save the current state of the index and metadata."""
        try:
            # Save the FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save the metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
                
            logger.debug(f"State saved: {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
    
    def _load_state(self):
        """Load the index and metadata from files."""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Load the FAISS index
                self.index = faiss.read_index(self.index_path)
                
                # Load the metadata
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                logger.debug(f"State loaded: {self.index.ntotal} vectors")
            else:
                logger.debug("No saved state found")
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
