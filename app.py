import os
import logging
import tempfile
import uuid
import streamlit as st
import numpy as np

from utils.document_processor import process_document
from utils.image_processor import process_image
from utils.video_processor import process_video
from utils.embedding import get_embeddings
from utils.vector_store import VectorStore
from utils.rag import generate_response

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Knowledge Base",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'csv': 'text/csv',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'png': 'image/png',
    'mp4': 'video/mp4',
    'mov': 'video/quicktime',
    'avi': 'video/x-msvideo'
}

def secure_filename(filename):
    """
    Sanitize a filename to be safely stored in the filesystem.
    Replaces dangerous characters with underscores.
    """
    import re
    # Remove potentially dangerous characters
    filename = re.sub(r'[^\w\s.-]', '_', filename).strip()
    # Ensure it doesn't start with a dot (hidden files in Unix)
    if filename.startswith('.'):
        filename = f"file_{filename}"
    return filename

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(filename):
    """Get file type based on extension."""
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in ['pdf', 'docx', 'csv']:
        return 'document'
    elif ext in ['jpg', 'jpeg', 'png']:
        return 'image'
    elif ext in ['mp4', 'mov', 'avi']:
        return 'video'
    return None

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()

def display_chat_history():
    """Display the chat history."""
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat['query'])
        with st.chat_message("assistant"):
            st.write(chat['response'])

# Sidebar for file upload and document management
with st.sidebar:
    st.title("Knowledge Base")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a file", 
        type=list(ALLOWED_EXTENSIONS.keys()),
        help="Supported formats: PDF, DOCX, CSV, JPG, JPEG, PNG, MP4, MOV, AVI"
    )
    
    # Upload button and processing
    if uploaded_file is not None:
        if st.button("Process File"):
            with st.spinner("Processing file..."):
                try:
                    # Create a unique filename
                    original_filename = secure_filename(uploaded_file.name)
                    file_type = get_file_type(original_filename)
                    file_id = str(uuid.uuid4())
                    
                    if not file_type:
                        st.error("Unsupported file extension")
                    else:
                        # Create a temporary directory for processing
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_filepath = os.path.join(temp_dir, original_filename)
                            
                            # Save uploaded file to temporary location
                            with open(temp_filepath, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Process the file based on its type
                            if file_type == 'document':
                                st.info(f"Processing document: {original_filename}")
                                text_chunks = process_document(temp_filepath, original_filename)
                            elif file_type == 'image':
                                st.info(f"Processing image: {original_filename}")
                                text_chunks = process_image(temp_filepath)
                            elif file_type == 'video':
                                st.info(f"Processing video: {original_filename}")
                                text_chunks = process_video(temp_filepath)
                            
                            # Make sure we have something to work with
                            if not text_chunks or len(text_chunks) == 0:
                                st.error("No content could be extracted from the file")
                            else:
                                # Get embeddings for each text chunk
                                successful_embeddings = 0
                                for i, chunk in enumerate(text_chunks):
                                    try:
                                        # Skip empty chunks
                                        if not chunk or not chunk.strip():
                                            continue
                                        
                                        # Generate embedding
                                        embedding = get_embeddings(chunk)
                                        
                                        # Add to vector store with metadata
                                        st.session_state.vector_store.add_embedding(
                                            embedding, 
                                            {
                                                'text': chunk,
                                                'file_id': file_id,
                                                'file_name': original_filename,
                                                'chunk_id': i
                                            }
                                        )
                                        successful_embeddings += 1
                                    except Exception as e:
                                        logger.error(f"Error embedding chunk {i}: {str(e)}")
                                
                                # Check if we have any successful embeddings
                                if successful_embeddings == 0:
                                    st.error("Failed to create embeddings for the document")
                                else:
                                    # Update session documents
                                    st.session_state.documents.append({
                                        'id': file_id,
                                        'name': original_filename,
                                        'type': file_type
                                    })
                                    
                                    st.success(f"Successfully processed {original_filename} with {successful_embeddings} embeddings from {len(text_chunks)} chunks.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    logger.error(f"Error processing file: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
    
    # Document management
    st.subheader("Uploaded Documents")
    if not st.session_state.documents:
        st.info("No documents uploaded yet")
    else:
        for doc in st.session_state.documents:
            st.text(f"{doc['name']} ({doc['type']})")
    
    # Clear all button
    if st.session_state.documents and st.button("Clear All Documents"):
        st.session_state.documents = []
        st.session_state.vector_store.clear()
        st.session_state.chat_history = []
        st.success("All documents and chat history cleared!")
        st.rerun()

# Main page content
st.title("Document Query System")

# Instructions
st.markdown("""
Upload your documents, images, or videos in the sidebar to create a knowledge base.
Then ask questions about your documents in the chat below.
""")

# Display chat history
display_chat_history()

# Chat input
if user_query := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history immediately
    with st.chat_message("user"):
        st.write(user_query)
    
    # Process the query
    with st.spinner("Thinking..."):
        try:
            # Generate embedding for query
            query_embedding = get_embeddings(user_query, timeout=60)
            
            # Search for relevant documents
            search_results = st.session_state.vector_store.search(query_embedding, k=5)
            
            # Generate response
            if not search_results:
                response = "I don't have enough information to answer that question. Try uploading relevant documents first."
                sources = []
            else:
                # Extract context texts
                context_texts = [result['metadata']['text'] for result in search_results]
                context = "\n\n".join(context_texts)
                
                # Generate response
                response = generate_response(user_query, context)
                
                # Get source documents for citation
                sources = []
                for result in search_results:
                    if result['metadata']['file_name'] not in sources:
                        sources.append(result['metadata']['file_name'])
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response)
                if sources:
                    st.write("Sources:")
                    for source in sources:
                        st.write(f"- {source}")
            
            # Update chat history
            st.session_state.chat_history.append({
                'query': user_query,
                'response': response
            })
            
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
            logger.error(f"Error in chat: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
