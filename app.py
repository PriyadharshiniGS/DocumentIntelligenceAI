import os
import logging
import tempfile
import uuid
import streamlit as st
import numpy as np
import base64
from pathlib import Path

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
    page_title="Smart Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
def load_css():
    css_file = Path("css/styles.css")
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load CSS
load_css()

# Emoji Icons for different file types
FILE_ICONS = {
    "document": "📄",
    "image": "🖼️",
    "video": "🎬",
}

# Helper function to create colorful badges
def file_type_badge(file_type):
    badge_class = ""
    if file_type == "document":
        badge_class = "document-badge"
    elif file_type == "image":
        badge_class = "image-badge"
    elif file_type == "video":
        badge_class = "video-badge"
    
    return f'<span class="file-badge {badge_class}">{file_type.upper()}</span>'

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

# Sidebar for file upload and document management
with st.sidebar:
    st.markdown('<h1 class="sidebar-title">📚 Knowledge Library</h1>', unsafe_allow_html=True)
    
    # File uploader with custom styling
    uploaded_file = st.file_uploader(
        "Upload a file", 
        type=list(ALLOWED_EXTENSIONS.keys()),
        help="Supported formats: PDF, DOCX, CSV, JPG, JPEG, PNG, MP4, MOV, AVI"
    )
    
    # Upload button and processing
    if uploaded_file is not None:
        process_button = st.button("📤 Process File", use_container_width=True, type="primary")
        if process_button:
            with st.spinner("🔄 Processing your file..."):
                try:
                    # Create a unique filename
                    original_filename = secure_filename(uploaded_file.name)
                    file_type = get_file_type(original_filename)
                    file_id = str(uuid.uuid4())
                    
                    if not file_type:
                        st.error("❌ Unsupported file extension")
                    else:
                        # Create a temporary directory for processing
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_filepath = os.path.join(temp_dir, original_filename)
                            
                            # Save uploaded file to temporary location
                            with open(temp_filepath, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Process the file based on its type
                            file_icon = FILE_ICONS.get(file_type, "📄")
                            if file_type == 'document':
                                st.info(f"{file_icon} Processing document: {original_filename}")
                                text_chunks = process_document(temp_filepath, original_filename)
                            elif file_type == 'image':
                                st.info(f"{file_icon} Processing image: {original_filename}")
                                text_chunks = process_image(temp_filepath)
                            elif file_type == 'video':
                                st.info(f"{file_icon} Processing video: {original_filename}")
                                text_chunks = process_video(temp_filepath)
                            
                            # Make sure we have something to work with
                            if not text_chunks or len(text_chunks) == 0:
                                st.error("❌ No content could be extracted from the file")
                            else:
                                # Get embeddings for each text chunk
                                with st.status("Generating embeddings...", expanded=True) as status:
                                    successful_embeddings = 0
                                    total_chunks = len(text_chunks)
                                    
                                    for i, chunk in enumerate(text_chunks):
                                        try:
                                            # Skip empty chunks
                                            if not chunk or not chunk.strip():
                                                continue
                                            
                                            status.update(label=f"Processing chunk {i+1}/{total_chunks}")
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
                                    
                                    status.update(label="Embedding generation complete!", state="complete")
                                
                                # Check if we have any successful embeddings
                                if successful_embeddings == 0:
                                    st.error("❌ Failed to create embeddings for the document")
                                else:
                                    # Update session documents
                                    st.session_state.documents.append({
                                        'id': file_id,
                                        'name': original_filename,
                                        'type': file_type
                                    })
                                    
                                    st.success(f"✅ Successfully processed {original_filename} with {successful_embeddings} embeddings from {len(text_chunks)} chunks.")
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    logger.error(f"Error processing file: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
    
    # Document management
    st.markdown('<div class="document-list">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #7792E3; margin-bottom: 15px;">📋 Your Knowledge Base</h3>', unsafe_allow_html=True)
    
    if not st.session_state.documents:
        st.markdown('<div style="text-align: center; padding: 20px; opacity: 0.7;">No documents uploaded yet.<br>Upload files to start building your knowledge base.</div>', unsafe_allow_html=True)
    else:
        for doc in st.session_state.documents:
            file_icon = FILE_ICONS.get(doc['type'], "📄")
            badge = file_type_badge(doc['type'])
            st.markdown(f"""
            <div class="document-item">
                <span class="document-icon">{file_icon}</span>
                <span>{doc['name']}</span>
                {badge}
            </div>
            """, unsafe_allow_html=True)
    
    # Clear all button
    if st.session_state.documents:
        st.markdown('<div style="padding-top: 10px;">', unsafe_allow_html=True)
        if st.button("🗑️ Clear All Documents", use_container_width=True, type="secondary"):
            st.session_state.documents = []
            st.session_state.vector_store.clear()
            st.session_state.chat_history = []
            st.success("🔄 All documents and chat history cleared!")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close document-list

# Main page content
st.markdown('<h1 class="main-title">🧠 Smart Knowledge Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your interactive AI-powered knowledge base using Anthropic\'s Claude</p>', unsafe_allow_html=True)

# Instructions
st.markdown("""
<div class="instructions-box">
    <h3>💡 How to Use</h3>
    <ol>
        <li><strong>Upload your content</strong> - Add documents, images, or videos to your personal knowledge base</li>
        <li><strong>Ask questions</strong> - The AI will search through your content and provide relevant answers</li>
        <li><strong>Explore knowledge</strong> - Get insights and information from your uploaded materials</li>
    </ol>
    <p style="margin-top: 10px; font-style: italic; opacity: 0.8;">Powered by Anthropic's Claude AI with RAG (Retrieval Augmented Generation) technology</p>
</div>
""", unsafe_allow_html=True)

# Display chat history with enhanced styling
def display_chat_history():
    """Display the chat history with improved styling."""
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align: center; padding: 50px 20px; opacity: 0.7;">
            <div style="font-size: 60px; margin-bottom: 20px;">💬</div>
            <h3>Your conversation will appear here</h3>
            <p>Upload documents and start asking questions!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, chat in enumerate(st.session_state.chat_history):
            # User message with custom styling
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"<div class='user-message'>{chat['query']}</div>", unsafe_allow_html=True)
            
            # Assistant message with custom styling
            with st.chat_message("assistant", avatar="🤖"):
                response_parts = chat['response'].split("Sources:")
                main_response = response_parts[0]
                
                st.markdown(f"<div class='assistant-message'>{main_response}</div>", unsafe_allow_html=True)
                
                # Display sources if available
                if len(response_parts) > 1:
                    source_text = response_parts[1]
                    st.markdown(f"<div class='source-citation'><strong>Sources:</strong>{source_text}</div>", unsafe_allow_html=True)

# Call the custom display function
display_chat_history()

# Chat input with enhanced styling
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
if user_query := st.chat_input("Ask a question about your documents...", key="chat_input"):
    # Add user message to chat history immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"<div class='user-message'>{user_query}</div>", unsafe_allow_html=True)
    
    # Process the query
    with st.spinner("🤔 Thinking..."):
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
            
            # Display assistant response with custom styling
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)
                
                if sources:
                    source_list = "<br>".join([f"• {source}" for source in sources])
                    st.markdown(f"<div class='source-citation'><strong>Sources:</strong><br>{source_list}</div>", unsafe_allow_html=True)
            
            # Update chat history
            st.session_state.chat_history.append({
                'query': user_query,
                'response': response + ("\nSources: " + ", ".join(sources) if sources else "")
            })
            
        except Exception as e:
            st.error(f"❌ Error processing your question: {str(e)}")
            logger.error(f"Error in chat: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
