import os
import logging
import tempfile
import uuid
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
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

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Initialize vector store
vector_store = VectorStore()

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

@app.route('/')
def index():
    """Render the main page."""
    # Initialize session if not already done
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'documents' not in session:
        session['documents'] = []
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process a file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Create a unique filename
            original_filename = secure_filename(file.filename)
            file_type = get_file_type(original_filename)
            file_id = str(uuid.uuid4())
            
            # Create a temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_filepath = os.path.join(temp_dir, original_filename)
                file.save(temp_filepath)
                
                # Process the file based on its type
                if file_type == 'document':
                    logger.debug(f"Processing document: {original_filename}")
                    text_chunks = process_document(temp_filepath, original_filename)
                elif file_type == 'image':
                    logger.debug(f"Processing image: {original_filename}")
                    text_chunks = process_image(temp_filepath)
                elif file_type == 'video':
                    logger.debug(f"Processing video: {original_filename}")
                    text_chunks = process_video(temp_filepath)
                else:
                    return jsonify({'error': 'Unsupported file type'}), 400
                
                # Get embeddings for each text chunk
                for i, chunk in enumerate(text_chunks):
                    try:
                        embedding = get_embeddings(chunk)
                        # Add to vector store with metadata
                        vector_store.add_embedding(
                            embedding, 
                            {
                                'text': chunk,
                                'file_id': file_id,
                                'file_name': original_filename,
                                'chunk_id': i
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error embedding chunk {i}: {str(e)}")
                
                # Update session documents
                if 'documents' not in session:
                    session['documents'] = []
                    
                session['documents'].append({
                    'id': file_id,
                    'name': original_filename,
                    'type': file_type
                })
                session.modified = True
                
                return jsonify({
                    'success': True,
                    'filename': original_filename,
                    'fileId': file_id,
                    'fileType': file_type,
                    'chunkCount': len(text_chunks)
                })
                
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    """Process a chat message and generate a response."""
    data = request.json
    query = data.get('message', '')
    
    if not query:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Get query embedding
        query_embedding = get_embeddings(query)
        
        # Search for relevant documents
        search_results = vector_store.search(query_embedding, k=5)
        
        if not search_results:
            return jsonify({
                'answer': "I don't have enough information to answer that question. Try uploading relevant documents first."
            })
        
        # Generate response using RAG
        context_texts = [result['metadata']['text'] for result in search_results]
        context = "\n\n".join(context_texts)
        
        response = generate_response(query, context)
        
        # Update chat history
        if 'chat_history' not in session:
            session['chat_history'] = []
            
        session['chat_history'].append({
            'query': query,
            'response': response
        })
        session.modified = True
        
        # Get source documents for citation
        sources = []
        for result in search_results:
            if result['metadata']['file_name'] not in sources:
                sources.append(result['metadata']['file_name'])
        
        return jsonify({
            'answer': response,
            'sources': sources
        })
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500

@app.route('/documents', methods=['GET'])
def get_documents():
    """Get list of uploaded documents."""
    if 'documents' not in session:
        session['documents'] = []
    return jsonify(session['documents'])

@app.route('/clear', methods=['POST'])
def clear_session():
    """Clear session data and vector store."""
    try:
        session.clear()
        vector_store.clear()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        return jsonify({'error': f'Error clearing session: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
