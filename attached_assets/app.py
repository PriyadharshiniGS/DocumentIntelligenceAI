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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB

# Initialize vector store
vector_store = VectorStore()

# Configure gunicorn timeout via environment variable
if 'GUNICORN_TIMEOUT' not in os.environ:
    os.environ['GUNICORN_TIMEOUT'] = '300'  # 5 minutes timeout

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
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        # Create a unique filename
        original_filename = secure_filename(file.filename)
        file_type = get_file_type(original_filename)
        file_id = str(uuid.uuid4())

        if not file_type:
            return jsonify({'error': 'Unsupported file extension'}), 400

        # Log the file upload attempt
        logger.debug(f"Attempting to process file: {original_filename} of type {file_type}")

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

            # Make sure we have something to work with
            if not text_chunks or len(text_chunks) == 0:
                return jsonify({'error': 'No content could be extracted from the file'}), 400

            # Log successful text extraction
            logger.debug(f"Successfully extracted {len(text_chunks)} text chunks from {original_filename}")

            # Get embeddings for each text chunk
            successful_embeddings = 0
            for i, chunk in enumerate(text_chunks):
                try:
                    # Skip empty chunks
                    if not chunk or not chunk.strip():
                        logger.warning(f"Skipping empty chunk {i}")
                        continue

                    # Generate embedding
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
                    successful_embeddings += 1
                except Exception as e:
                    logger.error(f"Error embedding chunk {i}: {str(e)}")

            # Check if we have any successful embeddings
            if successful_embeddings == 0:
                return jsonify({'error': 'Failed to create embeddings for the document'}), 500

            # Log successful embeddings
            logger.debug(f"Successfully created {successful_embeddings} embeddings for {original_filename}")

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
                'chunkCount': len(text_chunks),
                'embeddingCount': successful_embeddings
            })

    except Exception as e:
        # Detailed error logging
        logger.error(f"Error processing file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Process a chat message and generate a response."""
    try:
        # Check if we have valid JSON
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.json
        query = data.get('message', '').strip()

        # Validate query
        if not query:
            return jsonify({'error': 'No message provided'}), 400

        logger.debug(f"Processing chat query: {query[:50]}...")

        # Generate embedding for query with timeout handling
        try:
            query_embedding = get_embeddings(query, timeout=60)
            logger.debug("Successfully generated query embedding")
        except TimeoutError:
            logger.error("Timeout while generating embeddings")
            return jsonify({'error': 'Processing took too long. Please try with a smaller file or chunk size.'}), 504
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return jsonify({'error': 'Failed to process your question. Please try again.'}), 500

        # Search for relevant documents
        try:
            search_results = vector_store.search(query_embedding, k=5)
            logger.debug(f"Found {len(search_results)} relevant document chunks")
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return jsonify({'error': 'Error searching knowledge base. Please try again.'}), 500

        # Handle case with no relevant documents
        if not search_results:
            logger.debug("No relevant documents found for query")
            return jsonify({
                'answer': "I don't have enough information to answer that question. Try uploading relevant documents first."
            })

        # Generate response using RAG
        try:
            # Extract context texts
            context_texts = [result['metadata']['text'] for result in search_results]
            context = "\n\n".join(context_texts)

            # Generate response
            response = generate_response(query, context)
            logger.debug("Successfully generated response")
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Failed to generate a response. Please try again.'}), 500

        # Update chat history
        try:
            if 'chat_history' not in session:
                session['chat_history'] = []

            session['chat_history'].append({
                'query': query,
                'response': response
            })
            session.modified = True
        except Exception as e:
            # Non-critical error, just log
            logger.error(f"Error updating chat history: {str(e)}")

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
        # Catch-all for any other errors
        logger.error(f"Unexpected error in chat: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/documents', methods=['GET'])
def get_documents():
    """Get list of uploaded documents."""
    try:
        if 'documents' not in session:
            session['documents'] = []
        return jsonify(session['documents'])
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to retrieve documents. Please refresh the page.'}), 500

@app.route('/clear', methods=['POST'])
def clear_session():
    """Clear session data and vector store."""
    try:
        # Clear session first
        logger.debug("Clearing session data")
        session.clear()

        # Then clear vector store
        logger.debug("Clearing vector store")
        vector_store.clear()

        logger.debug("Session and vector store successfully cleared")
        return jsonify({'success': True, 'message': 'Session data and documents cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error clearing session: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)