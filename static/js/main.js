document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatContainer = document.getElementById('chat-container');
    const fileUploadArea = document.getElementById('file-upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadButton = document.getElementById('upload-button');
    const uploadingMessage = document.getElementById('uploading-message');
    const fileList = document.getElementById('file-list');
    const clearButton = document.getElementById('clear-button');
    
    // Add system message to start
    addBotMessage("Hello! I'm your document assistant. Please upload documents, images, or videos, and I'll answer questions about them.");
    
    // Load existing documents
    fetchDocuments();
    
    // File upload area event listeners
    fileUploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        fileUploadArea.classList.add('dropzone-active');
    });
    
    fileUploadArea.addEventListener('dragleave', function() {
        fileUploadArea.classList.remove('dropzone-active');
    });
    
    fileUploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        fileUploadArea.classList.remove('dropzone-active');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileUpload(files[0]);
        }
    });
    
    fileUploadArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });
    
    uploadButton.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Chat form event listener
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (message) {
            sendMessage(message);
            messageInput.value = '';
        }
    });
    
    // Clear button event listener
    clearButton.addEventListener('click', function() {
        clearSession();
    });
    
    // Function to add user message to chat
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';
        messageElement.textContent = message;
        chatContainer.appendChild(messageElement);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Function to add bot message to chat
    function addBotMessage(message, sources = []) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message bot-message';
        
        // Create message text
        const messageText = document.createElement('div');
        messageText.textContent = message;
        messageElement.appendChild(messageText);
        
        // Add sources if provided
        if (sources && sources.length > 0) {
            const sourceList = document.createElement('div');
            sourceList.className = 'source-list mt-2';
            sourceList.innerHTML = '<strong>Sources:</strong> ' + sources.join(', ');
            messageElement.appendChild(sourceList);
        }
        
        chatContainer.appendChild(messageElement);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Function to add thinking message
    function addThinkingMessage() {
        const messageElement = document.createElement('div');
        messageElement.className = 'message bot-message thinking-message';
        messageElement.innerHTML = '<div class="d-flex align-items-center">' +
                                  '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>' +
                                  'Thinking...</div>';
        chatContainer.appendChild(messageElement);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return messageElement;
    }
    
    // Function to handle file upload
    function handleFileUpload(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        uploadingMessage.classList.remove('d-none');
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            // Check if response is valid before parsing JSON
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(`Server returned ${response.status}: ${text.substring(0, 100)}...`);
                });
            }
            return response.json();
        })
        .then(data => {
            uploadingMessage.classList.add('d-none');
            if (data.error) {
                addBotMessage(`Error: ${data.error}`);
            } else {
                addBotMessage(`Successfully processed ${data.filename}. I can now answer questions about this file.`);
                fetchDocuments();
            }
        })
        .catch(error => {
            uploadingMessage.classList.add('d-none');
            addBotMessage(`Error uploading file: ${error.message}`);
            console.error('Error:', error);
        });
    }
    
    // Function to send message to server
    function sendMessage(message) {
        addUserMessage(message);
        const thinkingMessage = addThinkingMessage();
        
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => {
            // Check if response is valid before parsing JSON
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(`Server returned ${response.status}: ${text.substring(0, 100)}...`);
                });
            }
            return response.json();
        })
        .then(data => {
            // Remove thinking message
            thinkingMessage.remove();
            
            if (data.error) {
                addBotMessage(`Error: ${data.error}`);
            } else {
                addBotMessage(data.answer, data.sources);
            }
        })
        .catch(error => {
            thinkingMessage.remove();
            addBotMessage(`Error: ${error.message}`);
            console.error('Error:', error);
        });
    }
    
    // Function to fetch documents
    function fetchDocuments() {
        fetch('/documents')
        .then(response => {
            // Check if response is valid before parsing JSON
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(`Server returned ${response.status}: ${text.substring(0, 100)}...`);
                });
            }
            return response.json();
        })
        .then(documents => {
            fileList.innerHTML = '';
            
            if (documents.length === 0) {
                const emptyItem = document.createElement('div');
                emptyItem.className = 'text-muted';
                emptyItem.textContent = 'No documents uploaded';
                fileList.appendChild(emptyItem);
            } else {
                documents.forEach(doc => {
                    const docItem = document.createElement('div');
                    docItem.className = 'file-item';
                    
                    // Icon based on file type
                    let icon = 'file-text';
                    if (doc.type === 'image') {
                        icon = 'image';
                    } else if (doc.type === 'video') {
                        icon = 'film';
                    }
                    
                    docItem.innerHTML = `
                        <div>
                            <i data-feather="${icon}" class="me-2"></i>
                            <span>${doc.name}</span>
                        </div>
                    `;
                    fileList.appendChild(docItem);
                });
                
                // Initialize Feather icons
                if (window.feather) {
                    feather.replace();
                }
            }
        })
        .catch(error => {
            console.error('Error fetching documents:', error);
            // Show error in file list area
            fileList.innerHTML = `<div class="alert alert-danger">Error loading documents: ${error.message}</div>`;
        });
    }
    
    // Function to clear session
    function clearSession() {
        fetch('/clear', {
            method: 'POST'
        })
        .then(response => {
            // Check if response is valid before parsing JSON
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(`Server returned ${response.status}: ${text.substring(0, 100)}...`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                chatContainer.innerHTML = '';
                addBotMessage("Session cleared. I've forgotten all previous documents. You can upload new ones.");
                fetchDocuments();
            } else if (data.error) {
                addBotMessage(`Error: ${data.error}`);
            }
        })
        .catch(error => {
            addBotMessage(`Error clearing session: ${error.message}`);
            console.error('Error:', error);
        });
    }
});
