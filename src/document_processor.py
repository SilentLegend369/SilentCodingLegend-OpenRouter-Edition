"""
Document processing utilities for the SilentCodingLegend AI agent.
"""

import os
import tempfile
from pathlib import Path
import mimetypes
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Supported file types and their MIME types
SUPPORTED_TEXT_TYPES = {
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.py': 'text/x-python',
    '.js': 'text/javascript',
    '.html': 'text/html',
    '.css': 'text/css',
    '.json': 'application/json',
    '.csv': 'text/csv',
}

def is_supported_file_type(file_path):
    """
    Check if the file type is supported for processing.
    
    Args:
        file_path (str or Path): Path to the file to check
        
    Returns:
        bool: True if the file type is supported, False otherwise
    """
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()
    return file_extension in SUPPORTED_TEXT_TYPES

def process_text_file(file, max_length=None):
    """
    Process a text file and return its content.
    
    Args:
        file: File object from st.file_uploader
        max_length (int, optional): Maximum number of characters to read
        
    Returns:
        str: Content of the file
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write uploaded file content to temp file
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name
        
        # Read content from the temp file
        with open(temp_file_path, 'r', encoding='utf-8', errors='replace') as f:
            if max_length:
                content = f.read(max_length)
                if len(content) >= max_length:
                    content += "\n... [Content truncated due to length]"
            else:
                content = f.read()
        
        # Clean up the temp file
        os.unlink(temp_file_path)
        
        return content
    except Exception as e:
        logger.error(f"Error processing text file: {e}")
        return f"Error processing file: {str(e)}"

def get_file_format_prompt(file_name, content):
    """
    Create a prompt for the AI based on the file format.
    
    Args:
        file_name (str): Name of the file
        content (str): Content of the file
        
    Returns:
        str: Formatted prompt for the AI
    """
    file_extension = Path(file_name).suffix.lower()
    
    # Format the prompt based on file type
    if file_extension == '.py':
        return f"I've uploaded a Python file named '{file_name}'. Please analyze this code:\n\n```python\n{content}\n```"
    elif file_extension == '.js':
        return f"I've uploaded a JavaScript file named '{file_name}'. Please analyze this code:\n\n```javascript\n{content}\n```"
    elif file_extension == '.json':
        return f"I've uploaded a JSON file named '{file_name}'. Please analyze this JSON data:\n\n```json\n{content}\n```"
    elif file_extension == '.md':
        return f"I've uploaded a Markdown file named '{file_name}'. Please analyze this document:\n\n```markdown\n{content}\n```"
    elif file_extension == '.html':
        return f"I've uploaded an HTML file named '{file_name}'. Please analyze this HTML code:\n\n```html\n{content}\n```"
    elif file_extension == '.css':
        return f"I've uploaded a CSS file named '{file_name}'. Please analyze this CSS code:\n\n```css\n{content}\n```"
    elif file_extension == '.csv':
        return f"I've uploaded a CSV file named '{file_name}'. Here's the content (first part):\n\n```\n{content}\n```\nPlease analyze this data."
    else:  # Default for .txt and other formats
        return f"I've uploaded a text file named '{file_name}'. Please analyze this content:\n\n{content}"
