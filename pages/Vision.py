"""
Vision page for the SilentCodingLegend AI agent.
This page allows users to upload images and get AI analysis with vision-capable models.
"""

import streamlit as st
from src.utils import apply_dark_theme

# Page configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="Vision Analysis",
    page_icon="👁️",
    layout="wide"
)

# Apply dark theme using utility function
apply_dark_theme()

import pandas as pd
import json
import os
import base64
from io import BytesIO
from datetime import datetime
from pathlib import Path

from src.agents.openrouter_agent import OpenRouterAgent
from src.model_config import MODELS_BY_PROVIDER

# Function to get models with vision capabilities
def get_vision_models():
    """Get models that support vision capabilities from the available models."""
    vision_models = []
    
    # Models known to support vision capabilities
    vision_capable_models = [
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-1.5-pro",
        "google/gemini-1.5-flash",
        "google/gemini-2.5-pro",
        "openai/gpt-4-vision",
        "openai/gpt-4o",
        "qwen/qwen2.5-vl-3b-instruct",
        "qwen/qwen-2.5-vl-7b-instruct",
        "qwen/qwen2.5-vl-32b-instruct",
        "qwen/qwen2.5-vl-72b-instruct"
    ]
    
    # Check all available models
    for provider, models in MODELS_BY_PROVIDER.items():
        for model_name, model_id in models.items():
            # Check if the model ID has a vision-capable prefix or contains VL (Vision-Language)
            if any(model_id.startswith(prefix) for prefix in vision_capable_models) or "vl" in model_id.lower():
                vision_models.append({
                    "provider": provider,
                    "name": model_name,
                    "id": model_id,
                    "display": f"{provider} - {model_name}"
                })
    
    return vision_models

# Function to encode image to base64
def encode_image(uploaded_file):
    """Encode the uploaded image to base64."""
    bytes_data = uploaded_file.getvalue()
    return base64.b64encode(bytes_data).decode('utf-8')

# Function to backup vision analysis history
def backup_vision_history(history_data):
    """Backup vision history to the Chat_History directory."""
    # Create the backup directory if it doesn't exist
    backup_dir = Path("/home/silentlegendkali/scl-openrouter/Chat_History")
    backup_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Backup as JSON (CSV might not be suitable for image data)
    json_path = backup_dir / f"vision_history_{timestamp}.json"
    
    # Create a version without the base64 image data for saving
    save_data = []
    for item in history_data:
        # Create a copy without the full image data to save space
        save_item = item.copy()
        if "image_data" in save_item:
            # Store just the first 100 chars of image data to identify it
            save_item["image_data"] = save_item["image_data"][:100] + "..."
        save_data.append(save_item)
    
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # Also add to the global chat history if it exists
    if "chat_history" in st.session_state:
        for item in history_data:
            item_copy = item.copy()
            # Don't store the full image data in the chat history
            if "image_data" in item_copy:
                item_copy["image_data"] = "[Image data]" 
            
            chat_entry = {
                "timestamp": item_copy.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "model": item_copy.get("model", "Unknown"),
                "prompt": f"[Image] {item_copy.get('prompt', '')}",
                "response": item_copy.get("response", ""),
                "source": "Vision Analysis"  # Mark the source
            }
            st.session_state.chat_history.append(chat_entry)
    
    return json_path

# Function for generating a response based on image and prompt
def analyze_image(image_file, prompt, model_id, temperature=0.7, max_tokens=1024):
    """Generate an analysis of the image using the selected model."""
    try:
        # Encode the image to base64
        base64_image = encode_image(image_file)
        
        # Create the agent with the selected model
        agent = OpenRouterAgent(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create system prompt for vision analysis
        system_prompt = """
        You are a helpful vision assistant that can analyze images.
        Provide detailed and accurate descriptions based on what you see in the image.
        If you're uncertain about anything in the image, acknowledge that uncertainty.
        Be specific in your descriptions and respond to the user's query directly.
        You can recognize and analyze objects, scenes, text (in multiple languages), 
        spatial relationships, actions, and emotions in the image.
        For any text in the image, provide accurate transcriptions.
        """
        
        # Generate response with the image
        response = agent.generate_vision_response(
            prompt=prompt,
            image_base64=base64_image,
            system_prompt=system_prompt
        )
        
        return response, base64_image
    except Exception as e:
        return f"Error analyzing image: {str(e)}", None

# Initialize session state variables
if "vision_history" not in st.session_state:
    st.session_state.vision_history = []

if "selected_vision_model" not in st.session_state:
    # Try to find a good default model in this order of preference
    preferred_models = [
        "anthropic/claude-3.5-sonnet:thinking",
        "qwen/qwen2.5-vl-72b-instruct:free",
        "qwen/qwen2.5-vl-32b-instruct:free",
        "anthropic/claude-3-sonnet:free"
    ]
    
    for model in preferred_models:
        found = False
        for provider, models in MODELS_BY_PROVIDER.items():
            for model_name, model_id in models.items():
                if model_id == model:
                    st.session_state.selected_vision_model = model
                    found = True
                    break
            if found:
                break
        if found:
            break
    
    # If no preferred models are found, default to Claude 3.5 Sonnet
    if "selected_vision_model" not in st.session_state:
        st.session_state.selected_vision_model = "anthropic/claude-3.5-sonnet:thinking"

# Auto-backup if history has changed
if "last_vision_history_length" not in st.session_state:
    st.session_state.last_vision_history_length = 0

current_length = len(st.session_state.vision_history)
if current_length > 0 and current_length != st.session_state.last_vision_history_length:
    # Perform backup
    json_path = backup_vision_history(st.session_state.vision_history)
    st.session_state.last_vision_history_length = current_length

# Main page UI
st.title("👁️ Vision Analysis")
st.subheader("Upload images and get AI-powered visual analysis")

# Qwen VL Models Information
with st.expander("ℹ️ About Qwen VL Models"):
    st.markdown("""
    ### Qwen Vision-Language Models
    
    The Vision page now supports Qwen VL (Vision-Language) models, which offer powerful multimodal capabilities:
    
    - **Qwen 2.5 VL Series**: Advanced vision-language models from 3B to 72B parameters
    - **Multilingual Support**: Excellent at handling both Chinese and English content
    - **Fine-grained Recognition**: Better at identifying detailed objects and their properties
    - **Text Recognition**: Strong ability to read and understand text within images
    - **Spatial Understanding**: Good at understanding the layout and relationship between objects
    
    **When to use Qwen VL models:**
    - When analyzing images containing text in multiple languages
    - For detailed object recognition and relationship analysis
    - When working with complex visual scenes
    - For specialized image analysis tasks requiring fine-grained understanding
    
    The larger models (32B, 72B) generally provide better analysis quality at the cost of slightly longer processing time.
    """)

# Sidebar for model selection and settings
with st.sidebar:
    st.title("Vision Settings")
    
    # Get available vision models
    vision_models = get_vision_models()
    
    if not vision_models:
        st.warning("No vision-capable models found in your configuration.")
        st.info("Using default model: Claude 3.5 Sonnet")
        vision_models = [{
            "provider": "Anthropic",
            "name": "claude-3.5-sonnet",
            "id": "anthropic/claude-3.5-sonnet:thinking",
            "display": "Anthropic - Claude 3.5 Sonnet"
        }]
    
    # Model selection
    model_options = [model["display"] for model in vision_models]
    model_ids = [model["id"] for model in vision_models]
    
    # Find the current model index
    selected_index = 0
    for i, model_id in enumerate(model_ids):
        if model_id == st.session_state.selected_vision_model:
            selected_index = i
            break
    
    selected_model = st.selectbox(
        "Select Vision Model",
        options=model_options,
        index=min(selected_index, len(model_options)-1),
        key="vision_model_selection"
    )
    
    # Update the model ID based on selection
    st.session_state.selected_vision_model = model_ids[model_options.index(selected_model)]
    
    # Temperature setting
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Controls randomness: Lower values are more focused and deterministic, higher values are more creative."
    )
    
    # Max tokens
    max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=4096,
        value=1024,
        step=256,
        help="Maximum length of the AI response."
    )
    
    # Vision prompt templates
    st.subheader("Prompt Templates")
    
    vision_templates = {
        "Describe Image": "Please describe what you see in this image in detail.",
        "Identify Objects": "Identify and list all the main objects or elements visible in this image.",
        "Analyze Scene": "Analyze this scene and tell me what's happening.",
        "Identify Text": "Can you read and transcribe any text visible in this image?",
        "Technical Analysis": "Provide a technical analysis of this image (e.g., composition, lighting, etc.)",
        "Creative Interpretation": "Give me a creative interpretation or story inspired by this image.",
        "Object Relations": "Describe the spatial relationships between the objects in this image.",
        "Cultural Context": "Analyze any cultural elements or references in this image.",
        "Multilingual Text": "Identify and translate any text in this image, including non-English languages."
    }
    
    selected_template = st.selectbox(
        "Choose a prompt template",
        options=list(vision_templates.keys())
    )
    
    if "prompt_input" not in st.session_state:
        st.session_state.prompt_input = ""
    
    if st.button("Apply Template"):
        st.session_state.prompt_input = vision_templates[selected_template]
        st.rerun()

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    # Image upload section
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg', 'webp'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Prompt input
        prompt = st.text_area(
            "What would you like to know about this image?", 
            value=st.session_state.prompt_input,
            height=100
        )
        
        # Analysis button
        if st.button("Analyze Image"):
            if prompt:
                with st.spinner("Analyzing image..."):
                    # Call the analyze_image function
                    response, base64_image = analyze_image(
                        uploaded_file, 
                        prompt, 
                        st.session_state.selected_vision_model,
                        temperature,
                        max_tokens
                    )
                    
                    # Display the analysis result
                    st.markdown("### Analysis Result:")
                    st.markdown(response)
                    
                    # Save to history
                    st.session_state.vision_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": st.session_state.selected_vision_model,
                        "prompt": prompt,
                        "response": response,
                        "image_name": uploaded_file.name,
                        "image_data": base64_image
                    })
                    
                    # Backup current analysis
                    json_path = backup_vision_history([st.session_state.vision_history[-1]])
                    
                    # Update the history length
                    st.session_state.last_vision_history_length = len(st.session_state.vision_history)
            else:
                st.warning("Please enter a prompt before analyzing.")

with col2:
    # Vision history
    st.subheader("Analysis History")
    
    if not st.session_state.vision_history:
        st.info("No analysis history yet. Upload an image and analyze it to get started.")
    else:
        # Show most recent analyses first
        for i, entry in enumerate(reversed(st.session_state.vision_history)):
            with st.expander(f"Analysis {len(st.session_state.vision_history)-i} - {entry['image_name']} - {entry['timestamp']}"):
                # Display the image
                if 'image_data' in entry:
                    try:
                        st.image(
                            BytesIO(base64.b64decode(entry['image_data'])),
                            caption=f"Analyzed Image: {entry['image_name']}",
                            use_container_width=True
                        )
                    except:
                        st.warning("Unable to display saved image.")
                
                # Display prompt and response
                st.markdown("**Prompt:**")
                st.write(entry['prompt'])
                st.markdown("**Response:**")
                st.write(entry['response'])
                st.write(f"Model: {entry['model']}")
    
    # Clear history button
    if st.session_state.vision_history and st.button("Clear Analysis History"):
        st.session_state.vision_history = []
        st.rerun()
    
    # Link to Chat History
    if st.session_state.vision_history:
        st.markdown("---")
        st.info("All vision analyses are also saved to the main Chat History. [View Chat History](/Chat_History)")

# Additional information
with st.expander("ℹ️ About Vision Analysis"):
    st.markdown("""
    ### How Vision Analysis Works
    
    1. **Upload an image** - Supported formats include PNG, JPG, JPEG, and WEBP
    2. **Enter a prompt** - Be specific about what you want to know about the image
    3. **Select a model** - Different models have different visual capabilities
    4. **Get analysis** - The AI will analyze the image and respond to your prompt
    
    ### Tips for Better Results
    
    - **Be specific** in your prompts
    - **Ask about particular aspects** of the image rather than general questions
    - **Try different models** for different types of analysis
    - **Lower the temperature** (0.1-0.3) for more factual descriptions
    - **Increase the temperature** (0.7-0.9) for more creative interpretations
    
    ### Model Capabilities
    
    **Anthropic Claude Models**
    - Strong at detailed image descriptions
    - Great for understanding complex scenes with multiple elements
    - Excellent at identifying text in images
    
    **Google Gemini Models**
    - Strong visual reasoning capabilities
    - Good at detailed visual analysis
    - Excellent for technical image assessment
    
    **Qwen VL Models**
    - Specialized vision-language capabilities
    - Excellent at handling Chinese and English content
    - Strong performance on detailed object recognition
    - Good at understanding image context and relationships
    - Available in various sizes (3B to 72B parameters)
    
    ### Privacy Note
    
    Images are encoded and sent to the AI model for analysis but are not permanently stored
    on external servers. Your image history is saved locally in your session.
    """)