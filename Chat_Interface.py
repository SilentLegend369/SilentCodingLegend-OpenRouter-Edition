"""
Main Chat Interface for the SilentCodingLegend AI agent.
This is a Streamlit application that provides a chat interface to interact with
various AI models through the OpenRouter API.
"""

import streamlit as st
import os
import json
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# Import custom modules
from src.agents.openrouter_agent import OpenRouterAgent
from src.model_config import (
    MODELS_BY_PROVIDER,
    MODELS,
    PREFERRED_PROVIDERS,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_MODEL_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    CACHE_FILE,
    refresh_model_cache,
    display_available_models
)
from src.utils import (
    get_best_model_for_task,
    list_models_for_task,
    save_user_model_preference,
    get_user_model_preference
)
from src.usage_tracker import UsageTracker

# Load environment variables
load_dotenv()

# Page configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="SilentCodingLegend AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS is loaded from external file

# Apply dark theme using utility function
from src.utils import apply_dark_theme
apply_dark_theme()

# Verify API key is set
if not os.getenv("OPENROUTER_API_KEY"):
    st.error("OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable.")
    st.stop()

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    
if "current_provider" not in st.session_state:
    st.session_state.current_provider = DEFAULT_PROVIDER
    
if "current_model" not in st.session_state:
    st.session_state.current_model = DEFAULT_MODEL

if "current_task_type" not in st.session_state:
    st.session_state.current_task_type = "general"
    
if "conversation_titles" not in st.session_state:
    st.session_state.conversation_titles = {}

if "agent" not in st.session_state:
    # Check if there's a user preference for the general task
    preferred_model_id = get_user_model_preference("general")
    if preferred_model_id:
        st.session_state.agent = OpenRouterAgent(
            model_id=preferred_model_id,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            task_type="general"
        )
    else:
        # Use the default model
        st.session_state.agent = OpenRouterAgent(
            model_id=DEFAULT_MODEL_ID,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            task_type="general"
        )

# Function to refresh model list
def refresh_models():
    with st.spinner("Refreshing models from OpenRouter API..."):
        updated_models = refresh_model_cache()
        if updated_models:
            st.success("Successfully refreshed model list from OpenRouter API.")
            # Update the model selection dropdowns
            st.rerun()
        else:
            st.error("Failed to refresh models. Check your API key and internet connection.")

# Function to toggle model caching
def toggle_model_cache():
    current_state = os.getenv("DISABLE_MODEL_CACHE", "false").lower() == "true"
    new_state = not current_state
    
    # Update the environment variable
    if new_state:  # If we're disabling caching
        os.environ["DISABLE_MODEL_CACHE"] = "true"
        # Delete the cache file if it exists
        if CACHE_FILE.exists():
            try:
                os.remove(CACHE_FILE)
                st.success("Model caching disabled and cache file deleted.")
            except:
                st.warning("Model caching disabled but could not delete existing cache file.")
        else:
            st.success("Model caching disabled.")
    else:  # If we're enabling caching
        os.environ["DISABLE_MODEL_CACHE"] = "false"
        st.success("Model caching enabled.")
        
    # Update the session state
    st.session_state.disable_model_cache = new_state

# Function to update provider
def update_provider():
    provider = st.session_state.provider_select
    
    # When provider changes, set the model to the first available model for that provider
    if provider in MODELS_BY_PROVIDER:
        models = list(MODELS_BY_PROVIDER[provider].keys())
        if models:
            # Update the model selection to the first model in the list
            st.session_state.model_select = models[0]
            # Update current provider in session state
            st.session_state.current_provider = provider
            # Update current model in session state
            st.session_state.current_model = models[0]
            # Get the model ID
            model_id = MODELS_BY_PROVIDER[provider][models[0]]
            # Update the agent
            st.session_state.agent = OpenRouterAgent(
                model_id=model_id,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                task_type=st.session_state.current_task_type
            )
            # Save this as user preference for current task type
            save_user_model_preference(st.session_state.current_task_type, model_id)
            # Display success message
            st.sidebar.success(f"Provider changed to {provider}. Model set to {models[0]}.")
    else:
        st.sidebar.error(f"Provider {provider} not found. Reverting to default.")
        st.session_state.provider_select = DEFAULT_PROVIDER
        # The next rerun will handle setting the model

# Function to update model (called when model selection changes)
def update_model():
    provider = st.session_state.provider_select
    model = st.session_state.model_select
    
    try:
        model_id = MODELS_BY_PROVIDER[provider][model]
        
        st.session_state.current_model = model
        
        st.session_state.agent = OpenRouterAgent(
            model_id=model_id,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
            task_type=st.session_state.current_task_type
        )
        
        # Save this as user preference for current task type
        save_user_model_preference(st.session_state.current_task_type, model_id)
        
        st.sidebar.success(f"Model updated to {provider}/{model}")
    except KeyError:
        st.sidebar.error(f"Model '{model}' not found for provider '{provider}'. Please select a different model.")
        
        if provider in MODELS_BY_PROVIDER:
            models = list(MODELS_BY_PROVIDER[provider].keys())
            if models:
                st.session_state.current_model = models[0]
                st.session_state.model_select = models[0]

# Function to update task type
def update_task_type():
    task_type = st.session_state.task_type_select
    st.session_state.current_task_type = task_type
    
    # Check if there's a user preference for this task type
    preferred_model_id = get_user_model_preference(task_type)
    
    if preferred_model_id:
        # Use the preferred model
        st.session_state.agent = OpenRouterAgent(
            model_id=preferred_model_id,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
            task_type=task_type
        )
        # Update UI to reflect this model
        for provider, models in MODELS_BY_PROVIDER.items():
            for model_name, model_id in models.items():
                if model_id == preferred_model_id:
                    st.session_state.provider_select = provider
                    st.session_state.model_select = model_name
                    st.session_state.current_provider = provider
                    st.session_state.current_model = model_name
                    break
        
        st.sidebar.success(f"Changed to {task_type} mode with preferred model: {preferred_model_id}")
    else:
        # Get best model for this task
        model_name, model_id = get_best_model_for_task(task_type)
        
        if model_id:
            st.session_state.agent = OpenRouterAgent(
                model_id=model_id,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                task_type=task_type
            )
            
            # Update UI to reflect this model
            for provider, models in MODELS_BY_PROVIDER.items():
                if model_name in models:
                    st.session_state.provider_select = provider
                    st.session_state.model_select = model_name
                    st.session_state.current_provider = provider
                    st.session_state.current_model = model_name
                    break
            
            st.sidebar.success(f"Changed to {task_type} mode with recommended model: {model_name}")
        else:
            st.sidebar.warning(f"No recommended model found for {task_type}. Using current model.")

# Function to generate response
def generate_response(user_input):
    if not user_input:
        st.warning("Please enter a prompt.")
        return
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Display thinking message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.info("Thinking...")
        
        # Get response from agent
        provider = st.session_state.current_provider
        model = st.session_state.current_model
        
        try:
            model_id = MODELS_BY_PROVIDER[provider][model]
            
            # Update agent model if needed
            if st.session_state.agent.model_id != model_id:
                st.session_state.agent.model_id = model_id
            
            system_prompt = st.session_state.system_prompt if st.session_state.system_prompt else None
        except KeyError:
            message_placeholder.error(f"Error: Model {provider}/{model} not found. Please select a different model.")
            return
        
        # Generate response with streaming effect
        response_text = ""
        
        try:
            # Get the full response first
            full_response = st.session_state.agent.generate_response(
                user_input,
                system_prompt=system_prompt
            )
            
            # Simulate streaming by displaying chunks of text
            for i in range(0, len(full_response), 3):
                response_text = full_response[:i+3]
                message_placeholder.markdown(response_text + "▌")
                time.sleep(0.001)  # Slight delay for visual effect
            
            # Display final response
            message_placeholder.markdown(full_response)
            response = full_response
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            message_placeholder.error(error_msg)
            response = error_msg
    
    # Add to chat history
    st.session_state.chat_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": f"{st.session_state.current_provider}/{st.session_state.current_model}",
        "model_id": st.session_state.agent.model_id,
        "task_type": st.session_state.current_task_type,
        "prompt": user_input,
        "response": response
    })

def save_chat_history():
    """
    Save the chat history to CSV and JSON files.
    Files are saved in a Chat_History folder with timestamp in the filename.
    """
    if not st.session_state.chat_history:
        st.warning("No chat history to save.")
        return
    
    # Create Chat_History directory if it doesn't exist
    chat_history_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "Chat_History"
    chat_history_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define filenames
    csv_filename = chat_history_dir / f"chat_history_{timestamp}.csv"
    json_filename = chat_history_dir / f"chat_history_{timestamp}.json"
    
    try:
        # Save as CSV
        df = pd.DataFrame(st.session_state.chat_history)
        df.to_csv(csv_filename, index=False)
        
        # Save as JSON
        with open(json_filename, 'w') as f:
            json.dump(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "chats": st.session_state.chat_history
                }, 
                f, 
                indent=2
            )
        
        st.success(f"Chat history saved to {csv_filename} and {json_filename}")
        return True
    except Exception as e:
        st.error(f"Error saving chat history: {e}")
        return False

def new_conversation():
    """
    Start a new conversation.
    """
    # Save current conversation if it exists
    if st.session_state.chat_history:
        st.session_state.conversations[st.session_state.conversation_id] = st.session_state.chat_history.copy()
        
    # Generate new conversation ID
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Clear chat history
    st.session_state.chat_history = []
    
    # Track new conversation in usage statistics
    try:
        tracker = UsageTracker()
        tracker.track_conversation()
    except Exception as e:
        st.warning(f"Failed to track conversation: {e}", icon="⚠️")
    
    # Use a session state flag to trigger rerun in the main loop instead
    st.session_state.new_conversation_created = True
    st.success("Started new conversation")

def save_conversation():
    """
    Save the current conversation.
    """
    if not st.session_state.chat_history:
        st.warning("No conversation to save.")
        return
    
    # Save current conversation
    st.session_state.conversations[st.session_state.conversation_id] = st.session_state.chat_history.copy()
    
    # Generate title if it doesn't exist
    if st.session_state.conversation_id not in st.session_state.conversation_titles:
        # Use first user message as title, truncated to 30 chars
        first_msg = st.session_state.chat_history[0]["prompt"]
        title = first_msg[:30] + ("..." if len(first_msg) > 30 else "")
        st.session_state.conversation_titles[st.session_state.conversation_id] = title
    
    st.success("Conversation saved")

def load_conversation(conv_id):
    """
    Load a previous conversation.
    """
    if conv_id not in st.session_state.conversations:
        st.error("Conversation not found.")
        return
    
    # Save current conversation if it exists
    if st.session_state.chat_history and st.session_state.conversation_id not in st.session_state.conversations:
        st.session_state.conversations[st.session_state.conversation_id] = st.session_state.chat_history.copy()
    
    # Set conversation ID and load chat history
    st.session_state.conversation_id = conv_id
    st.session_state.chat_history = st.session_state.conversations[conv_id].copy()
    
    st.success(f"Loaded conversation: {st.session_state.conversation_titles.get(conv_id, conv_id)}")
    # Use a session state flag to trigger rerun in the main loop
    st.session_state.loaded_conversation = True

def delete_conversation(conv_id):
    """
    Delete a conversation.
    """
    if conv_id not in st.session_state.conversations:
        st.error("Conversation not found.")
        return
    
    # Delete conversation
    del st.session_state.conversations[conv_id]
    
    # Delete title if it exists
    if conv_id in st.session_state.conversation_titles:
        del st.session_state.conversation_titles[conv_id]
    
    # If we deleted the current conversation, start a new one
    if conv_id == st.session_state.conversation_id:
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state.chat_history = []
    
    st.success("Conversation deleted")
    # Use a session state flag to trigger rerun in the main loop
    st.session_state.deleted_conversation = True

# Main app
def main():
    
    # Check if we need to rerun after conversation changes
    if st.session_state.get("new_conversation_created", False):
        st.session_state.new_conversation_created = False
        st.rerun()
        
    if st.session_state.get("loaded_conversation", False):
        st.session_state.loaded_conversation = False
        st.rerun()
        
    if st.session_state.get("deleted_conversation", False):
        st.session_state.deleted_conversation = False
        st.rerun()
    
    # Sidebar tabs for different functions
    with st.sidebar:
        st.title("SilentCodingLegend AI")
        st.caption("Powered by OpenRouter")
        
        # Create tabs for organization
        tab1, tab2, tab3 = st.tabs(["Models", "Conversations", "Stats"])
        
        with tab1:
            st.subheader("Model Selection")
        
        # Provider selection
        providers = list(MODELS_BY_PROVIDER.keys())
        st.selectbox(
            "Select Provider",
            providers,
            index=providers.index(st.session_state.current_provider) if st.session_state.current_provider in providers else 0,
            key="provider_select",
            on_change=update_provider
        )
        
        # Model selection (depends on provider)
        try:
            provider = st.session_state.provider_select
            models = list(MODELS_BY_PROVIDER[provider].keys())
            model_index = models.index(st.session_state.current_model) if st.session_state.current_model in models else 0
        except (KeyError, ValueError):
            # Handle case where provider or model isn't found
            provider = DEFAULT_PROVIDER
            models = list(MODELS_BY_PROVIDER[provider].keys())
            model_index = 0
            
        st.selectbox(
            "Select Model",
            models,
            index=model_index,
            key="model_select",
            on_change=update_model
        )
        
        # Task type selection
        st.subheader("Task Type")
        task_types = list(PREFERRED_PROVIDERS.keys())  # This uses our config: general, coding, vision, reasoning
        st.selectbox(
            "Select Task Type",
            task_types,
            index=task_types.index(st.session_state.current_task_type) if st.session_state.current_task_type in task_types else 0,
            key="task_type_select",
            on_change=update_task_type,
            help="Select the type of task to optimize model selection."
        )
        
        # Model refresh button
        st.button("Refresh Models", on_click=refresh_models, help="Fetch the latest models from OpenRouter API")
        
        # Current model info box
        model_id = MODELS_BY_PROVIDER.get(st.session_state.current_provider, {}).get(st.session_state.current_model, "Unknown")
        is_free = ":free" in model_id
        st.info(
            f"**Current Model:**  \n{st.session_state.current_provider}/{st.session_state.current_model}  \n"
            f"**Task Type:** {st.session_state.current_task_type}  \n"
            f"**Free Model:** {'✓' if is_free else '✗'}"
        )
        
        # Model parameters
        st.subheader("Model Parameters")
        
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.1,
            key="temperature",
            on_change=update_model,
            help="Controls randomness. Lower values are more deterministic, higher values more creative."
        )
        
        st.slider(
            "Max Tokens",
            min_value=128,
            max_value=4096,
            value=DEFAULT_MAX_TOKENS,
            step=128,
            key="max_tokens",
            on_change=update_model,
            help="Maximum number of tokens to generate."
        )
        
        # System prompt
        st.text_area(
            "System Prompt (Optional)",
            height=150,
            key="system_prompt",
            help="Sets the behavior of the AI assistant."
        )
        
        # Add helpful information about preferred models
        st.subheader("Preferred Models")
        st.caption("Current task type: " + st.session_state.current_task_type)
        
        # Show preferred providers for current task
        preferred_providers = PREFERRED_PROVIDERS.get(st.session_state.current_task_type, [])
        if preferred_providers:
            st.caption("Preferred providers (in order):")
            for provider in preferred_providers:
                if provider in MODELS_BY_PROVIDER:
                    model_count = len(MODELS_BY_PROVIDER[provider])
                    st.caption(f"- {provider} ({model_count} models)")
        
        # Add a divider
        st.divider()
        
        # Advanced options
        st.subheader("Advanced Options")
        
        # Initialize the disable_model_cache state if not present
        if "disable_model_cache" not in st.session_state:
            st.session_state.disable_model_cache = os.getenv("DISABLE_MODEL_CACHE", "false").lower() == "true"
        
        # Model caching toggle
        cache_status = "Disabled" if st.session_state.disable_model_cache else "Enabled"
        st.button(f"Model Cache: {cache_status}", on_click=toggle_model_cache)
        
        st.caption("Disabling the model cache can help if you're experiencing issues with model fetching.")
        
        # Add a divider
        st.divider()
        
        # Save chat history button
        if st.session_state.chat_history:
            st.button("Save Chat History", on_click=save_chat_history)
        
        # Add links and information
        st.markdown("[View Chat History](/Chat_History)")
        st.markdown("[About OpenRouter](https://openrouter.ai)")
        st.caption("Created by SilentCodingLegend")
        
        # Conversations tab content
        with tab2:
            st.subheader("Conversations")
            
            # New conversation button
            st.button("New Conversation", on_click=new_conversation)
            
            # Save button for current conversation
            if st.session_state.chat_history:
                st.button("Save Current Conversation", on_click=save_conversation)
            
            # Saved conversations list
            if st.session_state.conversations:
                st.subheader("Saved Conversations")
                for conv_id, chat_history in st.session_state.conversations.items():
                    # Get title or use truncated first message
                    title = st.session_state.conversation_titles.get(conv_id, conv_id)
                    
                    # Create a container for this conversation
                    with st.container():
                        # Highlight current conversation
                        is_current = conv_id == st.session_state.conversation_id
                        title_prefix = "🟢 " if is_current else "📄 "
                        
                        # Show date from the ID (format: YYYYMMDDHHMMSS)
                        try:
                            date_obj = datetime.strptime(conv_id, "%Y%m%d%H%M%S")
                            date_str = date_obj.strftime("%b %d, %H:%M")
                        except:
                            date_str = "Unknown date"
                        
                        col1, col2, col3 = st.columns([5, 2, 1])
                        with col1:
                            if st.button(f"{title_prefix}{title}", key=f"load_{conv_id}"):
                                load_conversation(conv_id)
                        with col2:
                            st.caption(date_str)
                        with col3:
                            if st.button("🗑️", key=f"delete_{conv_id}"):
                                delete_conversation(conv_id)
        
        # Stats tab content
        with tab3:
            st.subheader("Usage Statistics")
            
            # Load current usage data
            try:
                tracker = UsageTracker()
                current_usage = tracker.get_monthly_summary()
                
                # Display summary
                st.caption(f"**Monthly Summary ({current_usage.get('month', 'Current')})**")
                st.caption(f"Total tokens: {current_usage.get('total_tokens', 0):,}")
                st.caption(f"Requests: {current_usage.get('requests', 0):,}")
                st.caption(f"Conversations: {current_usage.get('conversations', 0):,}")
                
                # Calculate estimated cost based on average rates
                prompt_tokens = current_usage.get("prompt_tokens", 0)
                completion_tokens = current_usage.get("completion_tokens", 0)
                est_cost = (prompt_tokens / 1000) * 0.002 + (completion_tokens / 1000) * 0.002
                
                st.caption(f"Estimated cost: ${est_cost:.4f}")
                
                # Link to detailed dashboard
                st.markdown("[View Detailed Dashboard](/Usage_Stats)")
                
            except Exception as e:
                st.warning(f"Error loading usage statistics: {e}", icon="⚠️")
                st.markdown("[View Detailed Dashboard](/Usage_Stats)")
    
    # Main interface
    st.title("SilentCodingLegend AI Agent")
    st.caption("Powered by OpenRouter")
    
    # Show current task and model in a prominent way
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Current Task Type:")
        st.subheader(f"📋 {st.session_state.current_task_type.capitalize()}")
    with col2:
        st.caption("Current Model:")
        model_name = f"{st.session_state.current_provider}/{st.session_state.current_model}"
        model_id = st.session_state.agent.model_id
        is_free = ":free" in model_id
        st.subheader(f"🤖 {model_name} {'(Free)' if is_free else ''}")
    
    # Welcome message if no chat history
    if not st.session_state.chat_history:
        st.info("👋 Welcome to the SilentCodingLegend AI Agent! Choose a model from the sidebar and start chatting.")
    
    # Display chat history in a container
    with st.container():
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["prompt"])
            with st.chat_message("assistant"):
                st.write(chat["response"])
        
        # Add spacer to ensure content isn't hidden behind the fixed chat input
        st.markdown('<div class="chat-spacer"></div>', unsafe_allow_html=True)
    
    # File uploader with custom styling
    uploaded_file = st.file_uploader("Upload a file (optional)", 
                                     type=["txt", "md", "py", "js", "html", "css", "json", "csv"])
    
    if uploaded_file:
        from src.document_processor import process_text_file, get_file_format_prompt
        
        # Process the uploaded file
        st.info(f"Processing file: {uploaded_file.name}")
        
        # Get file content
        file_content = process_text_file(uploaded_file, max_length=10000)
        
        # Create prompt from file content
        file_prompt = get_file_format_prompt(uploaded_file.name, file_content)
        
        # Display a button to use this content in chat
        if st.button(f"Analyze {uploaded_file.name}"):
            generate_response(file_prompt)
    
    # Chat input
    user_input = st.chat_input(
        "Type your message here..."
    )
    
    # Process user input if provided
    if user_input:
        generate_response(user_input)

if __name__ == "__main__":
    main()