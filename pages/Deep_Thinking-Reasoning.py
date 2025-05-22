"""
Deep Thinking & Reasoning page for the SilentCodingLegend AI agent.
This page focuses on models with enhanced reasoning capabilities.
"""

import streamlit as st
from src.utils import apply_dark_theme

# Page configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="Deep Thinking & Reasoning",
    page_icon="🧠",
    layout="wide"
)

# Apply dark theme using utility function
apply_dark_theme()

import pandas as pd
from datetime import datetime
import json
import os
from pathlib import Path

from src.agents.openrouter_agent import OpenRouterAgent
from src.model_config import MODELS_BY_PROVIDER

# Function for backing up reasoning history
def backup_reasoning_history(history_data):
    """Backup reasoning history to the Chat_History directory."""
    # Create the backup directory if it doesn't exist
    backup_dir = Path("/home/silentlegendkali/scl-openrouter/Chat_History")
    backup_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Backup as CSV
    csv_path = backup_dir / f"reasoning_history_{timestamp}.csv"
    pd.DataFrame(history_data).to_csv(csv_path, index=False)
    
    # Backup as JSON
    json_path = backup_dir / f"reasoning_history_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    
    # Also add to the global chat history if it exists
    if "chat_history" in st.session_state:
        for item in history_data:
            chat_entry = {
                "timestamp": item.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "model": item.get("model", "Unknown"),
                "prompt": item.get("prompt", ""),
                "response": item.get("response", ""),
                "source": "Deep Thinking & Reasoning"  # Mark the source
            }
            st.session_state.chat_history.append(chat_entry)
    
    return csv_path, json_path

# Initialize session state for this page
if "dt_history" not in st.session_state:
    st.session_state.dt_history = []

# Track if we need to back up history
if "last_dt_history_length" not in st.session_state:
    st.session_state.last_dt_history_length = 0
    
# Auto-backup if history has changed
current_length = len(st.session_state.dt_history)
if current_length > 0 and current_length != st.session_state.last_dt_history_length:
    # Create history data in the right format for backup
    history_data = []
    for entry in st.session_state.dt_history:
        history_data.append({
            "timestamp": entry.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "model": entry.get("model", "Unknown"),
            "prompt": entry.get("prompt", ""),
            "response": entry.get("response", "")
        })
    
    # Perform backup
    csv_path, json_path = backup_reasoning_history(history_data)
    st.session_state.last_dt_history_length = current_length
    
if "dt_model" not in st.session_state:
    # Default to Microsoft's phi-4-reasoning model if available
    if "Microsoft" in MODELS_BY_PROVIDER and "phi-4-reasoning" in MODELS_BY_PROVIDER["Microsoft"]:
        st.session_state.dt_model = "microsoft/phi-4-reasoning:free"
    else:
        # Fallback to first reasoning-capable model
        reasoning_models = [
            "microsoft/phi-4-reasoning:free",
            "microsoft/phi-4-reasoning-plus:free",
            "anthropic/claude-3.5-sonnet:thinking",
            "deepseek/deepseek-r1-zero:free",
            "deepseek/deepseek-r1-distill-qwen-14b:free",
            "deepseek/deepseek-r1-distill-qwen-32b:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "qwen/qwen3-32b:free",
            "qwen/qwen3-235b-a22b:free",
            "qwen/qwq-32b:free",
            "meta-llama/llama-4-scout:free"
        ]
        
        # Find the first available reasoning model
        for model in reasoning_models:
            for provider in MODELS_BY_PROVIDER:
                for model_name, model_id in MODELS_BY_PROVIDER[provider].items():
                    if model_id == model:
                        st.session_state.dt_model = model
                        break

# Function to get suitable reasoning models
def get_reasoning_models():
    reasoning_models = []
    
    # Collect models known for reasoning capabilities
    reasoning_keywords = ["reasoning", "thinking", "scout", "r1", "sonnet", "opus", "qwen3", "qwq"]
    
    for provider, models in MODELS_BY_PROVIDER.items():
        for model_name, model_id in models.items():
            # Check if model name contains reasoning keywords
            if any(keyword in model_name.lower() for keyword in reasoning_keywords):
                reasoning_models.append({
                    "provider": provider,
                    "name": model_name,
                    "id": model_id,
                    "display": f"{provider} - {model_name}"
                })
    
    return reasoning_models

# Function to generate a reasoned response
def generate_reasoned_response(prompt=""):
    # Use either the passed prompt or get it from query params if exists
    user_prompt = prompt or st.query_params.get("prompt", "")
    
    if not user_prompt:
        st.warning("Please enter a prompt.")
        return
        
    # Create a reasoning agent with the selected model
    agent = OpenRouterAgent(
        model_id=st.session_state.dt_model,
        temperature=st.session_state.dt_temperature,
        max_tokens=st.session_state.dt_max_tokens
    )
    
    # Add reasoning instruction to system prompt
    base_system_prompt = st.session_state.dt_system_prompt if st.session_state.dt_system_prompt else ""
    reasoning_instruction = """
    Please think through this problem step by step. 
    First, break down the problem into smaller parts.
    Then, analyze each part logically.
    Consider different approaches and perspectives.
    Evaluate the evidence and draw conclusions based on sound reasoning.
    Show your work and explain your thought process clearly.
    For complex reasoning tasks, use mathematical notation when appropriate.
    """
    
    system_prompt = base_system_prompt + "\n\n" + reasoning_instruction if base_system_prompt else reasoning_instruction
    
    # Show thinking state
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.info("🧠 Thinking deeply...")
        
        # Generate response with reasoning
        response = agent.generate_response(
            user_prompt,
            system_prompt=system_prompt,
            temperature=st.session_state.dt_temperature,
            max_tokens=st.session_state.dt_max_tokens
        )
        
        # Display the response
        thinking_placeholder.markdown(response)
    
    # Add to history
    st.session_state.dt_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": st.session_state.dt_model,
        "prompt": user_prompt,
        "response": response
    })
    
    # Add to global chat history if it exists
    if "chat_history" in st.session_state:
        st.session_state.chat_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": st.session_state.dt_model,
            "prompt": user_prompt,
            "response": response,
            "source": "Deep Thinking & Reasoning"  # Mark the source
        })
    
# Main interface
st.title("🧠 Deep Thinking & Reasoning")
st.subheader("Leverage AI models with enhanced reasoning capabilities")

# Sidebar for configuration
with st.sidebar:
    st.title("Reasoning Settings")
    
    # Model selection
    reasoning_models = get_reasoning_models()
    model_options = [model["display"] for model in reasoning_models]
    model_ids = [model["id"] for model in reasoning_models]
    
    # Find the current model index
    selected_index = 0
    for i, model_id in enumerate(model_ids):
        if model_id == st.session_state.dt_model:
            selected_index = i
            break
    
    selected_model = st.selectbox(
        "Select Reasoning Model",
        options=model_options,
        index=selected_index,
        key="dt_model_selection"
    )
    
    # Update the model ID based on selection
    st.session_state.dt_model = model_ids[model_options.index(selected_model)]
    
    # Temperature setting
    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.5,  # Lower default for reasoning
        step=0.1,
        key="dt_temperature",
        help="Lower values (0.1-0.3) are better for logical reasoning. Higher values can be more creative."
    )
    
    # Max tokens
    st.slider(
        "Max Tokens",
        min_value=256,
        max_value=8192,
        value=2048,  # Higher default for reasoning
        step=256,
        key="dt_max_tokens",
        help="Maximum tokens for the response. Higher values allow for more detailed reasoning."
    )
    
    # System prompt
    st.text_area(
        "Custom System Prompt (Optional)",
        height=100,
        key="dt_system_prompt",
        help="Additional instructions for the AI. The reasoning instructions will be added automatically."
    )
    
    # Reasoning task templates
    st.subheader("Task Templates")
    
    reasoning_templates = {
        "Logical Problem": "Solve this logical problem step by step: {problem}",
        "Decision Analysis": "I'm trying to decide between: {options}. Help me analyze this decision step by step with pros and cons.",
        "Critical Evaluation": "Critically evaluate this argument: {argument}",
        "Root Cause Analysis": "Analyze the root causes of this issue: {issue}",
        "Ethical Dilemma": "Help me work through this ethical dilemma: {dilemma}",
        "Mathematical Problem": "Solve this mathematical problem step by step: {problem}",
        "Coding Challenge": "Solve this coding challenge and explain your reasoning step by step: {challenge}",
        "Multi-step Math": "Break down this complex mathematical problem and solve it step by step with clear explanations: {problem}"
    }
    
    selected_template = st.selectbox(
        "Choose a reasoning template",
        options=list(reasoning_templates.keys()),
        key="dt_template"
    )
    
    # Store template text in a session state variable that's not connected to a widget
    if "current_template" not in st.session_state:
        st.session_state.current_template = ""
    
    if st.button("Apply Template"):
        st.session_state.current_template = reasoning_templates[selected_template]
        # Set the URL query parameter to indicate we want to use a template
        st.query_params["use_template"] = "true"
        st.rerun()

# Main area - displays chat history and input
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat history
    for chat in st.session_state.dt_history:
        with st.chat_message("user"):
            st.write(chat["prompt"])
        with st.chat_message("assistant"):
            st.write(chat["response"])
    
    # Handle template application if requested
    initial_value = ""
    if st.query_params.get("use_template") == "true" and st.session_state.current_template:
        initial_value = st.session_state.current_template
        # Clear the query parameter and template after using it
        st.query_params.pop("use_template", None)
        placeholder_text = "Complete the template..."
    else:
        placeholder_text = "Enter your deep reasoning question..."
    
    # Chat input
    user_prompt = st.chat_input(
        placeholder_text,
        key="dt_chat_input"
    )
    
    # Process the input when submitted
    if user_prompt:
        generate_reasoned_response(user_prompt)

with col2:
    # Reasoning tips
    with st.expander("📝 Tips for Better Reasoning", expanded=True):
        st.markdown("""
        ### How to Get Better Results
        
        1. **Be specific** in your questions
        2. **Ask for step-by-step** analysis
        3. **Break complex problems** into smaller parts
        4. Use **lower temperature** (0.1-0.3) for logical tasks
        5. Use **higher temperature** (0.6-0.8) for creative reasoning
        6. Ask the model to **consider alternatives**
        7. Request **pros and cons** for decisions
        8. Have the model **evaluate its own reasoning**
        """)
    
    # Example use cases
    with st.expander("🔍 Example Use Cases"):
        st.markdown("""
        - **Problem Solving**: Mathematical or logical puzzles
        - **Decision Making**: Analyzing options with pros/cons
        - **Critical Thinking**: Evaluating arguments for flaws
        - **Root Cause Analysis**: Finding underlying issues
        - **Ethical Reasoning**: Working through moral dilemmas
        - **Scientific Reasoning**: Hypothesis formation and testing
        - **Legal Reasoning**: Analyzing legal scenarios
        """)
    
    # Qwen model capabilities
    with st.expander("🧮 Qwen Model Capabilities"):
        st.markdown("""
        ### Qwen Models for Reasoning
        
        **Qwen3 Series**
        - Excellent for multi-step mathematical reasoning
        - Strong at code-based problem solving
        - Good at logical deduction tasks
        
        **Qwen R1 Models**
        - Specialized for chain-of-thought reasoning
        - Enhanced capabilities for step-by-step analysis
        - Good at maintaining reasoning consistency
        
        **QWQ-32B**
        - Advanced reasoning with strong mathematical capabilities
        - Good for complex logic problems
        - Improved performance on scientific reasoning tasks
        """)
    
    # Clear history button
    if st.button("Clear Reasoning History"):
        st.session_state.dt_history = []
        st.rerun()
    
    # Backup history
    if st.session_state.dt_history:
        st.subheader("Backup Options")
        
        if st.button("Backup to Chat History"):
            # Create history data in the right format for backup
            history_data = []
            for entry in st.session_state.dt_history:
                history_data.append({
                    "timestamp": entry.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "model": entry.get("model", "Unknown"),
                    "prompt": entry.get("prompt", ""),
                    "response": entry.get("response", "")
                })
            
            # Perform backup
            csv_path, json_path = backup_reasoning_history(history_data)
            st.success(f"Reasoning history backed up to {csv_path.name} and {json_path.name}")
        
        # Download options
        st.subheader("Download Options")
        csv = pd.DataFrame(st.session_state.dt_history).to_csv(index=False)
        st.download_button(
            label="Download History as CSV",
            data=csv,
            file_name=f"reasoning_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Also offer JSON download option
        json_str = json.dumps(st.session_state.dt_history, indent=2)
        st.download_button(
            label="Download History as JSON",
            data=json_str,
            file_name=f"reasoning_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Link to Chat History page
        st.markdown("---")
        st.info("All reasoning conversations are also saved to the main Chat History. [View Chat History](/Chat_History)")