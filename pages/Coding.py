"""
Coding page for the SilentCodingLegend AI agent.
This page provides specialized coding assistance using AI models optimized for programming.
"""

import streamlit as st
from src.utils import apply_dark_theme

# Page configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="Coding Assistant",
    page_icon="💻",
    layout="wide"
)

# Apply dark theme using utility function
apply_dark_theme()

import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
import re

from src.agents.openrouter_agent import OpenRouterAgent
from src.model_config import MODELS_BY_PROVIDER, DEFAULT_PROVIDER, DEFAULT_MODEL, DEFAULT_MODEL_ID

# Function to get models with strong coding capabilities
def get_coding_models(all_models_by_provider):
    """Get models that are particularly good at coding from the available models."""
    if not all_models_by_provider:
        return []
        
    coding_models = []
    
    # Models known to have strong coding capabilities
    coding_capable_models = [
        "anthropic/claude-3-opus",
        "anthropic/claude-3.5-sonnet",
        "qwen/qwen-2.5-coder",
        "meta-llama/llama-3",
        "meta-llama/llama-3.1",
        "meta-llama/llama-4",
        "mistral/mistral-large",
        "openai/gpt-4",
        "openai/gpt-4o",
        "google/gemini-1.5-pro",
        "google/gemini-2.5-pro",
        "agentica-org/deepcoder"
    ]
    
    # Additional keywords that suggest coding capabilities
    coding_keywords = [
        "coder", "code", "deepcoder", "starcoder", "codellama", "wizard-coder"
    ]
    
    # Check all available models
    for provider, models in all_models_by_provider.items():
        for model_name, model_id in models.items():
            # Check if the model ID starts with known coding-capable models
            is_coding_model = any(model_id.startswith(prefix) for prefix in coding_capable_models)
            
            # Or if it contains coding keywords
            has_coding_keyword = any(keyword.lower() in model_id.lower() or 
                                    keyword.lower() in model_name.lower() 
                                    for keyword in coding_keywords)
            
            if is_coding_model or has_coding_keyword:
                coding_models.append({
                    "provider": provider,
                    "name": model_name,
                    "id": model_id,
                    "display": f"{provider} - {model_name}"
                })
    
    return coding_models

# Function to backup coding history
def backup_coding_history(history_data):
    """Backup coding history to the Chat_History directory."""
    # Create the backup directory if it doesn't exist
    backup_dir = Path("/home/silentlegendkali/scl-openrouter/Chat_History")
    backup_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Backup as JSON
    json_path = backup_dir / f"coding_history_{timestamp}.json"
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
                "source": "Coding Assistant"  # Mark the source
            }
            st.session_state.chat_history.append(chat_entry)
    
    return json_path

# Function for formatting code responses
def format_code_response(response):
    """Format code responses with syntax highlighting."""
    # Pattern to identify code blocks with specific language
    pattern = r"```([a-zA-Z0-9_\+\-#]+)?\n(.*?)```"
    
    # Function to replace code blocks with formatted HTML
    def replace_code_block(match):
        language = match.group(1) or ""
        code = match.group(2)
        formatted = f"<pre><code class='language-{language}'>{code}</code></pre>"
        return formatted
    
    # Replace all code blocks in the text
    formatted_response = re.sub(pattern, replace_code_block, response, flags=re.DOTALL)
    
    return formatted_response

# Function to generate coding assistance
def generate_code_assistance(prompt, model_id, temperature=0.3, max_tokens=2048):
    """Generate coding assistance using the selected model."""
    try:
        # Create the agent with the selected model
        agent = OpenRouterAgent(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create system prompt for coding assistance
        system_prompt = """
        You are an expert coding assistant specialized in helping with programming tasks.
        Provide clear, efficient, and well-documented code solutions.
        When providing code:
        - Include helpful comments to explain complex logic
        - Suggest best practices relevant to the task
        - Highlight potential issues or edge cases
        - Provide explanations before and after code blocks
        - Format code properly within markdown code blocks with appropriate language tags
        - If relevant, suggest alternative approaches with their pros and cons
        
        Your goal is to help the user write better, more efficient, and more maintainable code.
        """
        
        # Generate response
        response = agent.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response
    except Exception as e:
        return f"Error generating code assistance: {str(e)}"

# Initialize session state variables
if "coding_history" not in st.session_state:
    st.session_state.coding_history = []

# Initialize selected_coding_model if not already set
if "selected_coding_model" not in st.session_state:
    preferred_model_ids = [
        "qwen/qwen-2.5-coder-32b-instruct:free",
        "anthropic/claude-3.5-sonnet:thinking",
        "agentica-org/deepcoder-14b-preview:free",
        "meta-llama/llama-4-scout:free"
    ]
    default_set = False
    available_coding_models = get_coding_models(MODELS_BY_PROVIDER)

    for preferred_id in preferred_model_ids:
        for model_info in available_coding_models:
            if model_info["id"] == preferred_id:
                st.session_state.selected_coding_model = preferred_id
                default_set = True
                break
        if default_set:
            break
    
    if not default_set and available_coding_models:
        # Fallback to the first available coding model if preferred ones are not found
        st.session_state.selected_coding_model = available_coding_models[0]["id"]
    elif not default_set:
        # Ultimate fallback if no coding models are found at all
        # This might happen if the list is empty or filtering is too strict
        st.warning("No specific coding models found. Defaulting to a general model if available.")
        found = False
        for provider_models in MODELS_BY_PROVIDER.values():
            if provider_models:
                st.session_state.selected_coding_model = list(provider_models.values())[0] # First model from first provider
                found = True
                break
        if not found:
             st.session_state.selected_coding_model = None # No models available at all

# Auto-backup if history has changed
if "last_coding_history_length" not in st.session_state:
    st.session_state.last_coding_history_length = 0

current_length = len(st.session_state.coding_history)
if current_length > 0 and current_length != st.session_state.last_coding_history_length:
    # Perform backup
    json_path = backup_coding_history(st.session_state.coding_history)
    st.session_state.last_coding_history_length = current_length

# Main page UI
st.title("💻 Coding Assistant")
st.subheader("Get AI assistance with code, algorithms, debugging, and more")

# Initialize additional session state variables if not already created
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Code Assistant"
    
if "coding_prompt_input" not in st.session_state:
    st.session_state.coding_prompt_input = ""
    
if "code_input" not in st.session_state:
    st.session_state.code_input = ""
    
if "analysis_code_input" not in st.session_state:
    st.session_state.analysis_code_input = ""

# Initialize prompt input if not already created
if "coding_prompt_input" not in st.session_state:
    st.session_state.coding_prompt_input = ""

# Coding models information
with st.expander("ℹ️ About Coding Models"):
    st.markdown("""
    ### Specialized Coding Models
    
    This page features models that excel at programming-related tasks:
    
    - **Code Generation**: Create efficient code based on requirements
    - **Debugging**: Find and fix issues in existing code
    - **Refactoring**: Improve code structure and readability
    - **Algorithm Design**: Develop algorithmic solutions to problems
    - **Documentation**: Generate clear code documentation
    - **Best Practices**: Receive guidance on coding standards
    
    **Recommended Models:**
    
    - **Qwen Coder**: Specialized for programming tasks with excellent multilingual code generation
    - **DeepCoder**: Optimized for complex programming challenges
    - **Claude Models**: Strong at understanding code context and providing detailed explanations
    - **LLaMA Coding Variants**: Good at generating efficient code across multiple languages
    
    For best results, use lower temperature settings (0.1-0.3) when you need precise, deterministic code.
    """)

# Qwen coding models information
with st.expander("ℹ️ About Qwen Coding Models"):
    st.markdown("""
    ### Qwen Coding Models
    
    Qwen coding models are specifically fine-tuned for programming tasks with several advantages:
    
    - **Comprehensive Language Support**: Excellent performance across Python, JavaScript, Java, C++, Go, Rust, and more
    - **Code Completion**: Intelligent code completion and suggestions as you type
    - **Documentation Generation**: Automatically create clear documentation for code
    - **Code Translation**: Convert code between different programming languages
    - **Technical Interview**: Help with technical interview preparation and algorithm challenges
    - **Large Context Windows**: Can understand and work with larger code bases and files
    
    **When to use Qwen coding models:**
    - For building complex applications with well-structured code
    - When you need detailed explanations along with code
    - For refactoring and improving existing codebases
    - When working on multilingual code projects
    
    The Qwen 2.5 Coder 32B model provides an excellent balance of capabilities and performance for most coding tasks.
    """)

# Sidebar for model selection and settings
with st.sidebar:
    st.title("Coding Settings")
    
    # Get available coding models
    coding_models = get_coding_models(MODELS_BY_PROVIDER)

    if st.session_state.selected_coding_model is None and coding_models: # If it was None but now we have options
        st.session_state.selected_coding_model = coding_models[0]["id"]

    
    if not coding_models:
        st.warning("No specialized coding models found in your configuration.")
        st.info("Using default model for coding assistance.")
        coding_models = [{
            "provider": "Default",
            "name": "Model",
            "id": st.session_state.get("selected_coding_model"), # Use .get for safety
            "display": "Default Model"
        }]
    
    # Model selection
    model_options = [model["display"] for model in coding_models]
    model_ids = [model["id"] for model in coding_models]
    
    # Find the current model index
    selected_index = 0
    if st.session_state.selected_coding_model in model_ids:
        selected_index = model_ids.index(st.session_state.selected_coding_model)
    elif model_ids: # If current selection is invalid, default to first
        st.session_state.selected_coding_model = model_ids[0]

    
    selected_model = st.selectbox(
        "Select Coding Model",
        options=model_options,
        index=min(selected_index, len(model_options)-1),
        key="coding_model_selection"
    )
    
    # Update the model ID based on selection
    if model_options: # Ensure model_options is not empty
        st.session_state.selected_coding_model = model_ids[model_options.index(selected_model)]
    
    # Temperature setting - Using lower default for coding tasks
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,  # Lower default for coding
        step=0.1,
        help="Controls randomness: Lower values (0.0-0.3) are better for precise code generation. Higher values can be more creative."
    )
    
    # Max tokens
    max_tokens = st.slider(
        "Max Tokens",
        min_value=512,
        max_value=8192,
        value=2048,
        step=512,
        help="Maximum length of the AI response. Higher values allow for more detailed code and explanations."
    )
    
    # Coding task templates
    st.subheader("Task Templates")
    
    coding_templates = {
        "Generate Function": "Write a function that {task_description}. Use {language} programming language.",
        "Debug Code": "Debug this code and explain what's wrong:\n```\n{code}\n```",
        "Optimize Code": "Optimize this code for better performance:\n```\n{code}\n```",
        "Explain Code": "Explain what this code does in detail:\n```\n{code}\n```",
        "Convert Code": "Convert this code from {source_language} to {target_language}:\n```\n{code}\n```",
        "Unit Test": "Write unit tests for the following code:\n```\n{code}\n```",
        "Algorithm": "Design an algorithm to solve this problem: {problem_description}",
        "Data Structure": "Explain how to implement {data_structure} and provide an example in {language}.",
        "Design Pattern": "Show how to implement the {design_pattern} pattern in {language} with an example."
    }
    
    selected_template = st.selectbox(
        "Choose a coding template",
        options=list(coding_templates.keys())
    )
    
    if "coding_prompt_input" not in st.session_state:
        st.session_state.coding_prompt_input = ""
    
    if st.button("Apply Template"):
        st.session_state.coding_prompt_input = coding_templates[selected_template]
        st.rerun()

# Main content area - Tabbed Interface
tab1, tab2, tab3, tab4 = st.tabs(["💬 Code Assistant", "📝 Code Editor", "📊 Code Analyzer", "📜 History"])

# Tab 1: Standard Code Assistant
with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Coding prompt input - Using larger height for code entry
        prompt = st.text_area(
            "Enter your coding question or task",
            value=st.session_state.coding_prompt_input, 
            height=200,
            placeholder="Example: Write a Python function that calculates the Fibonacci sequence up to n terms."
        )
        st.session_state.coding_prompt_input = prompt
        
        # Generate button
        if st.button("Generate Solution"):
            if prompt:
                with st.spinner("Generating code solution..."):
                    # Call the generate_code_assistance function
                    response = generate_code_assistance(
                        prompt, 
                        st.session_state.selected_coding_model,
                        temperature,
                        max_tokens
                    )
                    
                    # Display the response
                    st.markdown("### Solution:")
                    st.markdown(response)
                    
                    # Save to history
                    st.session_state.coding_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": st.session_state.selected_coding_model,
                        "prompt": prompt,
                        "response": response
                    })
                    
                    # Backup current analysis
                    json_path = backup_coding_history([st.session_state.coding_history[-1]])
                    
                    # Update the history length
                    st.session_state.last_coding_history_length = len(st.session_state.coding_history)
            else:
                st.warning("Please enter a coding question or task before generating.")

# Tab 2: Code Editor with AI suggestions
with tab2:
    st.subheader("Interactive Code Editor")
    
    # Language selection
    languages = ["Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust", "SQL", "PHP", "HTML/CSS", "Other"]
    selected_language = st.selectbox("Select Language", languages)
    
    # Initialize code input if needed
    if "code_input" not in st.session_state:
        st.session_state.code_input = ""
    
    # Code editor
    code_input = st.text_area(
        "Write or paste your code here",
        value=st.session_state.code_input,
        height=300,
        placeholder=f"Enter your {selected_language} code here..."
    )
    st.session_state.code_input = code_input
    
    # Code actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Improve Code"):
            if code_input:
                with st.spinner("Analyzing and improving code..."):
                    prompt = f"Improve this {selected_language} code. Make it more efficient, readable, and follow best practices. Explain the improvements made.\n\n```{selected_language.lower()}\n{code_input}\n```"
                    response = generate_code_assistance(prompt, st.session_state.selected_coding_model, temperature, max_tokens)
                    
                    st.markdown("### Improved Code:")
                    st.markdown(response)
                    
                    # Save to history
                    st.session_state.coding_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": st.session_state.selected_coding_model,
                        "prompt": prompt,
                        "response": response,
                        "action": "Improve Code"
                    })
                    
                    # Backup
                    backup_coding_history([st.session_state.coding_history[-1]])
                    st.session_state.last_coding_history_length = len(st.session_state.coding_history)
            else:
                st.warning("Please enter some code first.")
    
    with col2:
        if st.button("Add Comments"):
            if code_input:
                with st.spinner("Adding detailed comments..."):
                    prompt = f"Add comprehensive comments to this {selected_language} code to make it more understandable. Explain what each section does and why.\n\n```{selected_language.lower()}\n{code_input}\n```"
                    response = generate_code_assistance(prompt, st.session_state.selected_coding_model, temperature, max_tokens)
                    
                    st.markdown("### Commented Code:")
                    st.markdown(response)
                    
                    # Save to history
                    st.session_state.coding_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": st.session_state.selected_coding_model,
                        "prompt": prompt,
                        "response": response,
                        "action": "Add Comments"
                    })
                    
                    # Backup
                    backup_coding_history([st.session_state.coding_history[-1]])
                    st.session_state.last_coding_history_length = len(st.session_state.coding_history)
            else:
                st.warning("Please enter some code first.")
                
    with col3:
        if st.button("Debug Code"):
            if code_input:
                with st.spinner("Debugging code..."):
                    prompt = f"Debug this {selected_language} code. Identify any issues, errors, or potential bugs, and explain how to fix them.\n\n```{selected_language.lower()}\n{code_input}\n```"
                    response = generate_code_assistance(prompt, st.session_state.selected_coding_model, temperature, max_tokens)
                    
                    st.markdown("### Debugging Results:")
                    st.markdown(response)
                    
                    # Save to history
                    st.session_state.coding_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": st.session_state.selected_coding_model,
                        "prompt": prompt,
                        "response": response,
                        "action": "Debug Code"
                    })
                    
                    # Backup
                    backup_coding_history([st.session_state.coding_history[-1]])
                    st.session_state.last_coding_history_length = len(st.session_state.coding_history)
            else:
                st.warning("Please enter some code first.")

# Tab 3: Code Analyzer
with tab3:
    st.subheader("Code Analysis Tools")
    
    analysis_options = [
        "Generate Unit Tests",
        "Complexity Analysis",
        "Security Review",
        "Convert to Another Language",
        "Generate Documentation",
        "Performance Optimization"
    ]
    
    selected_analysis = st.selectbox("Select Analysis Type", analysis_options)
    
    # Language selection for code input
    languages = ["Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust", "SQL", "PHP", "HTML/CSS", "Other"]
    source_language = st.selectbox("Source Language", languages, key="source_lang")
    
    if selected_analysis == "Convert to Another Language":
        target_language = st.selectbox("Target Language", languages, key="target_lang")
    
    # Code input
    if "analysis_code_input" not in st.session_state:
        st.session_state.analysis_code_input = ""
        
    analysis_code = st.text_area(
        "Enter code to analyze",
        value=st.session_state.analysis_code_input,
        height=250,
        placeholder=f"Enter your {source_language} code here..."
    )
    st.session_state.analysis_code_input = analysis_code
    
    # Analyze button
    if st.button("Run Analysis"):
        if analysis_code:
            with st.spinner(f"Running {selected_analysis}..."):
                # Create appropriate prompt based on analysis type
                if selected_analysis == "Generate Unit Tests":
                    prompt = f"Generate comprehensive unit tests for this {source_language} code. Include test cases for normal operation, edge cases, and error conditions.\n\n```{source_language.lower()}\n{analysis_code}\n```"
                elif selected_analysis == "Complexity Analysis":
                    prompt = f"Analyze the time and space complexity of this {source_language} code. Identify the big O notation for key functions and any bottlenecks.\n\n```{source_language.lower()}\n{analysis_code}\n```"
                elif selected_analysis == "Security Review":
                    prompt = f"Perform a security review of this {source_language} code. Identify potential vulnerabilities, security issues, and suggest fixes.\n\n```{source_language.lower()}\n{analysis_code}\n```"
                elif selected_analysis == "Convert to Another Language":
                    prompt = f"Convert this code from {source_language} to {target_language}. Maintain the same functionality and structure, using idiomatic patterns in the target language.\n\n```{source_language.lower()}\n{analysis_code}\n```"
                elif selected_analysis == "Generate Documentation":
                    prompt = f"Generate comprehensive documentation for this {source_language} code. Include function descriptions, parameter details, return values, and usage examples.\n\n```{source_language.lower()}\n{analysis_code}\n```"
                elif selected_analysis == "Performance Optimization":
                    prompt = f"Optimize this {source_language} code for performance. Identify bottlenecks, suggest improvements, and explain why they would improve performance.\n\n```{source_language.lower()}\n{analysis_code}\n```"
                
                # Generate response
                response = generate_code_assistance(prompt, st.session_state.selected_coding_model, temperature, max_tokens)
                
                st.markdown(f"### {selected_analysis} Results:")
                st.markdown(response)
                
                # Save to history
                st.session_state.coding_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model": st.session_state.selected_coding_model,
                    "prompt": prompt,
                    "response": response,
                    "action": selected_analysis
                })
                
                # Backup
                backup_coding_history([st.session_state.coding_history[-1]])
                st.session_state.last_coding_history_length = len(st.session_state.coding_history)
        else:
            st.warning("Please enter some code to analyze first.")

# Tab 4: History
with tab4:
    st.subheader("Full Coding History")
    
    if not st.session_state.coding_history:
        st.info("No coding history yet. Use the other tabs to generate coding solutions.")
    else:
        # Add filtering options
        filter_options = ["All"]
        
        # Collect unique action types from history
        action_types = set()
        for entry in st.session_state.coding_history:
            action = entry.get('action', 'Code Query')
            action_types.add(action)
        
        # Add collected action types to filter options
        filter_options.extend(sorted(list(action_types)))
        
        # Filter selection
        selected_filter = st.selectbox("Filter by action type:", filter_options)
        
        # Search box
        search_query = st.text_input("Search in history:", placeholder="Enter keywords to search...")
        
        # Apply filters
        filtered_history = []
        for entry in st.session_state.coding_history:
            entry_action = entry.get('action', 'Code Query')
            
            # Filter by action type
            if selected_filter != "All" and entry_action != selected_filter:
                continue
                
            # Filter by search query
            if search_query and search_query.lower() not in entry.get('prompt', '').lower() and search_query.lower() not in entry.get('response', '').lower():
                continue
                
            filtered_history.append(entry)
        
        # Display filtered history
        st.write(f"Displaying {len(filtered_history)} of {len(st.session_state.coding_history)} entries")
        
        # Sort by most recent first
        filtered_history = list(reversed(filtered_history))
        
        # Display entries in expandable sections
        for i, entry in enumerate(filtered_history):
            entry_action = entry.get('action', 'Code Query')
            timestamp = entry.get('timestamp', '')
            model = entry.get('model', 'Unknown')
            
            with st.expander(f"{entry_action} - {timestamp} - {model.split('/')[-1]}"):
                # Create tabs for prompt and response
                prompt_tab, response_tab = st.tabs(["Prompt", "Response"])
                
                with prompt_tab:
                    st.markdown("### Query:")
                    st.write(entry['prompt'])
                    
                with response_tab:
                    st.markdown("### Solution:")
                    st.markdown(entry['response'])
                
                # Reuse button - Adds the prompt to the active tab
                if st.button(f"Reuse This Prompt {i}", key=f"reuse_{i}"):
                    st.session_state.coding_prompt_input = entry['prompt']
                    st.session_state.active_tab = "Code Assistant"
                    st.rerun()
        
        # Export and backup options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export History to JSON"):
                # Create a JSON file in the Chat_History directory
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                json_path = backup_coding_history(st.session_state.coding_history)
                st.success(f"History exported to JSON successfully")
                
        with col2:
            if st.button("Clear All History"):
                st.session_state.coding_history = []
                st.session_state.last_coding_history_length = 0
                st.success("History cleared successfully")
                st.rerun()

# Coding languages reference
with st.expander("📚 Language Reference"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Popular Languages
        - **Python**: Data science, web backends, automation
        - **JavaScript**: Web development, frontend, Node.js
        - **Java**: Enterprise apps, Android development
        - **C#**: Windows apps, game development (Unity)
        - **C++**: System programming, game engines, performance-critical applications
        - **TypeScript**: Type-safe JavaScript development
        - **Go**: Cloud services, microservices, concurrent systems
        - **Rust**: Systems programming with memory safety
        - **Swift**: iOS and macOS application development
        - **Kotlin**: Modern Android development
        """)
    
    with col2:
        st.markdown("""
        ### Specialized Languages
        - **SQL**: Database queries and management
        - **R**: Statistical computing and data visualization
        - **PHP**: Web development, particularly server-side
        - **Ruby**: Web development with Ruby on Rails
        - **Scala**: Functional programming, big data with Spark
        - **MATLAB**: Scientific computing and engineering
        - **Haskell**: Functional programming
        - **Julia**: High-performance scientific computing
        - **Solidity**: Ethereum smart contracts
        - **Dart**: Cross-platform development with Flutter
        """)

# Tips section
with st.expander("💡 Tips for Better Coding Assistance"):
    st.markdown("""
    ### How to Get Better Results
    
    1. **Be specific** in your requests
       - Instead of "Write a sorting function", try "Write a Python function that implements merge sort for an array of integers with detailed comments explaining each step"
    
    2. **Include relevant context**
       - Mention frameworks, libraries, or specific versions you're working with
       - Provide information about your environment (OS, runtime version, etc.)
    
    3. **For debugging help**
       - Include the complete error message
       - Provide enough code to reproduce the issue
       - Explain what you've already tried
    
    4. **For optimization**
       - Specify what aspect you want to optimize (speed, memory usage, readability)
       - Include performance constraints or requirements
    
    5. **Use appropriate temperature settings**
       - Lower (0.1-0.3) for deterministic, precise code
       - Medium (0.4-0.6) for balanced solutions
       - Higher (0.7-0.9) for creative approaches or brainstorming
    """)

# Common coding scenarios
with st.expander("📋 Common Coding Scenarios"):
    st.markdown("""
    ### Effective Prompts for Common Scenarios
    
    #### Web Development
    - "Create a responsive navigation bar with HTML, CSS, and JavaScript that collapses into a hamburger menu on mobile"
    - "Write a React component for a dynamic form that validates user input"
    - "Implement a Node.js API endpoint that queries a MongoDB database and returns paginated results"
    
    #### Data Science
    - "Write a Python function to preprocess this dataset including handling missing values and normalizing numeric columns"
    - "Create a visualization for this time series data showing trend and seasonality using matplotlib"
    - "Implement a simple machine learning pipeline using scikit-learn for this classification problem"
    
    #### DevOps
    - "Write a Docker Compose file for a web application with separate containers for frontend, backend, and database"
    - "Create a GitHub Actions workflow that tests and deploys a Python application"
    - "Implement a Bash script to automate backup of a PostgreSQL database"
    
    #### Mobile Development
    - "Create a SwiftUI view that displays a list of items with custom styling"
    - "Write a Kotlin function for an Android app to handle push notifications"
    - "Implement a Flutter widget for a customizable profile card"
    
    #### Security
    - "Write a function to securely hash and verify passwords in Python"
    - "Implement input validation to prevent SQL injection in this PHP code"
    - "Create a JWT authentication middleware for an Express.js application"
    """)

# Advanced capabilities
with st.expander("🚀 Advanced Coding Capabilities"):
    st.markdown("""
    ### What Our Coding Models Can Do
    
    #### Code Generation
    - Complete functions and classes based on descriptions or signatures
    - Generate code from natural language specifications
    - Create boilerplate code for common patterns and structures
    - Implement algorithms and data structures from descriptions
    
    #### Code Understanding
    - Explain complex code line-by-line
    - Summarize the functionality of functions or classes
    - Identify potential bugs or edge cases
    - Analyze the approach used in an implementation
    
    #### Code Transformation
    - Refactor code for improved readability or performance
    - Convert code between different programming languages
    - Update code to use newer language features or APIs
    - Modify code to follow specific design patterns
    
    #### Development Assistance
    - Generate unit tests with high coverage
    - Create documentation and docstrings for existing code
    - Debug and fix issues in problematic code
    - Optimize code for specific performance requirements
    
    #### Learning Support
    - Explain programming concepts with examples
    - Provide step-by-step guides for implementing features
    - Compare different approaches to solving a problem
    - Create learning materials and exercises
    """)

# Educational resources section
with st.expander("📚 Learn Coding Resources"):
    st.markdown("""
    ### Learning Resources
    
    #### Beginner Resources
    - **[freeCodeCamp](https://www.freecodecamp.org/)** - Free courses on web development, data science, and more
    - **[Codecademy](https://www.codecademy.com/)** - Interactive courses on many programming languages
    - **[The Odin Project](https://www.theodinproject.com/)** - Full web development curriculum
    - **[LeetCode](https://leetcode.com/)** - Practice coding problems and algorithms
    
    #### Practice Projects
    
    Try asking the coding assistant to help you create:
    
    1. **A personal portfolio website** (HTML/CSS/JavaScript)
    2. **A to-do list application** (Any language)
    3. **A weather app that uses an API** (Python, JavaScript, etc.)
    4. **A simple database CRUD application** (SQL + backend language)
    5. **A basic machine learning model** (Python with scikit-learn)
    
    #### How to ask the assistant for learning help:
    
    - "Explain [concept] with simple examples"
    - "Create a step-by-step tutorial for learning [technology]"
    - "What's the difference between [concept A] and [concept B]?"
    - "Show me a simple project to practice [language/framework]"
    - "What are common mistakes beginners make when learning [technology]?"
    """)
    
    st.markdown("---")
    
    # Add custom learning paths by language
    st.subheader("Learning Paths by Language")
    
    learning_language = st.selectbox(
        "Select a programming language to learn:",
        ["Python", "JavaScript", "Java", "C#", "Go", "Rust", "SQL", "TypeScript", "PHP", "Ruby"]
    )
    
    # Generate a custom learning path based on the selected language
    if st.button("Generate Learning Path"):
        with st.spinner(f"Creating a learning path for {learning_language}..."):
            prompt = f"Create a detailed learning path for someone who wants to learn {learning_language} programming. Include key concepts to master in order, recommended resources (websites, tutorials, books), practice project ideas for each stage (beginner, intermediate, advanced), and how to know when to move to the next level. Format this as a step-by-step guide with clear sections and bullet points."
            
            response = generate_code_assistance(
                prompt,
                st.session_state.selected_coding_model,
                temperature=0.5,
                max_tokens=2048
            )
            
            st.markdown(f"### {learning_language} Learning Path")
            st.markdown(response)
            
            # Save to history but don't display in main feed
            learning_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": st.session_state.selected_coding_model,
                "prompt": prompt,
                "response": response,
                "action": f"Learning Path - {learning_language}"
            }
            
            # Add to history without triggering backup to avoid clutter
            st.session_state.coding_history.append(learning_entry)