"""
Utility functions for the SilentCodingLegend AI agent.
"""
import os
import requests
import json
import logging
from collections import defaultdict
import streamlit as st # For st.secrets or st.error if needed in future
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

def apply_dark_theme():
    """
    Apply dark theme to Streamlit pages.
    This utility function loads the dark_theme.css file and applies it to the current page.
    If the file is not found, it applies a simplified dark theme inline CSS.
    """
    # Try multiple possible paths for the CSS file
    possible_paths = [
        'dark_theme.css',  # When running from root directory
        '../dark_theme.css',  # When running from pages directory
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dark_theme.css')  # Absolute path
    ]
    
    for path in possible_paths:
        try:
            with open(path, 'r') as f:
                dark_theme_css = f.read()
            st.markdown(f'<style>{dark_theme_css}</style>', unsafe_allow_html=True)
            return True
        except Exception as e:
            continue
    
    # Fallback to simplified inline CSS if the file isn't found
    dark_theme_css = """
    /* Main container and all elements */
    [data-testid="stAppViewContainer"], .stApp, body, section.main {
        background-color: #111827 !important;
        color: #f3f4f6 !important;
    }
    /* Fix for white area */
    [data-testid="stSidebar"] + section.main, div.block-container {
        background-color: #111827 !important;
    }
    """
    st.markdown(f'<style>{dark_theme_css}</style>', unsafe_allow_html=True)
    st.warning("Could not load dark theme CSS file, using simplified dark mode", icon="⚠️")
    return False

def fetch_and_process_models():
    """
    Fetches models from OpenRouter API, processes them, and sorts them.
    Sorts providers alphabetically.
    Sorts models within each provider: free models first (alphabetically by name), 
    then non-free models (alphabetically by name).
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        # This function might be called before Streamlit context is fully available for st.error
        print("ERROR: OpenRouter API key not found. Cannot fetch models.")
        return None

    api_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://scl-ai.streamlit.app"), # Replace with your app's actual URL
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "SCL OpenRouter Agent") # Replace with your app's title
        }
        response = requests.get(f"{api_base_url}/models", headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        api_models_data = response.json().get("data", [])
        if not api_models_data:
            print("Warning: No models returned from OpenRouter API.")
            return {}

        provider_models_temp = defaultdict(list)
        
        for model_data in api_models_data:
            model_id = model_data.get("id")
            model_display_name = model_data.get("name", model_id) # Fallback to ID if name is missing
            
            if not model_id:
                continue

            parts = model_id.split('/')
            provider_slug = parts[0] if len(parts) > 1 else "Other"
            provider_display_name = provider_slug.replace('-', ' ').replace('_', ' ').title()
            provider_models_temp[provider_display_name].append((model_display_name, model_id))

        sorted_models_by_provider = {}
        for provider_name in sorted(provider_models_temp.keys()):
            models_list = provider_models_temp[provider_name]
            # Sort models: free first (alphabetically), then non-free (alphabetically by name)
            models_list.sort(key=lambda m: (not m[1].endswith(':free'), m[0].lower()))
            sorted_models_by_provider[provider_name] = {name: id_ for name, id_ in models_list}
            
        return sorted_models_by_provider

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not fetch models from OpenRouter: {e}")
        return None
    except json.JSONDecodeError:
        print("ERROR: Could not decode JSON response from OpenRouter when fetching models.")
        return None

def get_best_model_for_task(task_type, provider_preference=None, require_free=True):
    """
    Get the best model for a specific task based on preferred providers.
    
    Args:
        task_type (str): The type of task ('general', 'coding', 'vision', 'reasoning')
        provider_preference (list): Optional list of preferred providers in order of preference
        require_free (bool): If True, only consider free models
        
    Returns:
        tuple: (model_name, model_id) or (None, None) if no suitable model found
    """
    from src.model_config import MODELS_BY_PROVIDER, PREFERRED_PROVIDERS
    
    # Use default preferences if none provided
    preferences = provider_preference or PREFERRED_PROVIDERS.get(task_type, PREFERRED_PROVIDERS["general"])
    
    # Try each preferred provider in order
    for provider in preferences:
        if provider in MODELS_BY_PROVIDER:
            provider_models = MODELS_BY_PROVIDER[provider]
            
            # Filter for free models if required
            if require_free:
                free_models = {name: model_id for name, model_id in provider_models.items() 
                               if ":free" in model_id}
                if free_models:
                    # Get the first free model (assuming they're already sorted by preference)
                    model_name = next(iter(free_models))
                    return model_name, free_models[model_name]
            else:
                # Get the first model from this provider (assuming they're already sorted)
                if provider_models:
                    model_name = next(iter(provider_models))
                    return model_name, provider_models[model_name]
    
    # If no suitable model found with preferred providers, try any provider
    for provider, models in MODELS_BY_PROVIDER.items():
        filtered_models = {name: model_id for name, model_id in models.items() 
                          if not require_free or ":free" in model_id}
        if filtered_models:
            model_name = next(iter(filtered_models))
            return model_name, filtered_models[model_name]
    
    return None, None

def list_models_for_task(task_type, include_paid=False):
    """
    List all suitable models for a specific task.
    
    Args:
        task_type (str): The type of task ('general', 'coding', 'vision', 'reasoning')
        include_paid (bool): Whether to include paid models
        
    Returns:
        list: List of (model_name, model_id, provider) tuples
    """
    from src.model_config import MODELS_BY_PROVIDER, PREFERRED_PROVIDERS
    
    # Get preferred providers for this task type
    preferred_providers = PREFERRED_PROVIDERS.get(task_type, PREFERRED_PROVIDERS["general"])
    
    # Create a list to store all suitable models
    suitable_models = []
    
    # First add models from preferred providers
    for provider in preferred_providers:
        if provider in MODELS_BY_PROVIDER:
            for model_name, model_id in MODELS_BY_PROVIDER[provider].items():
                if include_paid or ":free" in model_id:
                    suitable_models.append((model_name, model_id, provider))
    
    # Then add models from other providers
    for provider, models in MODELS_BY_PROVIDER.items():
        if provider not in preferred_providers:
            for model_name, model_id in models.items():
                if include_paid or ":free" in model_id:
                    suitable_models.append((model_name, model_id, provider))
    
    return suitable_models

def save_user_model_preference(task_type, model_id):
    """
    Save a user's model preference for a specific task type.
    
    Args:
        task_type (str): The type of task ('general', 'coding', 'vision', 'reasoning')
        model_id (str): The ID of the preferred model
    """
    try:
        # Get the preferences file path
        preferences_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        preferences_file = preferences_dir / "user_preferences.json"
        
        # Load existing preferences or create new
        if preferences_file.exists():
            with open(preferences_file, 'r') as f:
                preferences = json.load(f)
        else:
            preferences = {}
        
        # Update or create model preferences
        if 'model_preferences' not in preferences:
            preferences['model_preferences'] = {}
        
        # Update the preference for this task type
        preferences['model_preferences'][task_type] = model_id
        
        # Save back to file
        with open(preferences_file, 'w') as f:
            json.dump(preferences, f, indent=2)
            
        logger.info(f"Saved user preference for {task_type}: {model_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving user model preference: {e}")
        return False

def get_user_model_preference(task_type):
    """
    Get a user's saved model preference for a specific task type.
    
    Args:
        task_type (str): The type of task ('general', 'coding', 'vision', 'reasoning')
        
    Returns:
        str or None: The model ID or None if no preference is saved
    """
    try:
        # Get the preferences file path
        preferences_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        preferences_file = preferences_dir / "user_preferences.json"
        
        # Check if preferences file exists
        if not preferences_file.exists():
            return None
        
        # Load preferences
        with open(preferences_file, 'r') as f:
            preferences = json.load(f)
        
        # Check if we have a model preference for this task
        if 'model_preferences' in preferences and task_type in preferences['model_preferences']:
            return preferences['model_preferences'][task_type]
        
        return None
    except Exception as e:
        logger.error(f"Error getting user model preference: {e}")
        return None