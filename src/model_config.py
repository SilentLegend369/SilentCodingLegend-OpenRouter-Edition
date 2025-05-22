"""
Model configuration file for the SilentCodingLegend AI agent.
This file contains configuration for the OpenRouter models, automatically fetched from the API
with fallback to hardcoded models if the API is unavailable.
"""

import os
import json
import time
import requests
import logging
from dotenv import load_dotenv
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_config")

# Load environment variables
load_dotenv()

# API configuration
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/models")
INCLUDE_PAID_MODELS = os.getenv("INCLUDE_PAID_MODELS", "false").lower() == "true"

# Path for caching model data
CACHE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_FILE = CACHE_DIR / "model_cache.json"
CACHE_EXPIRY = int(os.getenv("MODEL_CACHE_EXPIRY", "3600"))  # Cache expires after 1 hour (in seconds) by default

# Hardcoded models as fallback
FALLBACK_MODELS_BY_PROVIDER = {
    "Agentica": {
        "deepcoder-14b-preview": "agentica-org/deepcoder-14b-preview:free"
    },
    "Arliai": {
        "qwq-32b-arliai-rpr-v1": "arliai/qwq-32b-arliai-rpr-v1:free"
    },
    "Deepseek": {
        "deepseek-prover-v2": "deepseek/deepseek-prover-v2:free",
        "deepseek-chat": "deepseek/deepseek-chat:free",
        "deepseek-v3-base": "deepseek/deepseek-v3-base:free",
        "deepseek-chat-v3-0324": "deepseek/deepseek-chat-v3-0324:free",
        "deepseek-r1-zero": "deepseek/deepseek-r1-zero:free",
        "deepseek-r1-distill-qwen-14b": "deepseek/deepseek-r1-distill-qwen-14b:free",
        "deepseek-r1-distill-qwen-32b": "deepseek/deepseek-r1-distill-qwen-32b:free"
    },
    "Google": {
        "gemini-2.5-pro-exp-03-25": "google/gemini-2.5-pro-exp-03-25",
        "gemini-2.0-flash-exp": "google/gemini-2.0-flash-exp:free",
        "gemma-2-9b-it": "google/gemma-2-9b-it:free",
        "gemma-3-1b-it": "google/gemma-3-1b-it:free",
        "gemma-3-4b-it": "google/gemma-3-4b-it:free",
        "gemma-3-12b-it": "google/gemma-3-12b-it:free",
        "gemma-3-27b-it": "google/gemma-3-27b-it:free"
    },
    "Meta": {
        "llama-3.3-8b-instruct": "meta-llama/llama-3.3-8b-instruct:free",
        "llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct:free",
        "llama-4-scout": "meta-llama/llama-4-scout:free",
        "llama-4-maverick": "meta-llama/llama-4-maverick:free"
    },
    "Microsoft": {
        "mai-ds-r1": "microsoft/mai-ds-r1:free",
        "phi-4-reasoning": "microsoft/phi-4-reasoning:free",
        "phi-4-reasoning-plus": "microsoft/phi-4-reasoning-plus:free"
    },
    "OpenGVLab": {
        "internvl3-2b": "opengvlab/internvl3-2b:free",
        "internvl3-14b": "opengvlab/internvl3-14b:free"
    },
    "Qwen": {
        "qwen-2.5-7b-instruct": "qwen/qwen-2.5-7b-instruct:free",
        "qwen-2.5-72b-instruct": "qwen/qwen-2.5-72b-instruct:free",
        "qwen2.5-vl-3b-instruct": "qwen/qwen2.5-vl-3b-instruct:free",
        "qwen-2.5-vl-7b-instruct": "qwen/qwen-2.5-vl-7b-instruct:free",
        "qwen2.5-vl-32b-instruct": "qwen/qwen2.5-vl-32b-instruct:free",
        "qwen2.5-vl-72b-instruct": "qwen/qwen2.5-vl-72b-instruct:free",
        "qwen-2.5-coder-32b-instruct": "qwen/qwen-2.5-coder-32b-instruct:free",
        "qwen3-0.6b-04-28": "qwen/qwen3-0.6b-04-28:free",
        "qwen3-1.7b": "qwen/qwen3-1.7b:free",
        "qwen3-4b": "qwen/qwen3-4b:free",
        "qwen3-8b": "qwen/qwen3-8b:free",
        "qwen3-14b": "qwen/qwen3-14b:free",
        "qwen3-32b": "qwen/qwen3-32b:free",
        "qwen3-235b-a22b": "qwen/qwen3-235b-a22b:free",
        "qwq-32b": "qwen/qwq-32b:free"
    }
}

def refresh_model_cache():
    """
    Force a refresh of the model cache by fetching the latest models from the API.
    Returns a dictionary of models organized by provider.
    """
    logger.info("Manually refreshing model cache...")
    # Delete existing cache file if it exists
    if CACHE_FILE.exists():
        try:
            CACHE_FILE.unlink()
            logger.info("Deleted existing cache file")
        except Exception as e:
            logger.warning(f"Error deleting cache file: {e}")
    
    # Temporarily enable caching for this refresh if it was disabled
    original_disable_setting = os.getenv("DISABLE_MODEL_CACHE", "false")
    if original_disable_setting.lower() == "true":
        os.environ["DISABLE_MODEL_CACHE"] = "false"
        logger.info("Temporarily enabling caching for refresh operation")
    
    try:
        # Fetch fresh models from the API
        models = fetch_models_from_api(force_refresh=True)
        
        # Restore original setting
        if original_disable_setting.lower() == "true":
            os.environ["DISABLE_MODEL_CACHE"] = "true"
            # And delete the cache file again if caching was originally disabled
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
                logger.info("Deleted cache file as caching is disabled")
        
        return models
    except Exception as e:
        logger.error(f"Error refreshing model cache: {e}")
        # Restore original setting in case of error
        if original_disable_setting.lower() == "true":
            os.environ["DISABLE_MODEL_CACHE"] = "true"
        return FALLBACK_MODELS_BY_PROVIDER

def fetch_models_from_api(force_refresh=False):
    """
    Fetch models from the OpenRouter API.
    Returns a dictionary of models organized by provider.
    Falls back to hardcoded models if the API is unavailable.
    
    Args:
        force_refresh (bool): If True, ignore the cache and force a refresh from the API
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OpenRouter API key not found. Using fallback models.")
        return FALLBACK_MODELS_BY_PROVIDER
    
    # Check if caching is disabled
    if os.getenv("DISABLE_MODEL_CACHE", "false").lower() == "true":
        logger.info("Model caching is disabled. Fetching from API.")
    # Check if we have a valid cached response
    elif not force_refresh and CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            
            # Validate the cache data structure
            if 'timestamp' not in cache_data or 'models' not in cache_data:
                logger.warning("Invalid cache data structure. Fetching fresh data.")
            # Check if cache is empty
            elif not cache_data.get('models'):
                logger.warning("Empty cache data. Fetching fresh data.")
            # Check if cache is still valid
            elif time.time() - cache_data['timestamp'] < CACHE_EXPIRY:
                # Verify we have actual models in the cache
                model_count = sum(len(models) for provider, models in cache_data['models'].items())
                if model_count > 0:
                    logger.info(f"Using cached model data with {model_count} models")
                    return cache_data['models']
                else:
                    logger.warning("Cache contains no models. Fetching fresh data.")
            else:
                logger.info(f"Cache expired (older than {CACHE_EXPIRY} seconds). Fetching fresh data.")
        except json.JSONDecodeError:
            logger.warning("Cache file contains invalid JSON. Fetching fresh data.")
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
    elif force_refresh:
        logger.info("Force refresh requested. Bypassing cache.")
    
    # Fetch from API if no valid cache or force refresh
    try:
        logger.info(f"Fetching models from OpenRouter API at {OPENROUTER_API_URL}")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            OPENROUTER_API_URL,
            headers=headers,
            timeout=10  # Set timeout to 10 seconds
        )
        
        if response.status_code == 200:
            api_models = response.json()
            
            # Organize models by provider
            models_by_provider = {}
            
            for model in api_models.get('data', []):
                model_id = model.get('id')
                if not model_id:
                    continue
                
                # Extract provider name from model ID
                provider_parts = model_id.split('/')
                if len(provider_parts) >= 2:
                    provider_name = provider_parts[0].title()  # Capitalize provider name
                    model_name = '/'.join(provider_parts[1:])
                    
                    # Check if it's a free model or if it has pricing
                    is_free = False
                    if not model.get('pricing') or model.get('pricing', {}).get('prompt') == 0:
                        is_free = True
                        model_id = f"{model_id}:free"
                    
                    # Skip non-free models unless explicitly asked to include them
                    if not is_free and not INCLUDE_PAID_MODELS:
                        continue
                    
                    # Add to the provider dictionary
                    if provider_name not in models_by_provider:
                        models_by_provider[provider_name] = {}
                    
                    # Use a friendly display name
                    display_name = model_name.split(':')[0]
                    models_by_provider[provider_name][display_name] = model_id
            
            model_count = sum(len(models) for models in models_by_provider.values())
            logger.info(f"Successfully fetched {model_count} models from OpenRouter API")
            
            # If we got an empty result, use fallback
            if not models_by_provider:
                logger.warning("API returned no usable models. Using fallback models.")
                return FALLBACK_MODELS_BY_PROVIDER
            
            # Only cache if we actually have models and caching is enabled
            if model_count > 0 and os.getenv("DISABLE_MODEL_CACHE", "false").lower() != "true":
                try:
                    with open(CACHE_FILE, 'w') as f:
                        json.dump({
                            'timestamp': time.time(),
                            'models': models_by_provider
                        }, f, indent=2)
                    logger.info(f"Model data cached to {CACHE_FILE}")
                except Exception as e:
                    logger.warning(f"Error caching model data: {e}")
                
            return models_by_provider
            
        else:
            error_msg = f"API returned status code {response.status_code}"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg += f": {error_data['error']}"
            except:
                pass
            logger.warning(f"{error_msg}. Using fallback models.")
            return FALLBACK_MODELS_BY_PROVIDER
            
    except requests.exceptions.Timeout:
        logger.warning("API request timed out. Using fallback models.")
        return FALLBACK_MODELS_BY_PROVIDER
    except requests.exceptions.ConnectionError:
        logger.warning("Connection error while fetching models. Using fallback models.")
        return FALLBACK_MODELS_BY_PROVIDER
    except Exception as e:
        logger.warning(f"Error fetching models from API: {e}. Using fallback models.")
        return FALLBACK_MODELS_BY_PROVIDER

def display_available_models(by_provider=True):
    """
    Display a formatted list of all available models.
    
    Args:
        by_provider (bool): If True, organize the display by provider
    
    Returns:
        str: Formatted string with model information
    """
    if by_provider:
        output = []
        for provider, models in MODELS_BY_PROVIDER.items():
            output.append(f"\n=== {provider} ===")
            for model_name in sorted(models.keys()):
                model_id = models[model_name]
                is_free = ":free" in model_id
                output.append(f"  - {model_name}{' (Free)' if is_free else ''}")
        return "\n".join(output)
    else:
        output = ["\n=== All Available Models ==="]
        sorted_models = sorted(MODELS.keys())
        for model_key in sorted_models:
            model_id = MODELS[model_key]
            is_free = ":free" in model_id
            output.append(f"  - {model_key}{' (Free)' if is_free else ''}")
        return "\n".join(output)

# Get models organized by provider
MODELS_BY_PROVIDER = fetch_models_from_api()

# Flattened model list for compatibility
MODELS = {}
for provider, model_dict in MODELS_BY_PROVIDER.items():
    for model_name, model_id in model_dict.items():
        MODELS[f"{provider}/{model_name}"] = model_id

# Define preferred providers for different use cases
PREFERRED_PROVIDERS = {
    "general": ["Meta", "Google", "Anthropic", "OpenAI"],
    "coding": ["Qwen", "Agentica", "Meta", "Google"],
    "vision": ["Anthropic", "Qwen", "Google", "OpenAI"],
    "reasoning": ["Anthropic", "Google", "Qwen", "Meta"]
}

# Find default provider from available models
try:
    # First check if any preferred provider exists in our available models
    PREFERRED_DEFAULT_PROVIDER = next(
        (p for p in PREFERRED_PROVIDERS["general"] if p in MODELS_BY_PROVIDER), 
        None
    )
    
    # If no preferred provider found, use the first available provider
    if PREFERRED_DEFAULT_PROVIDER is None and MODELS_BY_PROVIDER:
        PREFERRED_DEFAULT_PROVIDER = list(MODELS_BY_PROVIDER.keys())[0]
    
    # If still None (no models available at all), use a hardcoded default
    if PREFERRED_DEFAULT_PROVIDER is None:
        PREFERRED_DEFAULT_PROVIDER = "Meta"
        print("Warning: No models available. Using hardcoded default provider: Meta")
        
    # Default model configuration
    DEFAULT_PROVIDER = PREFERRED_DEFAULT_PROVIDER
    
    # Safely get a default model from the provider
    if MODELS_BY_PROVIDER and DEFAULT_PROVIDER in MODELS_BY_PROVIDER and MODELS_BY_PROVIDER[DEFAULT_PROVIDER]:
        DEFAULT_MODEL = next(iter(MODELS_BY_PROVIDER[DEFAULT_PROVIDER].keys()))
        DEFAULT_MODEL_ID = MODELS_BY_PROVIDER[DEFAULT_PROVIDER][DEFAULT_MODEL]
    else:
        # Fallback to hardcoded values if no models are available
        DEFAULT_MODEL = "llama-3.3-8b-instruct"
        DEFAULT_MODEL_ID = "meta-llama/llama-3.3-8b-instruct:free"
        print("Warning: Using hardcoded fallback model: llama-3.3-8b-instruct")
except Exception as e:
    # Ultimate fallback in case of any errors
    logger.error(f"Error setting up default models: {e}. Using hardcoded fallbacks.")
    PREFERRED_DEFAULT_PROVIDER = "Meta"
    DEFAULT_PROVIDER = "Meta"
    DEFAULT_MODEL = "llama-3.3-8b-instruct"
    DEFAULT_MODEL_ID = "meta-llama/llama-3.3-8b-instruct:free"

# Model parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024