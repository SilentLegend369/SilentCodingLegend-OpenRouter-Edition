#!/usr/bin/env python3
"""
Command-line utility for managing OpenRouter models.
This script allows you to view available models, refresh the model cache,
and check the status of the API connection.
"""

import os
import sys
import argparse
from src.model_config import (
    display_available_models, 
    refresh_model_cache, 
    MODELS_BY_PROVIDER,
    CACHE_FILE,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL
)

def parse_args():
    parser = argparse.ArgumentParser(description="Manage OpenRouter models for the SilentCodingLegend AI agent")
    
    # Main actions
    parser.add_argument("--list", "-l", action="store_true", help="List all available models")
    parser.add_argument("--refresh", "-r", action="store_true", help="Force refresh the model cache")
    parser.add_argument("--check-api", "-c", action="store_true", help="Check the API connection status")
    parser.add_argument("--provider", "-p", type=str, help="Filter models by specific provider")
    parser.add_argument("--free-only", "-f", action="store_true", help="Show only free models")
    parser.add_argument("--info", "-i", action="store_true", help="Show information about the default model")
    parser.add_argument("--disable-cache", "-d", action="store_true", help="Disable model caching (for this session)")
    parser.add_argument("--enable-cache", "-e", action="store_true", help="Enable model caching (for this session)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    return parser.parse_args()

def check_api_key():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("\033[91m✗ Error: OpenRouter API key not found in environment variables\033[0m")
        print("Please set the OPENROUTER_API_KEY environment variable.")
        print("You can add it to your .env file or set it directly:")
        print("  export OPENROUTER_API_KEY=your_key_here")
        return False
    print("\033[92m✓ OpenRouter API key found\033[0m")
    return True

def check_cache():
    # Check if caching is disabled
    if os.getenv("DISABLE_MODEL_CACHE", "false").lower() == "true":
        print("\033[93m! Model caching is disabled\033[0m")
        return False
    
    if CACHE_FILE.exists():
        import time
        import json
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            
            # Validate cache structure
            if 'timestamp' not in cache_data or 'models' not in cache_data:
                print("\033[93m! Cache file has invalid structure\033[0m")
                return False
            
            # Calculate cache age
            cache_age = time.time() - cache_data['timestamp']
            cache_age_minutes = int(cache_age / 60)
            if cache_age_minutes < 60:
                age_str = f"{cache_age_minutes} minutes"
            else:
                cache_age_hours = cache_age_minutes / 60
                age_str = f"{cache_age_hours:.1f} hours"
            
            # Check if cache is valid
            model_count = sum(len(models) for models in cache_data.get('models', {}).values())
            if model_count == 0:
                print("\033[93m! Cache exists but contains no models\033[0m")
                return False
            
            # Calculate cache expiry
            from src.model_config import CACHE_EXPIRY
            if cache_age > CACHE_EXPIRY:
                print(f"\033[93m! Cache has expired\033[0m (Age: {age_str})")
                return False
                
            print(f"\033[92m✓ Valid cache found\033[0m (Age: {age_str}, Models: {model_count})")
            return True
        except json.JSONDecodeError:
            print(f"\033[93m! Cache file contains invalid JSON\033[0m")
            return False
        except Exception as e:
            print(f"\033[93m! Cache file exists but could not be read: {e}\033[0m")
            return False
    else:
        print("\033[93m! No cache file found\033[0m")
        return False

def filter_models_by_provider(provider):
    if provider not in MODELS_BY_PROVIDER:
        print(f"\033[91m✗ Provider '{provider}' not found\033[0m")
        print("Available providers:")
        for p in sorted(MODELS_BY_PROVIDER.keys()):
            model_count = len(MODELS_BY_PROVIDER[p])
            print(f"  - {p} ({model_count} models)")
        return None
    
    models = MODELS_BY_PROVIDER[provider]
    return {provider: models}

def filter_free_models(models_by_provider):
    free_models = {}
    for provider, models in models_by_provider.items():
        free_provider_models = {}
        for model_name, model_id in models.items():
            if ":free" in model_id:
                free_provider_models[model_name] = model_id
        if free_provider_models:
            free_models[provider] = free_provider_models
    return free_models

def show_model_info():
    print(f"\n=== Default Model Configuration ===")
    print(f"Provider: {DEFAULT_PROVIDER}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Model ID: {MODELS_BY_PROVIDER.get(DEFAULT_PROVIDER, {}).get(DEFAULT_MODEL, 'Unknown')}")
    print(f"\nAvailable providers: {', '.join(sorted(MODELS_BY_PROVIDER.keys()))}")
    
    # Count models
    total_models = sum(len(models) for provider, models in MODELS_BY_PROVIDER.items())
    free_models = sum(1 for provider, models in MODELS_BY_PROVIDER.items() 
                    for model_name, model_id in models.items() if ":free" in model_id)
    
    print(f"Total models: {total_models} (Free: {free_models}, Paid: {total_models - free_models})")

def main():
    args = parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parse_args().print_help()
        return
    
    # Handle debug mode
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("model_config").setLevel(logging.DEBUG)
        print("Debug mode enabled")
    
    # Handle cache disabling/enabling
    if args.disable_cache:
        os.environ["DISABLE_MODEL_CACHE"] = "true"
        print("\033[93mModel caching disabled for this session\033[0m")
    
    if args.enable_cache:
        os.environ["DISABLE_MODEL_CACHE"] = "false"
        print("\033[92mModel caching enabled for this session\033[0m")
    
    # Check API info
    if args.check_api:
        check_api_key()
        check_cache()
        # Show cache status
        if os.getenv("DISABLE_MODEL_CACHE", "false").lower() == "true":
            print("\033[93m! Model caching is disabled\033[0m")
        else:
            print("\033[92m✓ Model caching is enabled\033[0m")
    
    # Refresh cache if requested
    if args.refresh:
        if check_api_key():
            print("Refreshing model cache...")
            models = refresh_model_cache()
            free_count = sum(1 for p, m in models.items() for model_name, model_id in m.items() if ":free" in model_id)
            total_count = sum(len(m) for p, m in models.items())
            print(f"\033[92m✓ Successfully refreshed model cache\033[0m")
            print(f"  Total models: {total_count}")
            print(f"  Free models: {free_count}")
            print(f"  Paid models: {total_count - free_count}")
    
    # Display models if requested
    if args.list:
        filtered_models = MODELS_BY_PROVIDER
        
        # Filter by provider if specified
        if args.provider:
            filtered_models = filter_models_by_provider(args.provider)
            if filtered_models is None:
                return
        
        # Filter for free models if requested
        if args.free_only:
            filtered_models = filter_free_models(filtered_models)
        
        # Create a temporary dictionary with the filtered models
        temp_models_by_provider = {}
        for provider, models in filtered_models.items():
            temp_models_by_provider[provider] = models
        
        # Display the filtered models
        from src.model_config import display_available_models
        print(display_available_models(by_provider=True))
    
    # Show model info if requested
    if args.info:
        show_model_info()

if __name__ == "__main__":
    main()
