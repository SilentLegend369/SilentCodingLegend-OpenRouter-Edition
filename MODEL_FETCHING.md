# OpenRouter Model Fetching in SCL-OpenRouter

This document explains how the enhanced model fetching functionality works in the SCL-OpenRouter project.

## Overview

The project now includes an improved system for fetching, caching, and selecting models from the OpenRouter API. The primary enhancements include:

1. **Automatic model refreshing** from the OpenRouter API
2. **Model caching** to reduce API calls
3. **Task-specific model selection** to optimize performance
4. **User preference saving** for preferred models
5. **Command-line interface** for managing models

## How Model Fetching Works

### API Request Process

1. The system attempts to fetch models from the OpenRouter API on startup
2. Models are organized by provider (Google, Meta, etc.)
3. Free models are marked with `:free` suffix
4. If the API is unavailable, the system falls back to hardcoded models
5. Successfully fetched models are cached to disk

### Model Caching

- Models are cached in a `model_cache.json` file at the project root
- Cache includes timestamp information for expiration checking
- Default cache expiry is 1 hour (configurable via environment variable)
- Cache can be manually refreshed via UI button or command-line tool

### Task-Specific Model Selection

The system supports different model preferences for different task types:

- **General**: General purpose conversation
- **Coding**: Code generation and analysis
- **Vision**: Image understanding
- **Reasoning**: Complex reasoning and problem solving

Each task type has a list of preferred providers that are tried in order when selecting the best model.

## Configuration Options

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `OPENROUTER_API_URL`: API endpoint URL (default: "https://openrouter.ai/api/v1/models")
- `INCLUDE_PAID_MODELS`: Whether to include non-free models (default: "false")
- `MODEL_CACHE_EXPIRY`: Time in seconds before cache expires (default: 3600)
- `DISABLE_MODEL_CACHE`: Set to "true" to disable model caching completely (default: "false")

### User Preferences

User model preferences are stored in `user_preferences.json` in the project root. This file stores:

- Preferred model for each task type
- Other user settings

## Command-Line Interface

The project includes a command-line tool (`manage_models.py`) for managing models:

```bash
# List all available models
./manage_models.py --list

# List only free models
./manage_models.py --list --free-only

# List models for a specific provider
./manage_models.py --list --provider Meta

# Force refresh the model cache
./manage_models.py --refresh

# Check API connection status
./manage_models.py --check-api

# Show information about default model
./manage_models.py --info
```

## API Functions

### Model Configuration Functions

- `fetch_models_from_api()`: Fetch models from the OpenRouter API
- `refresh_model_cache()`: Force a refresh of the model cache
- `display_available_models()`: Display a formatted list of available models

### Utility Functions

- `get_best_model_for_task(task_type)`: Get the best model for a specific task
- `list_models_for_task(task_type)`: List all suitable models for a task
- `save_user_model_preference(task_type, model_id)`: Save user model preference
- `get_user_model_preference(task_type)`: Get user saved preference for a task

## Usage in Code

```python
from src.agents.openrouter_agent import OpenRouterAgent
from src.utils import get_best_model_for_task

# Get best model for a coding task
model_name, model_id = get_best_model_for_task("coding")

# Create agent with task-specific optimization
agent = OpenRouterAgent(task_type="coding")

# Or specify a model directly
agent = OpenRouterAgent(model_id="meta-llama/llama-3.3-70b-instruct:free")

# Generate response
response = agent.generate_response("Write a Python function to calculate Fibonacci numbers")
```

## Troubleshooting

### API Key Issues

If you encounter errors related to the API key:

1. Ensure your `OPENROUTER_API_KEY` is set correctly
2. Verify the API key is valid and active
3. Check your account balance on OpenRouter

### Empty Model List

If no models are returned:

1. Try refreshing the cache with `./manage_models.py --refresh`
2. Check your internet connection
3. Verify that the OpenRouter API is operational

### Error Loading Models

If you see errors about loading models:

1. Delete the `model_cache.json` file to force a fresh fetch
2. Set `DISABLE_MODEL_CACHE=true` in your environment to bypass caching
3. Check file permissions
4. Verify that the OpenRouter API is returning the expected data structure

### Empty Model Cache

If the application keeps creating an empty model cache file:

1. Delete the existing `model_cache.json` file
2. Disable caching with `export DISABLE_MODEL_CACHE=true`
3. Check your OpenRouter API key permissions
4. Check your network connection to the OpenRouter API
5. Add `--debug` flag to the manage_models.py command to see detailed API responses
