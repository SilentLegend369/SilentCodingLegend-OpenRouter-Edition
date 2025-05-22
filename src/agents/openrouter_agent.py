"""
OpenRouter agent module for the SilentCodingLegend AI agent.
"""

import os
import requests
import json
import logging
from dotenv import load_dotenv
from src.model_config import DEFAULT_MODEL_ID, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from src.utils import get_best_model_for_task, get_user_model_preference

# Load environment variables from .env file
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class OpenRouterAgent:
    """
    Agent class for interacting with the OpenRouter API.
    """
    
    def __init__(self, model_id=None, temperature=None, max_tokens=None, task_type="general"):
        """
        Initialize the OpenRouter agent.
        
        Args:
            model_id (str): The model ID to use for the agent.
            temperature (float): The temperature to use for the agent.
            max_tokens (int): The maximum number of tokens to generate.
            task_type (str): Type of task ('general', 'coding', 'vision', 'reasoning')
        """
        # Set API key from environment
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable.")
        
        # Set API endpoint
        self.api_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        # Set model parameters
        self.task_type = task_type
        
        # If model_id is not provided, check for user preference or get best model for task
        if not model_id:
            # First check for user preference
            user_preference = get_user_model_preference(task_type)
            if user_preference:
                logger.info(f"Using user preferred model for {task_type}: {user_preference}")
                model_id = user_preference
            else:
                # Get best model for this task
                _, best_model_id = get_best_model_for_task(task_type)
                if best_model_id:
                    logger.info(f"Selected best model for {task_type}: {best_model_id}")
                    model_id = best_model_id
                else:
                    # Fall back to default model
                    logger.warning(f"Could not find suitable model for {task_type}. Using default.")
                    model_id = DEFAULT_MODEL_ID
        
        self.model_id = model_id
        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        
        logger.info(f"Initialized OpenRouterAgent with model: {self.model_id}")
        
    def generate_response(self, prompt, system_prompt=None, temperature=None, max_tokens=None):
        """
        Generate a response from the model.
        
        Args:
            prompt (str): The prompt to generate a response for.
            system_prompt (str, optional): The system prompt to provide context.
            temperature (float, optional): The temperature to use for generation.
            max_tokens (int, optional): The maximum number of tokens to generate.
            
        Returns:
            str: The generated response.
        """
        # Use default values if not provided
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Prepare messages
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Prepare the request payload
            payload = {
                "model": self.model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://silentcodinglegend.ai"),  # Optional
                "X-Title": os.getenv("OPENROUTER_APP_TITLE", "SilentCodingLegend AI Agent")  # Optional
            }
            
            logger.debug(f"Making request to OpenRouter API with model: {self.model_id}")
            
            # Make the API request
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=60  # Add timeout to prevent hanging requests
            )
            
            # Parse the response
            if response.status_code == 200:
                response_json = response.json()
                
                # Log usage information if available
                if "usage" in response_json:
                    usage = response_json["usage"]
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                    
                    logger.debug(f"Token usage - Prompt: {prompt_tokens}, " +
                                f"Completion: {completion_tokens}, " +
                                f"Total: {total_tokens}")
                    
                    # Track usage if tracking is enabled
                    try:
                        from src.usage_tracker import UsageTracker
                        tracker = UsageTracker()
                        tracker.track_request(
                            model_id=self.model_id,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            task_type=self.task_type
                        )
                    except Exception as e:
                        logger.warning(f"Failed to track usage: {e}")
                
                return response_json["choices"][0]["message"]["content"]
            else:
                error_msg = f"Error: API returned status code {response.status_code}. {response.text}"
                logger.error(error_msg)
                return error_msg
                
        except requests.exceptions.Timeout:
            error_msg = "Error: Request timed out. The OpenRouter API took too long to respond."
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            # Handle errors
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return error_msg
            
    def generate_vision_response(self, prompt, image_base64, system_prompt=None, temperature=None, max_tokens=None):
        """
        Generate a response from the model with an image.
        
        Args:
            prompt (str): The prompt to generate a response for.
            image_base64 (str): The base64-encoded image.
            system_prompt (str, optional): The system prompt to provide context.
            temperature (float, optional): The temperature to use for generation.
            max_tokens (int, optional): The maximum number of tokens to generate.
            
        Returns:
            str: The generated response.
        """
        # Use default values if not provided
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Prepare messages
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Add user message with text and image
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        })
        
        try:
            # Prepare the request payload
            payload = {
                "model": self.model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://silentcodinglegend.ai"),  # Optional
                "X-Title": os.getenv("OPENROUTER_APP_TITLE", "SilentCodingLegend AI Agent")  # Optional
            }
            
            # Make the API request
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            
            # Parse the response
            if response.status_code == 200:
                response_json = response.json()
                return response_json["choices"][0]["message"]["content"]
            else:
                return f"Error: API returned status code {response.status_code}. {response.text}"
                
        except Exception as e:
            # Handle errors
            return f"Error generating vision response: {str(e)}"
            
    def get_model_info(self):
        """
        Get information about the current model.
        
        Returns:
            dict: A dictionary containing model information.
        """
        return {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }