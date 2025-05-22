"""
Usage tracking utilities for the SilentCodingLegend AI agent.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logger = logging.getLogger(__name__)

class UsageTracker:
    """
    Class for tracking OpenRouter API usage.
    """
    
    def __init__(self):
        """
        Initialize the usage tracker.
        """
        self.usage_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "usage_logs"
        self.usage_dir.mkdir(exist_ok=True)
        
        self.current_month = datetime.now().strftime("%Y-%m")
        self.usage_file = self.usage_dir / f"usage_{self.current_month}.json"
        self.current_usage = self._load_current_usage()
    
    def _load_current_usage(self):
        """
        Load the current usage data from file.
        """
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading usage data: {e}")
        
        # Return empty usage data if file doesn't exist or has an error
        return {
            "month": self.current_month,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "models": {},
            "conversations": 0,
            "requests": 0
        }
    
    def track_request(self, model_id, prompt_tokens, completion_tokens, task_type="general"):
        """
        Track a request to the OpenRouter API.
        
        Args:
            model_id (str): The model ID
            prompt_tokens (int): Number of prompt tokens
            completion_tokens (int): Number of completion tokens
            task_type (str): The type of task ('general', 'coding', etc.)
        """
        # Calculate total tokens
        total_tokens = prompt_tokens + completion_tokens
        
        # Update total usage
        self.current_usage["total_tokens"] += total_tokens
        self.current_usage["prompt_tokens"] += prompt_tokens
        self.current_usage["completion_tokens"] += completion_tokens
        self.current_usage["requests"] += 1
        
        # Update model-specific usage
        if model_id not in self.current_usage["models"]:
            self.current_usage["models"][model_id] = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "requests": 0,
                "task_types": {}
            }
        
        # Update model usage
        self.current_usage["models"][model_id]["total_tokens"] += total_tokens
        self.current_usage["models"][model_id]["prompt_tokens"] += prompt_tokens
        self.current_usage["models"][model_id]["completion_tokens"] += completion_tokens
        self.current_usage["models"][model_id]["requests"] += 1
        
        # Update task type usage
        if task_type not in self.current_usage["models"][model_id]["task_types"]:
            self.current_usage["models"][model_id]["task_types"][task_type] = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "requests": 0
            }
        
        task_usage = self.current_usage["models"][model_id]["task_types"][task_type]
        task_usage["total_tokens"] += total_tokens
        task_usage["prompt_tokens"] += prompt_tokens
        task_usage["completion_tokens"] += completion_tokens
        task_usage["requests"] += 1
        
        # Save the updated usage data
        self._save_usage()
    
    def track_conversation(self):
        """
        Track a new conversation.
        """
        self.current_usage["conversations"] += 1
        self._save_usage()
    
    def _save_usage(self):
        """
        Save the usage data to file.
        """
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.current_usage, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving usage data: {e}")
    
    def get_monthly_summary(self):
        """
        Get a summary of usage for the current month.
        
        Returns:
            dict: Usage summary
        """
        return self.current_usage
    
    def get_usage_by_model(self, model_id):
        """
        Get usage data for a specific model.
        
        Args:
            model_id (str): The model ID
            
        Returns:
            dict or None: Model usage data or None if not found
        """
        return self.current_usage["models"].get(model_id)
    
    def get_estimated_cost(self, price_per_1k_prompt=0.002, price_per_1k_completion=0.002):
        """
        Get an estimated cost based on token usage.
        
        Args:
            price_per_1k_prompt (float): Price per 1,000 prompt tokens
            price_per_1k_completion (float): Price per 1,000 completion tokens
            
        Returns:
            float: Estimated cost in USD
        """
        prompt_cost = (self.current_usage["prompt_tokens"] / 1000) * price_per_1k_prompt
        completion_cost = (self.current_usage["completion_tokens"] / 1000) * price_per_1k_completion
        return prompt_cost + completion_cost
