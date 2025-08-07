import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class Config:
    """Central configuration for agentic workflows"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # Cost Management
    ENABLE_COST_TRACKING = os.getenv("ENABLE_COST_TRACKING", "True").lower() == "true"
    MAX_DAILY_COST = float(os.getenv("MAX_DAILY_COST", "10.00"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
    
    @classmethod
    def get_model_client(cls):
        """Get AutoGen v0.7.1 model client configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        
        return OpenAIChatCompletionClient(
            model=cls.OPENAI_MODEL,
            api_key=cls.OPENAI_API_KEY,
            base_url=cls.OPENAI_BASE_URL,
            timeout=120.0,
            temperature=0.1,  # Low temperature for consistent behavior
        )
    
    @classmethod 
    def get_llm_config(cls) -> Dict[str, Any]:
        """Legacy method for backward compatibility - use get_model_client() instead"""
        import warnings
        warnings.warn("get_llm_config() is deprecated, use get_model_client() instead", DeprecationWarning)
        
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
            
        return {
            "config_list": [
                {
                    "model": cls.OPENAI_MODEL,
                    "api_key": cls.OPENAI_API_KEY,
                    "base_url": cls.OPENAI_BASE_URL,
                    "api_type": "openai",
                }
            ],
            "timeout": 120,
            "cache_seed": 42,
            "temperature": 0.1,
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        
        if missing_vars:
            print(f"Missing required environment variables: {missing_vars}")
            print("Please copy .env.example to .env and fill in your values")
            return False
        
        return True