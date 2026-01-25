# config/llm_config.py
"""
LLM Configuration - Support for Anthropic, OpenAI, and MiniMax providers.
"""

from typing import Optional
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from config.settings import settings


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    provider: Optional[str] = None
):
    """
    Get an LLM instance based on the configured or specified provider.
    
    Args:
        model_name: Model name (e.g., "claude-sonnet-4-20250506" or "gpt-4o")
        temperature: Temperature for generation (default: 0.1)
        provider: Force a specific provider ("anthropic", "openai", or "minimax")
    
    Returns:
        LLM instance
    """
    # Determine provider
    provider = provider or settings.LLM_PROVIDER
    
    # Get model name
    model_name = model_name or settings.MODEL_NAME
    
    if provider == "openai":
        # OpenAI configuration
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env file")
        
        # Map model names to OpenAI equivalents
        openai_model_map = {
            "claude-sonnet-4-20250506": "gpt-4o",
            "claude-opus-4-20250506": "gpt-4o",
            "claude-haiku-3-20250506": "gpt-4o-mini",
            "claude-sonnet-3-7-20250506": "gpt-4o",
        }
        
        openai_model = openai_model_map.get(model_name, "gpt-4o")
        
        return ChatOpenAI(
            model=openai_model,
            api_key=api_key,
            temperature=temperature
        )
    
    elif provider == "minimax":
        # MiniMax configuration
        api_key = settings.MINIMAX_API_KEY
        if not api_key:
            raise ValueError("MINIMAX_API_KEY is not set in .env file")
        
        # MiniMax uses OpenAI-compatible API format
        return ChatOpenAI(
            model="minimax",
            base_url="https://api.minimax.chat/v1/openai/completions",
            api_key=api_key,
            temperature=temperature
        )
    
    else:
        # Default to Anthropic
        api_key = settings.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in .env file")
        
        return ChatAnthropic(
            model=model_name,
            api_key=api_key,
            temperature=temperature
        )
