"""
LLM Configuration - Support for Anthropic, OpenAI, and MiniMax providers.
"""

from typing import Optional
import warnings

from config.settings import settings

# Try to import LLM libraries, provide fallback if not available
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    ChatAnthropic = None

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    ChatOpenAI = None


class MockLLM:
    """
    Mock LLM for when real LLM libraries are not available.
    Returns neutral responses for testing.
    """
    def __init__(self, model_name="mock", temperature=0.1):
        self.model_name = model_name
        self.temperature = temperature
    
    async def ainvoke(self, messages):
        """Return mock response."""
        class MockResponse:
            content = '{"home_win_probability": 0.5, "away_win_probability": 0.5, "confidence": 0.3, "factors": {}}'
        return MockResponse()
    
    def invoke(self, messages):
        """Return mock response (sync)."""
        class MockResponse:
            content = '{"home_win_probability": 0.5, "away_win_probability": 0.5, "confidence": 0.3, "factors": {}}'
        return MockResponse()


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    provider: Optional[str] = None
):
    """
    Get an LLM instance based on the configured or specified provider.
    
    Falls back to MockLLM if libraries not installed.
    
    Args:
        model_name: Model name (e.g., "claude-sonnet-4-20250506" or "gpt-4o")
        temperature: Temperature for generation (default: 0.1)
        provider: Force a specific provider ("anthropic", "openai", or "minimax")
    
    Returns:
        LLM instance or MockLLM if libraries unavailable
    """
    # Check if any LLM library is available
    if not ANTHROPIC_AVAILABLE and not OPENAI_AVAILABLE:
        warnings.warn("No LLM libraries installed. Using MockLLM for testing.")
        return MockLLM(model_name="mock", temperature=temperature)
    
    # Determine provider
    provider = provider or settings.LLM_PROVIDER
    
    # Get model name
    model_name = model_name or settings.MODEL_NAME
    
    if provider == "openai" and OPENAI_AVAILABLE:
        # OpenAI configuration
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            warnings.warn("OPENAI_API_KEY not set, using MockLLM")
            return MockLLM(model_name="mock-openai", temperature=temperature)
        
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
    
    elif provider == "minimax" and OPENAI_AVAILABLE:
        # MiniMax configuration
        api_key = settings.MINIMAX_API_KEY
        if not api_key:
            warnings.warn("MINIMAX_API_KEY not set, using MockLLM")
            return MockLLM(model_name="mock-minimax", temperature=temperature)
        
        # MiniMax uses OpenAI-compatible API format
        return ChatOpenAI(
            model="minimax",
            base_url="https://api.minimax.chat/v1/openai/completions",
            api_key=api_key,
            temperature=temperature
        )
    
    elif ANTHROPIC_AVAILABLE:
        # Default to Anthropic
        api_key = settings.ANTHROPIC_API_KEY
        if not api_key:
            warnings.warn("ANTHROPIC_API_KEY not set, using MockLLM")
            return MockLLM(model_name="mock-anthropic", temperature=temperature)
        
        return ChatAnthropic(
            model=model_name,
            api_key=api_key,
            temperature=temperature
        )
    
    else:
        # Fallback to mock
        warnings.warn("Requested provider not available, using MockLLM")
        return MockLLM(model_name="mock-fallback", temperature=temperature)
