"""
Provider Dispatcher
Determines which provider client to call based on the model name.
All callers use call_provider() — never call openai/gemini clients directly.
"""

import logging
from typing import List, Dict, Any, Optional
from app.providers.openai_client import call_openai
from app.providers.gemini_client import call_gemini

logger = logging.getLogger("optillm.provider.dispatcher")

# Models that route to Gemini
_GEMINI_MODELS = {
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.0-pro",
    "gemini-pro",
}


def _detect_provider(model: str) -> str:
    """Returns 'gemini' or 'openai' based on the model name."""
    model_lower = model.lower()
    if model_lower in _GEMINI_MODELS or model_lower.startswith("gemini"):
        return "gemini"
    return "openai"


async def call_provider(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Dispatches the request to the correct LLM provider.
    Returns a normalised response dict.
    """
    provider = _detect_provider(model)
    logger.info("Dispatching to provider=%s model=%s", provider, model)

    if provider == "gemini":
        return await call_gemini(messages, model, temperature, max_tokens)
    else:
        return await call_openai(messages, model, temperature, max_tokens)
