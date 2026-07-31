"""
optillm_client
==============
A drop-in Python client for the OptiLLM gateway.

Wraps the standard openai.OpenAI client — all existing OpenAI SDK calls work
identically with zero changes. Every response is enriched with an
`optillm_metadata` attribute containing cache hit status, savings, and routing info.

Usage:
    from optillm_client import OptiLLM

    client = OptiLLM(
        api_key="sk-...",                        # Your OpenAI key (passed through)
        gateway_url="http://localhost:8000",      # OptiLLM gateway
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the return policy?"}]
    )

    print(response.choices[0].message.content)
    print(response.optillm_metadata.cache_hit)     # True / False
    print(response.optillm_metadata.savings_usd)   # e.g. 0.000420
    print(response.optillm_metadata.latency_ms)    # e.g. 12

Bypass flags (per-request):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[...],
        extra_body={
            "optillm": {
                "bypass_cache": True,
                "bypass_compression": True,
                "bypass_routing": False,
            }
        }
    )
"""

from optillm_client.client import OptiLLM
from optillm_client.types import OptiLLMMetadata

__all__ = ["OptiLLM", "OptiLLMMetadata"]
__version__ = "0.1.0"
