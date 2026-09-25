"""
AWS Bedrock Provider Adapter.
Routes requests to AWS Bedrock hosted models (Claude, Llama, Titan).
Uses boto3 for AWS auth — requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION.
"""

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.core.config import settings
from app.providers.base import BaseProvider

logger = logging.getLogger("optillm.provider.bedrock")

_PLACEHOLDER_KEYS = {"", None}

# Maps public model names to Bedrock model IDs
_BEDROCK_MODEL_IDS = {
    "claude-3-5-sonnet-bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-haiku-bedrock": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-3-opus-bedrock": "anthropic.claude-3-opus-20240229-v1:0",
    "llama3-70b-bedrock": "meta.llama3-70b-instruct-v1:0",
    "llama3-8b-bedrock": "meta.llama3-8b-instruct-v1:0",
    "mistral-large-bedrock": "mistral.mistral-large-2402-v1:0",
    "mistral-7b-bedrock": "mistral.mistral-7b-instruct-v0:2",
}


def _is_bedrock_available() -> bool:
    return (
        settings.AWS_ACCESS_KEY_ID not in _PLACEHOLDER_KEYS
        and settings.AWS_SECRET_ACCESS_KEY not in _PLACEHOLDER_KEYS
    )


def _get_boto3_client():
    """Creates a boto3 bedrock-runtime client."""
    try:
        import boto3
        return boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
    except ImportError:
        raise RuntimeError(
            "boto3 is required for AWS Bedrock support. Install it: pip install boto3"
        )


def _build_anthropic_payload(messages: List[Dict], temperature: float, max_tokens: Optional[int]) -> Dict:
    """Builds Anthropic Claude payload for Bedrock."""
    system = ""
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system = m.get("content", "")
        else:
            chat_messages.append({"role": m["role"], "content": m.get("content", "")})
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens or 1024,
        "temperature": temperature,
        "messages": chat_messages,
    }
    if system:
        payload["system"] = system
    return payload


def _build_meta_payload(messages: List[Dict], temperature: float, max_tokens: Optional[int]) -> Dict:
    """Builds Meta Llama payload for Bedrock."""
    prompt_parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>")
    prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>")
    return {
        "prompt": "<|begin_of_text|>" + "".join(prompt_parts),
        "max_gen_len": max_tokens or 512,
        "temperature": temperature,
    }


class BedrockProvider(BaseProvider):
    """AWS Bedrock Provider Adapter."""

    @property
    def name(self) -> str:
        return "bedrock"

    def is_available(self) -> bool:
        return _is_bedrock_available()

    def supports_model(self, model: str) -> bool:
        return model.lower() in _BEDROCK_MODEL_IDS

    async def call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise ValueError("AWS credentials not set for Bedrock. Configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")

        bedrock_model_id = _BEDROCK_MODEL_IDS.get(model)
        if not bedrock_model_id:
            raise ValueError(f"Unknown Bedrock model: {model}. Supported: {list(_BEDROCK_MODEL_IDS.keys())}")

        import asyncio

        is_claude = "anthropic" in bedrock_model_id
        is_llama = "llama" in bedrock_model_id or "meta" in bedrock_model_id

        if is_claude:
            payload = _build_anthropic_payload(messages, temperature, max_tokens)
        elif is_llama:
            payload = _build_meta_payload(messages, temperature, max_tokens)
        else:
            payload = _build_anthropic_payload(messages, temperature, max_tokens)

        start = time.time()

        def _invoke():
            client = _get_boto3_client()
            return client.invoke_model(
                modelId=bedrock_model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )

        # Run boto3 (sync) in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _invoke)

        elapsed_ms = int((time.time() - start) * 1000)
        body = json.loads(response["body"].read())

        if is_claude:
            content = body["content"][0]["text"]
            tokens_in = body.get("usage", {}).get("input_tokens", 0)
            tokens_out = body.get("usage", {}).get("output_tokens", 0)
        elif is_llama:
            content = body.get("generation", "")
            tokens_in = body.get("prompt_token_count", 0)
            tokens_out = body.get("generation_token_count", 0)
        else:
            content = str(body)
            tokens_in = tokens_out = 0

        logger.info("Bedrock response | model=%s | latency=%dms", bedrock_model_id, elapsed_ms)

        return {
            "id": f"bedrock-{uuid.uuid4().hex[:8]}",
            "content": content,
            "model": model,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "finish_reason": "stop",
            "latency_ms": elapsed_ms,
            "provider": "bedrock",
        }

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        # Bedrock streaming requires response_stream — for simplicity, call non-streaming and yield result
        result = await self.call(messages, model, temperature, max_tokens)
        yield result["content"]


bedrock_provider = BedrockProvider()
