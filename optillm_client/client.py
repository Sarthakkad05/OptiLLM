"""
OptiLLM Python SDK Client
OpenAI-compatible client library for interacting with OptiLLM Gateway.
"""

from typing import Any, Dict, List, Optional
import httpx


class OptiLLMClient:
    """
    Python SDK client for OptiLLM AI Gateway.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = "sk-optillm-dev-key",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "",
        }

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4o",
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Send chat completion request to OptiLLM Gateway."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(url, json=payload, headers=self._headers)
            res.raise_for_status()
            return res.json()

    def get_analytics(self) -> Dict[str, Any]:
        """Fetch dashboard analytics summary."""
        url = f"{self.base_url}/api/v1/analytics"
        with httpx.Client(timeout=self.timeout) as client:
            res = client.get(url, headers=self._headers)
            res.raise_for_status()
            return res.json()

    def get_providers_status(self) -> Dict[str, Any]:
        """Fetch real-time LLM provider health status."""
        url = f"{self.base_url}/api/v1/providers/status"
        with httpx.Client(timeout=self.timeout) as client:
            res = client.get(url, headers=self._headers)
            res.raise_for_status()
            return res.json()

    def ingest_rag_document(
        self,
        collection_name: str,
        content: str,
        document_name: str = "doc.txt",
    ) -> Dict[str, Any]:
        """Ingest a document into RAG collection."""
        url = f"{self.base_url}/api/v1/rag/ingest"
        payload = {
            "collection_name": collection_name,
            "content": content,
            "document_name": document_name,
        }
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(url, json=payload, headers=self._headers)
            res.raise_for_status()
            return res.json()

    def evaluate_response(
        self,
        messages: List[Dict[str, Any]],
        response_text: str,
    ) -> Dict[str, Any]:
        """Evaluate prompt-response quality across 5 dimensions."""
        url = f"{self.base_url}/api/v1/evaluate"
        payload = {
            "messages": messages,
            "response_text": response_text,
        }
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(url, json=payload, headers=self._headers)
            res.raise_for_status()
            return res.json()
