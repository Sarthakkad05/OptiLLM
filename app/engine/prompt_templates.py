"""
Prompt Template Engine & Registry.
Uses LangChain's ChatPromptTemplate for template versioning, variable injection,
and message rendering into standard OpenAI chat formats.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger("optillm.engine.prompt_templates")


class PromptTemplateRecord:
    """Stores metadata and compiled ChatPromptTemplate instance."""

    def __init__(
        self,
        name: str,
        version: str,
        system_template: str,
        user_template: str,
        description: Optional[str] = None,
    ):
        self.name = name
        self.version = version
        self.system_template = system_template
        self.user_template = user_template
        self.description = description or f"{name} template version {version}"

        # Compile LangChain ChatPromptTemplate
        messages = [
            ("system", system_template),
            ("user", user_template),
        ]
        self.chat_prompt = ChatPromptTemplate.from_messages(messages)
        self.input_variables = self.chat_prompt.input_variables

    def render(self, variables: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Renders template with variables into OpenAI-compatible messages list:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        formatted = self.chat_prompt.format_prompt(**variables)
        messages = []
        for msg in formatted.to_messages():
            role = "system" if msg.type == "system" else "user"
            messages.append({"role": role, "content": msg.content})
        return messages


class PromptTemplateEngine:
    """Registry managing versioned prompt templates."""

    def __init__(self):
        self._registry: Dict[Tuple[str, str], PromptTemplateRecord] = {}
        self._init_default_templates()

    def register(
        self,
        name: str,
        version: str,
        system_template: str,
        user_template: str,
        description: Optional[str] = None,
    ) -> PromptTemplateRecord:
        """Register or update a versioned prompt template."""
        key = (name.lower(), version.lower())
        record = PromptTemplateRecord(
            name=name,
            version=version,
            system_template=system_template,
            user_template=user_template,
            description=description,
        )
        self._registry[key] = record
        logger.info("Registered prompt template: %s:%s", name, version)
        return record

    def get(self, name: str, version: str = "v1") -> Optional[PromptTemplateRecord]:
        """Fetch registered prompt template by name and version."""
        return self._registry.get((name.lower(), version.lower()))

    def render(
        self, name: str, version: str, variables: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Renders prompt template into OpenAI message format."""
        record = self.get(name, version)
        if not record:
            raise ValueError(f"Prompt template '{name}:{version}' not found.")
        return record.render(variables)

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all registered templates and input variables."""
        result = []
        for (name, version), record in self._registry.items():
            result.append(
                {
                    "name": record.name,
                    "version": record.version,
                    "description": record.description,
                    "input_variables": record.input_variables,
                    "system_template": record.system_template,
                    "user_template": record.user_template,
                }
            )
        return result

    def _init_default_templates(self):
        """Pre-register default prompt templates."""
        self.register(
            name="qa_assistant",
            version="v1",
            system_template="You are a helpful and concise AI assistant specializing in {domain}.",
            user_template="{question}",
            description="General Q&A assistant with domain persona",
        )
        self.register(
            name="summarizer",
            version="v1",
            system_template="Summarize the provided text accurately in {style} format.",
            user_template="Text to summarize:\n{text}",
            description="Text summarization template",
        )
        self.register(
            name="classifier",
            version="v1",
            system_template="Classify the input text into one of these categories: {categories}.",
            user_template="Input Text: {input_text}",
            description="Text classification template",
        )


# Global singleton prompt template engine
prompt_engine = PromptTemplateEngine()
