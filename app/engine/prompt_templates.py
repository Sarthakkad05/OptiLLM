"""
Prompt Template Engine & Registry.
Uses LangChain's ChatPromptTemplate for template versioning, variable injection,
and message rendering into standard OpenAI chat formats.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from app.services.token_counter import count_tokens_in_messages

logger = logging.getLogger("optillm.engine.prompt_templates")


class PromptTemplateRecord:
    """Stores metadata, version history info, and compiled ChatPromptTemplate instance."""

    def __init__(
        self,
        name: str,
        version: str,
        system_template: str,
        user_template: str,
        description: Optional[str] = None,
        commit_message: Optional[str] = None,
        is_active: bool = True,
    ):
        self.name = name
        self.version = version
        self.system_template = system_template
        self.user_template = user_template
        self.description = description or f"{name} template version {version}"
        self.commit_message = commit_message or "Version created"
        self.created_at = time.time()
        self.is_active = is_active

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
    """Registry managing versioned prompt templates with Git-like lifecycle."""

    def __init__(self):
        self._registry: Dict[Tuple[str, str], PromptTemplateRecord] = {}
        self._active_versions: Dict[str, str] = {}
        self._init_default_templates()

    def register(
        self,
        name: str,
        version: str,
        system_template: str,
        user_template: str,
        description: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> PromptTemplateRecord:
        """Register or update a versioned prompt template."""
        key = (name.lower(), version.lower())
        record = PromptTemplateRecord(
            name=name,
            version=version,
            system_template=system_template,
            user_template=user_template,
            description=description,
            commit_message=commit_message or f"Registered {version}",
            is_active=True,
        )
        # Mark previous versions as not active
        for (k_name, k_ver), rec in self._registry.items():
            if k_name == name.lower():
                rec.is_active = False

        self._registry[key] = record
        self._active_versions[name.lower()] = version.lower()
        logger.info("Registered prompt template: %s:%s", name, version)
        return record

    def create_version(
        self,
        name: str,
        system_template: str,
        user_template: str,
        description: Optional[str] = None,
        commit_message: Optional[str] = None,
        version: Optional[str] = None,
    ) -> PromptTemplateRecord:
        """Create a new version of an existing template (auto-increments if version is None)."""
        existing = self.get_history(name)
        if not version:
            next_num = len(existing) + 1
            version = f"v{next_num}"

        return self.register(
            name=name,
            version=version,
            system_template=system_template,
            user_template=user_template,
            description=description,
            commit_message=commit_message,
        )

    def get(self, name: str, version: Optional[str] = None) -> Optional[PromptTemplateRecord]:
        """Fetch registered prompt template by name and version. Defaults to active version."""
        if not version:
            version = self._active_versions.get(name.lower(), "v1")
        return self._registry.get((name.lower(), version.lower()))

    def get_history(self, name: str) -> List[Dict[str, Any]]:
        """Return full version history for a prompt template."""
        versions = []
        for (k_name, k_ver), record in self._registry.items():
            if k_name == name.lower():
                versions.append(
                    {
                        "name": record.name,
                        "version": record.version,
                        "description": record.description,
                        "commit_message": record.commit_message,
                        "is_active": record.is_active,
                        "created_at": record.created_at,
                        "input_variables": record.input_variables,
                        "system_template": record.system_template,
                        "user_template": record.user_template,
                    }
                )
        versions.sort(key=lambda x: x["created_at"])
        return versions

    def rollback(self, name: str, target_version: str) -> PromptTemplateRecord:
        """Rollback active version of a template to a previous version."""
        target_record = self.get(name, target_version)
        if not target_record:
            raise ValueError(f"Target version '{name}:{target_version}' does not exist.")

        # Update active flags
        for (k_name, k_ver), rec in self._registry.items():
            if k_name == name.lower():
                rec.is_active = (k_ver == target_version.lower())

        self._active_versions[name.lower()] = target_version.lower()
        logger.info("Rolled back template '%s' to active version '%s'", name, target_version)
        return target_record

    def compare_versions(
        self,
        name: str,
        version_a: str,
        version_b: str,
        variables: Dict[str, Any],
        model: str = "gpt-4o",
    ) -> Dict[str, Any]:
        """Compare two versions of a prompt template rendered with sample variables."""
        rec_a = self.get(name, version_a)
        rec_b = self.get(name, version_b)

        if not rec_a:
            raise ValueError(f"Version '{name}:{version_a}' not found.")
        if not rec_b:
            raise ValueError(f"Version '{name}:{version_b}' not found.")

        msgs_a = rec_a.render(variables)
        msgs_b = rec_b.render(variables)

        tokens_a = count_tokens_in_messages(msgs_a, model)
        tokens_b = count_tokens_in_messages(msgs_b, model)

        return {
            "template": name,
            "version_a": {
                "version": rec_a.version,
                "description": rec_a.description,
                "messages": msgs_a,
                "token_count": tokens_a,
            },
            "version_b": {
                "version": rec_b.version,
                "description": rec_b.description,
                "messages": msgs_b,
                "token_count": tokens_b,
            },
            "token_difference": tokens_b - tokens_a,
            "token_change_percent": (
                round(((tokens_b - tokens_a) / max(tokens_a, 1)) * 100, 1)
            ),
        }

    def render(
        self, name: str, version: Optional[str], variables: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Renders prompt template into OpenAI message format."""
        record = self.get(name, version)
        if not record:
            ver_str = version or self._active_versions.get(name.lower(), "default")
            raise ValueError(f"Prompt template '{name}:{ver_str}' not found.")
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
                    "commit_message": record.commit_message,
                    "is_active": record.is_active,
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
