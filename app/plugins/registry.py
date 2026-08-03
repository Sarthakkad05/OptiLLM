"""
Plugin Registry
Manages plugin registration, runtime enablement, and hook dispatching.
"""

import logging
from typing import Any, Dict, List, Optional
from app.plugins.base import BasePlugin

logger = logging.getLogger("optillm.plugins.registry")


class PluginRegistry:
    """
    Registry for loading and executing gateway plugins.
    """

    def __init__(self):
        self._plugins: Dict[str, BasePlugin] = {}

    def register(self, plugin: BasePlugin, config: Optional[Dict[str, Any]] = None):
        """Register a new plugin instance."""
        plugin.initialize(config)
        self._plugins[plugin.name] = plugin
        logger.info("Registered plugin '%s' (enabled=%s)", plugin.name, plugin.enabled)

    def toggle(self, name: str, enabled: bool) -> bool:
        """Enable or disable plugin at runtime."""
        if name in self._plugins:
            self._plugins[name].enabled = enabled
            logger.info("Plugin '%s' enabled set to %s", name, enabled)
            return True
        return False

    def run_pre_process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pre_process hook across all enabled plugins."""
        curr_payload = payload
        for plugin in self._plugins.values():
            if plugin.enabled:
                try:
                    curr_payload = plugin.pre_process(curr_payload)
                except Exception as e:
                    logger.error("Plugin '%s' pre_process error: %s", plugin.name, e)
                    plugin.on_error(e, {"stage": "pre_process"})
        return curr_payload

    def run_post_process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute post_process hook across all enabled plugins."""
        curr_payload = payload
        for plugin in self._plugins.values():
            if plugin.enabled:
                try:
                    curr_payload = plugin.post_process(curr_payload)
                except Exception as e:
                    logger.error("Plugin '%s' post_process error: %s", plugin.name, e)
                    plugin.on_error(e, {"stage": "post_process"})
        return curr_payload

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List registered plugins and status."""
        return [
            {"name": p.name, "enabled": p.enabled, "class": p.__class__.__name__}
            for p in self._plugins.values()
        ]


_global_plugin_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    global _global_plugin_registry
    if _global_plugin_registry is None:
        _global_plugin_registry = PluginRegistry()
    return _global_plugin_registry
