"""
OptiLLM Demo Scenarios
======================
A library of self-contained demo scenarios used by the Streamlit
Demo Runner page. Each module exposes a `run()` function that:
  1. Calls the OptiLLM gateway via HTTP
  2. Returns a DemoResult containing structured step-level metadata

Import the catalog to get all runnable scenarios:
    from demos import SCENARIO_CATALOG
"""

from demos.base import DemoResult, DemoStep, call_gateway, check_health  # noqa: F401

from demos.cache_miss import run as run_cache_miss
from demos.cache_hit import run as run_cache_hit
from demos.compression_small import run as run_compression_small
from demos.compression_large import run as run_compression_large
from demos.routing_simple import run as run_routing_simple
from demos.routing_complex import run as run_routing_complex
from demos.routing_pair import run as run_routing_pair
from demos.full_pipeline import run as run_full_pipeline
from demos.analytics_summary import run as run_analytics_summary

# Ordered catalog of all available scenarios.
# Each entry: (display_name, icon, description, run_fn)
SCENARIO_CATALOG = [
    {
        "key": "cache_miss",
        "name": "Cache Miss",
        "icon": "❌",
        "category": "Semantic Cache",
        "description": "First-time request — cache is empty, full LLM call is made.",
        "run": run_cache_miss,
    },
    {
        "key": "cache_hit",
        "name": "Cache Hit",
        "icon": "⚡",
        "category": "Semantic Cache",
        "description": "Semantically similar query — served instantly from FAISS cache.",
        "run": run_cache_hit,
    },
    {
        "key": "compression_small",
        "name": "Compression (Moderate Bloat)",
        "icon": "🗜️",
        "category": "Context Compression",
        "description": "A moderately repetitive prompt is trimmed before being sent to the LLM.",
        "run": run_compression_small,
    },
    {
        "key": "compression_large",
        "name": "Compression (Severe Bloat)",
        "icon": "🗜️",
        "category": "Context Compression",
        "description": "A massively bloated prompt (80+ repetitions) — compressor saves hundreds of tokens.",
        "run": run_compression_large,
    },
    {
        "key": "routing_simple",
        "name": "Routing — Simple Query",
        "icon": "🔀",
        "category": "Model Router",
        "description": "Simple math question requested on gpt-4o → downgraded to a cheap model.",
        "run": run_routing_simple,
    },
    {
        "key": "routing_complex",
        "name": "Routing — Complex Query",
        "icon": "🔀",
        "category": "Model Router",
        "description": "Complex coding question stays on the frontier model — router leaves it alone.",
        "run": run_routing_complex,
    },
    {
        "key": "routing_pair",
        "name": "Routing — Simple vs Complex",
        "icon": "🔀",
        "category": "Model Router",
        "description": "Runs both a simple and complex query side-by-side to compare routing decisions.",
        "run": run_routing_pair,
    },
    {
        "key": "full_pipeline",
        "name": "Full Optimization Pipeline",
        "icon": "🚀",
        "category": "End-to-End",
        "description": "Demonstrates all 4 optimizations firing in sequence: cache miss → compress → route → cache hit.",
        "run": run_full_pipeline,
    },
    {
        "key": "analytics_summary",
        "name": "Analytics Snapshot",
        "icon": "📊",
        "category": "Analytics",
        "description": "Fetches and displays current KPIs: total savings, cache rate, token economics.",
        "run": run_analytics_summary,
    },
]
