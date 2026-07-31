"""
ShopEasy Support Bot
Powered by OptiLLM — demonstrates live semantic caching, context compression,
and model routing in a real customer support conversation.
"""

import os
import sys
import time
import requests
import streamlit as st

# Project root on path
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

GATEWAY_URL = os.environ.get("OPTILLM_URL", "http://localhost:8000")
MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are Alex, a friendly and concise customer support agent for ShopEasy, an online retail platform.

ShopEasy policies:
- 30-day hassle-free returns (original condition, tags attached)
- Free return shipping with prepaid label
- Refunds processed in 3–5 business days after receipt
- Cancel orders within 1 hour of placement
- Orders ship same day if placed before 2 PM EST (Mon–Fri)
- Standard shipping free on orders over $35
- Express shipping: $8.99 | Overnight: $19.99
- Price match within 7 days vs Amazon, Walmart, Target
- ShopEasy Plus membership: $9.99/month — free express shipping + 10% cashback

Keep replies brief (2–4 sentences). Be warm but efficient. If you don't know something, say so and offer to escalate."""


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopEasy Support — Powered by OptiLLM",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0a0a; color: #e2e2e2; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0; max-width: 100%; }

/* Header */
.app-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 32px; border-bottom: 1px solid #1a1a1a;
    background: #0d0d0d;
}
.brand { font-size: 1rem; font-weight: 600; color: #fff; }
.brand-sub { font-size: 0.75rem; color: #555; margin-top: 2px; }
.powered-by {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.72rem; color: #444;
    border: 1px solid #1e1e1e; border-radius: 5px; padding: 4px 10px;
}
.dot-on { width: 6px; height: 6px; background: #22c55e; border-radius: 50%; }
.dot-off { width: 6px; height: 6px; background: #ef4444; border-radius: 50%; }

/* Chat area */
.chat-wrap {
    height: calc(100vh - 180px);
    overflow-y: auto;
    padding: 24px 24px 0 24px;
}

/* Metrics panel */
.metrics-panel {
    background: #0d0d0d;
    border-left: 1px solid #1a1a1a;
    padding: 20px 20px;
    height: 100vh;
    overflow-y: auto;
}
.metrics-title {
    font-size: 0.68rem; font-weight: 500; color: #444;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 16px;
}

/* Request card */
.req-card {
    background: #111; border: 1px solid #1e1e1e; border-radius: 7px;
    padding: 12px 14px; margin-bottom: 10px; font-size: 0.78rem;
}
.req-card-hit { border-color: #14532d; }
.req-prompt { color: #555; margin-bottom: 8px; font-style: italic; }
.req-badges { margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 4px; }
.badge {
    font-size: 0.65rem; font-weight: 500; padding: 2px 6px;
    border-radius: 3px; font-family: 'JetBrains Mono', monospace;
}
.b-hit    { background: #052e16; color: #4ade80; border: 1px solid #14532d; }
.b-miss   { background: #1c0808; color: #f87171; border: 1px solid #3b0e0e; }
.b-comp   { background: #1a1000; color: #fbbf24; border: 1px solid #431407; }
.b-routed { background: #0f0a1e; color: #a78bfa; border: 1px solid #2e1065; }
.req-meta { display: flex; gap: 14px; flex-wrap: wrap; color: #555; }
.req-meta .val { color: #c0c0c0; font-weight: 500; }
.req-meta .val-green { color: #22c55e; font-weight: 500; }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }

/* Session total card */
.total-card {
    background: #0d0d0d; border: 1px solid #1a1a1a; border-radius: 7px;
    padding: 14px; margin-top: 16px;
}
.total-row {
    display: flex; justify-content: space-between; padding: 5px 0;
    border-bottom: 1px solid #141414; font-size: 0.78rem;
}
.total-row:last-child { border-bottom: none; }
.total-key { color: #444; }

/* Chat message styles */
.msg-user {
    background: #141414; border: 1px solid #1e1e1e; border-radius: 10px 10px 4px 10px;
    padding: 10px 14px; margin: 8px 0 8px 20%; font-size: 0.85rem; color: #ddd;
    text-align: right;
}
.msg-bot {
    background: #0f0f0f; border: 1px solid #1a1a1a; border-radius: 10px 10px 10px 4px;
    padding: 10px 14px; margin: 8px 20% 8px 0; font-size: 0.85rem; color: #c0c0c0;
    line-height: 1.6;
}
.msg-name { font-size: 0.65rem; color: #555; margin-bottom: 4px; font-weight: 500; }

/* Input */
.stChatInput { border-top: 1px solid #1a1a1a; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_gateway() -> bool:
    try:
        return requests.get(f"{GATEWAY_URL}/health", timeout=2).status_code == 200
    except Exception:
        return False


def _chat(user_message: str) -> dict:
    """Send a message through OptiLLM gateway and return parsed response."""
    history = st.session_state.messages
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *[{"role": m["role"], "content": m["content"]} for m in history],
            {"role": "user", "content": user_message},
        ],
    }
    t0 = time.time()
    resp = requests.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload, timeout=30)
    resp.raise_for_status()
    elapsed_ms = int((time.time() - t0) * 1000)

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    meta = data.get("optillm_metadata", {})

    # Fallback if gateway doesn't return metadata yet
    if not meta:
        meta = {
            "cache_hit": False, "routed": False, "compressed": False,
            "model_used": MODEL, "model_requested": MODEL,
            "cost_usd": 0.0, "savings_usd": 0.0, "latency_ms": elapsed_ms,
            "tokens_saved": 0,
        }

    return {"content": content, "meta": meta}


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []       # Chat history (role + content)
if "request_log" not in st.session_state:
    st.session_state.request_log = []    # OptiLLM metadata per request


# ── Layout ─────────────────────────────────────────────────────────────────────
gateway_up = _check_gateway()

# Header
dot = "dot-on" if gateway_up else "dot-off"
st.markdown(f"""
<div class="app-header">
  <div>
    <div class="brand">🛍️ ShopEasy Support</div>
    <div class="brand-sub">Powered by OptiLLM</div>
  </div>
  <div class="powered-by">
    <span class="{dot}"></span>
    <span>OptiLLM :{GATEWAY_URL.split(':')[-1]}</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not gateway_up:
    st.error("⚠️ OptiLLM gateway is offline. Start it with: `uvicorn app.main:app --reload --port 8000`")
    st.stop()

# Two columns: Chat | Metrics
col_chat, col_metrics = st.columns([1.6, 1], gap="medium")

with col_chat:
    # Render existing messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user"><div class="msg-name">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-bot"><div class="msg-name">Alex · ShopEasy Support</div>{msg["content"]}</div>', unsafe_allow_html=True)

    # Suggested questions (only on first load)
    if not st.session_state.messages:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.72rem; color:#333; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;">Try asking:</div>', unsafe_allow_html=True)
        suggestions = [
            "What is your return policy?",
            "How long does a refund take?",
            "What payment methods do you accept?",
            "How do I track my order?",
            "Can I cancel my order?",
        ]
        sug_cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            with sug_cols[i]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state._prefill = sug
                    st.rerun()

    # Chat input
    user_input = st.chat_input("Ask anything about your ShopEasy order…")

    # Handle prefill from suggestion buttons
    if hasattr(st.session_state, "_prefill") and st.session_state._prefill:
        user_input = st.session_state._prefill
        del st.session_state._prefill

    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Call OptiLLM
        with st.spinner(""):
            result = _chat(user_input)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": result["content"]})

        # Log OptiLLM metadata
        st.session_state.request_log.insert(0, {
            "prompt": user_input,
            "meta": result["meta"],
        })

        st.rerun()


with col_metrics:
    st.markdown('<div class="metrics-title">OptiLLM — Live Metrics</div>', unsafe_allow_html=True)

    if not st.session_state.request_log:
        st.markdown('<div style="color:#333; font-size:0.8rem; margin-top:20px;">Send a message to see optimization metrics.</div>', unsafe_allow_html=True)
    else:
        # Cumulative totals
        all_meta = [r["meta"] for r in st.session_state.request_log]
        total_cost   = sum(m.get("cost_usd", 0) for m in all_meta)
        total_saved  = sum(m.get("savings_usd", 0) for m in all_meta)
        total_hits   = sum(1 for m in all_meta if m.get("cache_hit"))
        total_routed = sum(1 for m in all_meta if m.get("routed"))
        total_comp   = sum(1 for m in all_meta if m.get("compressed"))
        hit_rate     = total_hits / len(all_meta) * 100 if all_meta else 0

        total_html = f"""
        <div class="total-card">
            <div style="font-size:0.68rem; color:#444; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;">
                Session Summary
            </div>
            <div class="total-row">
                <span class="total-key">Requests</span>
                <span class="val">{len(all_meta)}</span>
            </div>
            <div class="total-row">
                <span class="total-key">Cache hit rate</span>
                <span class="val {'val-green' if hit_rate >= 40 else 'val'}">{hit_rate:.0f}%</span>
            </div>
            <div class="total-row">
                <span class="total-key">Total cost</span>
                <span class="mono val">${total_cost:.6f}</span>
            </div>
            <div class="total-row">
                <span class="total-key">Total saved</span>
                <span class="mono val-green">${total_saved:.6f}</span>
            </div>
            <div class="total-row">
                <span class="total-key">Cache hits</span>
                <span class="val">{total_hits}</span>
            </div>
            <div class="total-row">
                <span class="total-key">Routed</span>
                <span class="val">{total_routed}</span>
            </div>
            <div class="total-row">
                <span class="total-key">Compressed</span>
                <span class="val">{total_comp}</span>
            </div>
        </div>
        """
        st.markdown(total_html, unsafe_allow_html=True)

        # Per-request log
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="metrics-title">Request Log</div>', unsafe_allow_html=True)

        for req in st.session_state.request_log:
            m = req["meta"]
            hit       = m.get("cache_hit", False)
            routed    = m.get("routed", False)
            compressed = m.get("compressed", False)
            model_req = m.get("model_requested", MODEL)
            model_used = m.get("model_used", MODEL)
            latency   = m.get("latency_ms", 0)
            cost      = m.get("cost_usd", 0.0)
            saved     = m.get("savings_usd", 0.0)

            card_cls = "req-card req-card-hit" if hit else "req-card"

            badges = ""
            if hit:
                badges += '<span class="badge b-hit">⚡ CACHE HIT</span>'
            else:
                badges += '<span class="badge b-miss">MISS</span>'
            if compressed:
                badges += '<span class="badge b-comp">COMPRESSED</span>'
            if routed:
                badges += '<span class="badge b-routed">ROUTED</span>'

            model_display = (
                f'<span class="mono" style="color:#555;">{model_req}</span>'
                f' <span style="color:#333;">→</span>'
                f' <span class="mono" style="color:#a78bfa;">{model_used}</span>'
                if routed and model_req != model_used
                else f'<span class="mono" style="color:#555;">{model_used}</span>'
            )

            prompt_short = req["prompt"][:55] + ("…" if len(req["prompt"]) > 55 else "")

            saved_html = (
                f'<span class="val-green">${saved:.6f} saved</span>'
                if saved > 0 else
                f'<span style="color:#333;">$0 saved</span>'
            )

            st.markdown(f"""
            <div class="{card_cls}">
                <div class="req-prompt">"{prompt_short}"</div>
                <div class="req-badges">{badges}</div>
                <div class="req-meta">
                    <span>model &nbsp;{model_display}</span>
                    <span>⏱ <span class="val">{latency:,}ms</span></span>
                    <span>💸 <span class="mono val">${cost:.6f}</span></span>
                    <span>{saved_html}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.messages = []
            st.session_state.request_log = []
            st.rerun()
