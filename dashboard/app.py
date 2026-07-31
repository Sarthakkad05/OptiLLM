"""
OptiLLM — Demo Runner
Interactive playground for testing the optimization pipeline.
"""

import os
import sys
import pandas as pd
import streamlit as st

# Allow importing demos/ from the project root
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from demos import SCENARIO_CATALOG
    _demos_available = True
except ImportError:
    _demos_available = False
    SCENARIO_CATALOG = []


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OptiLLM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d0d0d; color: #e2e2e2; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 28px 40px; max-width: 1400px; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }

.top-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 22px; border-bottom: 1px solid #1a1a1a; margin-bottom: 28px;
}
.logo { font-size: 1rem; font-weight: 600; color: #fff; letter-spacing: -0.02em; }
.logo-dot { color: #3b82f6; }
.page-tag { font-size: 0.75rem; color: #444; margin-left: 6px; }

.status-pill {
    display: flex; align-items: center; gap: 6px;
    background: #111; border: 1px solid #1e1e1e; border-radius: 5px;
    padding: 4px 10px; font-size: 0.72rem; color: #555;
}
.dot-green { width: 6px; height: 6px; background: #22c55e; border-radius: 50%; }
.dot-red   { width: 6px; height: 6px; background: #ef4444; border-radius: 50%; }

.section-label {
    font-size: 0.68rem; font-weight: 500; color: #444;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;
}

.cat-label {
    font-size: 0.68rem; color: #333; text-transform: uppercase;
    letter-spacing: 0.08em; padding: 10px 0 2px 0;
}

.stCheckbox label { font-size: 0.82rem !important; color: #aaa !important; }
.stCheckbox label:hover { color: #e2e2e2 !important; }

.stButton > button[kind="primary"] {
    background: #3b82f6 !important; color: #fff !important;
    border: none !important; border-radius: 5px !important;
    font-size: 0.8rem !important; font-weight: 500 !important;
    width: 100%; letter-spacing: 0.01em;
}
.stButton > button[kind="primary"]:hover { background: #2563eb !important; }
.stButton > button[kind="primary"]:disabled { background: #151515 !important; color: #333 !important; }

.stButton > button:not([kind="primary"]) {
    background: transparent !important; color: #444 !important;
    border: 1px solid #1e1e1e !important; border-radius: 5px !important;
    font-size: 0.75rem !important; width: 100%;
}
.stButton > button:not([kind="primary"]):hover { color: #888 !important; border-color: #2a2a2a !important; }

.stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid #141414; font-size: 0.78rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { color: #444; }
.v-white  { color: #ddd; font-weight: 500; }
.v-green  { color: #22c55e; font-weight: 500; }
.v-blue   { color: #3b82f6; font-weight: 500; }
.v-purple { color: #a78bfa; font-weight: 500; }
.v-amber  { color: #f59e0b; font-weight: 500; }
.v-red    { color: #ef4444; font-weight: 500; }

.empty-state {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 60px 24px; color: #2a2a2a;
    border: 1px dashed #1a1a1a; border-radius: 8px; text-align: center; gap: 6px;
}

.stExpander {
    background: #0f0f0f !important; border: 1px solid #1a1a1a !important;
    border-radius: 7px !important; margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary { font-size: 0.82rem !important; color: #bbb !important; padding: 12px 14px !important; }

.step-card {
    background: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 6px;
    padding: 13px 15px; margin: 7px 0; font-size: 0.8rem;
}
.step-label { color: #777; font-size: 0.75rem; font-weight: 500; margin-bottom: 6px; }
.prompt-text { color: #3a3a3a; font-style: italic; font-size: 0.75rem; margin-bottom: 10px; }

.badge {
    display: inline-block; font-size: 0.65rem; font-weight: 500;
    padding: 2px 6px; border-radius: 3px; margin-right: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.b-hit    { background: #052e16; color: #4ade80; border: 1px solid #14532d; }
.b-miss   { background: #1c0808; color: #f87171; border: 1px solid #3b0e0e; }
.b-comp   { background: #1a1000; color: #fbbf24; border: 1px solid #431407; }
.b-routed { background: #0f0a1e; color: #a78bfa; border: 1px solid #2e1065; }

.meta-line { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 8px; font-size: 0.75rem; color: #555; }
.meta-line strong { color: #c0c0c0; }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; }
.muted { color: #555; }

.response-box {
    margin-top: 10px; padding: 9px 11px; background: #080808;
    border-left: 2px solid #1e1e1e; border-radius: 3px;
    font-size: 0.75rem; color: #555; line-height: 1.65;
}

[data-testid="stStatus"] { background: #0f0f0f !important; border: 1px solid #1a1a1a !important; border-radius: 7px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Renderer functions — defined BEFORE the Streamlit render loop
# ══════════════════════════════════════════════════════════════════════════════

def _render_analytics_snapshot(demo_result):
    data    = demo_result.extra
    summary = data.get("summary", {})
    models  = data.get("model_distribution", [])
    recent  = data.get("recent_requests", [])

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Requests",     f"{summary.get('total_requests', 0):,}")
    with c2: st.metric("Cache Rate",   f"{summary.get('cache_hit_rate', 0)*100:.1f}%")
    with c3: st.metric("Saved",        f"${summary.get('total_savings_usd', 0):.6f}")
    with c4: st.metric("Tokens Saved", f"{summary.get('total_tokens_saved', 0):,}")

    if models:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Model Distribution</div>', unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame(models).set_index("model")["requests"], height=120)

    if recent:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Recent Requests</div>', unsafe_allow_html=True)
        wanted = ["timestamp", "model_used", "cache_hit", "latency_ms", "cost_usd", "savings_usd"]
        df = pd.DataFrame(recent[:5])[[c for c in wanted if c in pd.DataFrame(recent[:5]).columns]].copy()
        if "cache_hit" in df:
            df["cache_hit"] = df["cache_hit"].map({True: "hit", False: "miss"})
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_step(step):
    if step.status == "error":
        st.markdown(f"""
        <div class="step-card" style="border-color:#2a1515;">
            <div class="step-label" style="color:#f87171;">{step.label}</div>
            <div style="color:#555; font-size:0.75rem;">{step.error}</div>
        </div>""", unsafe_allow_html=True)
        return

    # Badges
    badges = ""
    if step.cache_hit:
        badges += '<span class="badge b-hit">CACHE HIT</span>'
    else:
        badges += '<span class="badge b-miss">MISS</span>'
    if step.compressed:
        badges += '<span class="badge b-comp">COMPRESSED</span>'
    if step.routed:
        badges += '<span class="badge b-routed">ROUTED</span>'

    # Model
    if step.routed and step.model_used and step.model_requested and step.model_used != step.model_requested:
        model_html = (f'<span class="mono muted">{step.model_requested}</span>'
                      f' <span style="color:#333;">→</span>'
                      f' <span class="mono" style="color:#a78bfa;">{step.model_used}</span>')
    else:
        model_html = f'<span class="mono muted">{step.model_used or "—"}</span>'

    # Tokens
    tok_str = ""
    if step.tokens_input or step.tokens_output:
        tok_str = f'<span class="mono">{step.tokens_input:,}/{step.tokens_output:,}</span>'
        if step.tokens_saved > 0:
            tok_str += f' <span style="color:#f59e0b;">−{step.tokens_saved:,}</span>'

    # Cost
    cost_str = f'<span style="color:#333;">${step.cost_usd:.6f}</span>'
    if step.savings_usd > 0:
        cost_str += f' <span style="color:#22c55e; margin-left:6px;">+${step.savings_usd:.6f} saved</span>'

    # Routing detail
    routing_html = ""
    if step.complexity or step.routing_reason:
        parts = []
        if step.complexity:
            parts.append(f'complexity: <strong style="color:#777;">{step.complexity}</strong>')
        if step.routing_reason:
            parts.append(step.routing_reason[:90])
        routing_html = f'<div style="margin-top:7px; font-size:0.72rem; color:#333;">{" · ".join(parts)}</div>'

    # Response preview
    preview_html = ""
    if step.response_preview:
        preview_html = f'<div class="response-box">{step.response_preview}</div>'

    prompt_disp = step.prompt[:110] + ("…" if len(step.prompt) > 110 else "")

    st.markdown(f"""
    <div class="step-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:5px;">
            <span class="step-label">{step.label}</span>
            <div>{badges}</div>
        </div>
        <div class="prompt-text">"{prompt_disp}"</div>
        <div class="meta-line">
            <span>model &nbsp;{model_html}</span>
            <span><strong>{step.latency_ms:,}ms</strong></span>
            <span>{cost_str}</span>
            {f'<span>tokens &nbsp;{tok_str}</span>' if tok_str else ''}
        </div>
        {routing_html}
        {preview_html}
    </div>""", unsafe_allow_html=True)


def _render_result_card(demo_result, run_idx: int):
    run_num = len(st.session_state.demo_history) - run_idx
    mark    = "✓" if demo_result.status == "success" else ("~" if demo_result.status == "partial" else "✗")

    chips = []
    cache_hits = sum(1 for s in demo_result.steps if s.cache_hit)
    if cache_hits:
        chips.append(f"⚡ {cache_hits} hit")
    if demo_result.total_tokens_saved > 0:
        chips.append(f"{demo_result.total_tokens_saved:,} tok saved")
    if demo_result.total_savings_usd > 0:
        chips.append(f"${demo_result.total_savings_usd:.6f}")

    label = f"{mark}  Run #{run_num} — {demo_result.scenario_name}"
    if chips:
        label += "    " + "  ·  ".join(chips)

    with st.expander(label, expanded=(run_idx == 0)):
        if demo_result.status == "error" and demo_result.steps and demo_result.steps[0].error:
            st.markdown(f'<div style="color:#f87171; font-size:0.8rem; padding:6px 0;">{demo_result.steps[0].error}</div>',
                        unsafe_allow_html=True)
            return

        if demo_result.scenario_key == "analytics_summary" and demo_result.extra:
            _render_analytics_snapshot(demo_result)
            return

        for step in demo_result.steps:
            _render_step(step)


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
if "demo_history" not in st.session_state:
    st.session_state.demo_history = []
if "demo_running" not in st.session_state:
    st.session_state.demo_running = False


# ── Health check ───────────────────────────────────────────────────────────────
def _is_backend_up():
    try:
        import requests
        return requests.get("http://localhost:8000/health", timeout=2).status_code == 200
    except Exception:
        return False

backend_ok = _is_backend_up()


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
dot_cls     = "dot-green" if backend_ok else "dot-red"
status_text = "API running" if backend_ok else "API offline"

st.markdown(f"""
<div class="top-bar">
  <div>
    <span class="logo">opti<span class="logo-dot">.</span>llm</span>
    <span class="page-tag">/ demo runner</span>
  </div>
  <div class="status-pill">
    <span class="{dot_cls}"></span>
    <span>{status_text} &nbsp;·&nbsp; :8000</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not _demos_available:
    st.error("`demos/` module not found. Run from the project root.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Main layout
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 2.5], gap="large")


# ─── LEFT: Catalog ────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-label">Scenarios</div>', unsafe_allow_html=True)

    categories = {}
    for sc in SCENARIO_CATALOG:
        categories.setdefault(sc["category"], []).append(sc)

    selected_keys = []
    for cat_name, scenarios in categories.items():
        st.markdown(f'<div class="cat-label">{cat_name}</div>', unsafe_allow_html=True)
        for sc in scenarios:
            if st.checkbox(f"{sc['icon']}  {sc['name']}", key=f"chk_{sc['key']}", help=sc["description"]):
                selected_keys.append(sc["key"])

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    btn_label = (
        f"Run {len(selected_keys)} scenario{'s' if len(selected_keys) != 1 else ''}"
        if selected_keys else "Select scenarios above"
    )
    run_clicked = st.button(
        btn_label,
        disabled=(not selected_keys or st.session_state.demo_running or not backend_ok),
        type="primary",
        key="run_btn",
    )
    if st.button("Clear history", key="clear_btn"):
        st.session_state.demo_history = []
        st.rerun()

    # Session totals
    if st.session_state.demo_history:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">This Session</div>', unsafe_allow_html=True)

        all_steps    = [s for r in st.session_state.demo_history for s in r.steps]
        total_saved  = sum(r.total_savings_usd for r in st.session_state.demo_history)
        total_toks   = sum(r.total_tokens_saved for r in st.session_state.demo_history)
        cache_hits   = sum(1 for s in all_steps if s.cache_hit)
        routed_n     = sum(1 for s in all_steps if s.routed)
        comp_n       = sum(1 for s in all_steps if s.compressed)
        errors_n     = sum(1 for r in st.session_state.demo_history if r.status == "error")

        rows = [
            ("runs",           f"{len(st.session_state.demo_history)}",  "v-white"),
            ("steps",          f"{len(all_steps)}",                      "v-white"),
            ("saved",          f"${total_saved:.6f}",                    "v-green"),
            ("tokens saved",   f"{total_toks:,}",                        "v-green"),
            ("cache hits",     f"{cache_hits}",                          "v-blue"),
            ("routed",         f"{routed_n}",                            "v-purple"),
            ("compressed",     f"{comp_n}",                              "v-amber"),
            ("errors",         f"{errors_n}",                            "v-red" if errors_n else "v-white"),
        ]
        html_rows = "".join(
            f'<div class="stat-row"><span class="stat-key">{k}</span><span class="{c}">{v}</span></div>'
            for k, v, c in rows
        )
        st.markdown(html_rows, unsafe_allow_html=True)


# ─── RIGHT: Results feed ───────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

    # Execute
    if run_clicked and selected_keys:
        st.session_state.demo_running = True
        scenario_map = {sc["key"]: sc for sc in SCENARIO_CATALOG}

        for key in selected_keys:
            sc = scenario_map.get(key)
            if not sc:
                continue
            with st.status(f"Running {sc['icon']} {sc['name']}…", expanded=True) as sw:
                result = sc["run"]()
                st.session_state.demo_history.insert(0, result)
                if result.status == "error":
                    sw.update(label=f"Failed — {sc['name']}", state="error")
                else:
                    sw.update(
                        label=f"Done — {sc['icon']} {sc['name']}  ({result.elapsed_ms:,}ms)",
                        state="complete",
                    )

        st.session_state.demo_running = False
        st.rerun()

    # Empty state
    if not st.session_state.demo_history:
        st.markdown("""
        <div class="empty-state">
            <span style="font-size:1.4rem; opacity:0.4;">⚡</span>
            <span style="color:#333; font-size:0.85rem;">No runs yet</span>
            <span style="color:#2a2a2a; font-size:0.75rem;">Select scenarios and click Run</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        for run_idx, demo_result in enumerate(st.session_state.demo_history):
            _render_result_card(demo_result, run_idx)
