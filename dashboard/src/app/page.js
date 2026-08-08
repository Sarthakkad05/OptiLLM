"use client";

import { useState } from "react";

export default function PromptingDashboard() {
  const [promptInput, setPromptInput] = useState("Explain machine learning in one short sentence.");
  const [bypassCache, setBypassCache] = useState(true);
  const [activeView, setActiveView] = useState("assistant"); // "assistant" or "json"
  
  const [loading, setLoading] = useState(false);
  const [responseOutput, setResponseOutput] = useState(null);
  const [rawJson, setRawJson] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [httpStatus, setHttpStatus] = useState(null);
  const [latency, setLatency] = useState(null);

  const handleSendPrompt = async () => {
    if (!promptInput.trim()) return;
    setLoading(true);
    setResponseOutput(null);
    setRawJson(null);
    setTelemetry(null);
    setHttpStatus(null);

    const startTime = performance.now();
    const payload = {
      model: "gpt-4o",
      messages: [{ role: "user", content: promptInput }],
      optillm: {
        bypass_cache: bypassCache,
        bypass_compression: false,
        bypass_routing: false,
      },
    };

    try {
      const res = await fetch("http://localhost:8000/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const elapsed = Math.round(performance.now() - startTime);
      setLatency(elapsed);
      setHttpStatus(res.status);

      const data = await res.json();
      setRawJson(data);

      if (res.ok) {
        setResponseOutput(data.choices?.[0]?.message?.content || "No response content.");
        setTelemetry(data.optillm_metadata || {});
      } else {
        setResponseOutput(`Error (${res.status}): ${data.detail || JSON.stringify(data)}`);
      }
    } catch (err) {
      setHttpStatus(500);
      setResponseOutput(`Network Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const loadPreset = (text, bypass) => {
    setPromptInput(text);
    setBypassCache(bypass);
  };

  return (
    <div style={{ padding: "32px", fontFamily: "'Plus Jakarta Sans', sans-serif", color: "#f8fafc", background: "#07090e", minHeight: "100vh", display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Header */}
      <header style={{ borderBottom: "1px solid #1e293b", paddingBottom: "20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{ width: "36px", height: "36px", background: "linear-gradient(135deg, #38bdf8, #6366f1)", borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "18px", fontWeight: "bold" }}>⚡</div>
          <h1 style={{ fontSize: "22px", margin: 0, color: "#38bdf8", fontWeight: "800", letterSpacing: "-0.5px" }}>OptiLLM Gateway — Live Prompting Studio</h1>
        </div>
        <div style={{ background: "rgba(16, 185, 129, 0.1)", border: "1px solid rgba(16, 185, 129, 0.25)", color: "#10b981", padding: "6px 14px", borderRadius: "20px", fontSize: "12px", fontWeight: "600", display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ width: "8px", height: "8px", background: "#10b981", borderRadius: "50%", boxShadow: "0 0 8px #10b981" }}></span>
          Gateway Connected (http://localhost:8000)
        </div>
      </header>

      {/* Main Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.1fr", gap: "28px", flex: 1 }}>
        {/* Left: Prompt Input */}
        <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: "20px", padding: "28px", display: "flex", flexDirection: "column", gap: "20px", boxShadow: "0 20px 40px rgba(0, 0, 0, 0.4)" }}>
          <h2 style={{ fontSize: "16px", margin: 0, color: "#38bdf8", fontWeight: "700" }}>✏️ Prompt Input</h2>

          <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: "12px", color: "#94a3b8", fontWeight: "600" }}>Try Presets:</span>
            <button onClick={() => loadPreset("Explain machine learning in one short sentence.", true)} style={{ background: "rgba(30, 41, 59, 0.7)", border: "1px solid #1e293b", color: "#94a3b8", padding: "6px 12px", borderRadius: "8px", fontSize: "12px", fontWeight: "600", cursor: "pointer" }}>
              ⚡ Simple Prompt (Routes to gpt-4o-mini)
            </button>
            <button onClick={() => loadPreset("What is quantum computing in simple terms?", false)} style={{ background: "rgba(30, 41, 59, 0.7)", border: "1px solid #1e293b", color: "#94a3b8", padding: "6px 12px", borderRadius: "8px", fontSize: "12px", fontWeight: "600", cursor: "pointer" }}>
              🔄 Repeat Query (Tests Cache Hit)
            </button>
            <button onClick={() => loadPreset("Write a Python function for quicksort with complexity comments.", true)} style={{ background: "rgba(30, 41, 59, 0.7)", border: "1px solid #1e293b", color: "#94a3b8", padding: "6px 12px", borderRadius: "8px", fontSize: "12px", fontWeight: "600", cursor: "pointer" }}>
              💻 Code Query (Retains gpt-4o)
            </button>
          </div>

          <textarea
            rows={8}
            value={promptInput}
            onChange={(e) => setPromptInput(e.target.value)}
            placeholder="Type your prompt here..."
            style={{ width: "100%", background: "#020617", border: "1px solid #1e293b", borderRadius: "14px", padding: "18px", color: "#f8fafc", fontSize: "15px", lineHeight: "1.6", outline: "none", resize: "vertical", fontFamily: "inherit" }}
          />

          <button
            onClick={handleSendPrompt}
            disabled={loading}
            style={{
              background: "linear-gradient(135deg, #38bdf8, #6366f1)",
              color: "#fff",
              border: "none",
              borderRadius: "12px",
              padding: "16px",
              fontSize: "15px",
              fontWeight: "700",
              cursor: loading ? "not-allowed" : "pointer",
              boxShadow: "0 4px 20px rgba(56, 189, 248, 0.35)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "10px"
            }}
          >
            {loading ? "Processing Gateway..." : "Send to OptiLLM Gateway ➔"}
          </button>
        </div>

        {/* Right: Live Telemetry & Output */}
        <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: "20px", padding: "28px", display: "flex", flexDirection: "column", gap: "20px", boxShadow: "0 20px 40px rgba(0, 0, 0, 0.4)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid #1e293b", paddingBottom: "14px" }}>
            <h2 style={{ fontSize: "16px", margin: 0, color: "#38bdf8", fontWeight: "700" }}>📊 Live OptiLLM Telemetry</h2>
            {httpStatus && (
              <span style={{ background: httpStatus === 200 ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)", color: httpStatus === 200 ? "#10b981" : "#ef4444", border: `1px solid ${httpStatus === 200 ? "rgba(16, 185, 129, 0.3)" : "rgba(239, 68, 68, 0.3)"}`, padding: "4px 10px", borderRadius: "6px", fontSize: "12px", fontWeight: "700" }}>
                HTTP {httpStatus} ({latency}ms)
              </span>
            )}
          </div>

          {/* Telemetry Metrics Grid */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px" }}>
            <div style={{ background: "#020617", border: "1px solid #1e293b", borderRadius: "12px", padding: "12px 16px" }}>
              <span style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "#94a3b8" }}>Cache Status</span>
              <div style={{ fontSize: "14px", fontWeight: "800", color: telemetry?.cache_hit ? "#10b981" : "#94a3b8", marginTop: "4px" }}>
                {telemetry ? (telemetry.cache_hit ? "CACHE HIT" : "CACHE MISS") : "-"}
              </div>
            </div>
            <div style={{ background: "#020617", border: "1px solid #1e293b", borderRadius: "12px", padding: "12px 16px" }}>
              <span style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "#94a3b8" }}>Model Used</span>
              <div style={{ fontSize: "14px", fontWeight: "800", color: "#38bdf8", marginTop: "4px" }}>
                {telemetry?.model_used || "-"}
              </div>
            </div>
            <div style={{ background: "#020617", border: "1px solid #1e293b", borderRadius: "12px", padding: "12px 16px" }}>
              <span style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "#94a3b8" }}>Latency</span>
              <div style={{ fontSize: "14px", fontWeight: "800", color: "#f8fafc", marginTop: "4px" }}>
                {telemetry?.latency_ms ? `${telemetry.latency_ms} ms` : (latency ? `${latency} ms` : "-")}
              </div>
            </div>
            <div style={{ background: "#020617", border: "1px solid #1e293b", borderRadius: "12px", padding: "12px 16px" }}>
              <span style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "#94a3b8" }}>Task Complexity</span>
              <div style={{ fontSize: "14px", fontWeight: "800", color: "#818cf8", marginTop: "4px" }}>
                {telemetry?.complexity ? telemetry.complexity.toUpperCase() : "-"}
              </div>
            </div>
            <div style={{ background: "#020617", border: "1px solid #1e293b", borderRadius: "12px", padding: "12px 16px" }}>
              <span style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "#94a3b8" }}>Actual Cost</span>
              <div style={{ fontSize: "14px", fontWeight: "800", color: "#f59e0b", marginTop: "4px" }}>
                ${(telemetry?.cost_usd || 0).toFixed(6)}
              </div>
            </div>
            <div style={{ background: "#020617", border: "1px solid #1e293b", borderRadius: "12px", padding: "12px 16px" }}>
              <span style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "#94a3b8" }}>Cost Saved</span>
              <div style={{ fontSize: "14px", fontWeight: "800", color: "#10b981", marginTop: "4px" }}>
                ${(telemetry?.savings_usd || 0).toFixed(6)}
              </div>
            </div>
          </div>

          {telemetry?.routing_reason && (
            <div style={{ background: "rgba(56, 189, 248, 0.08)", border: "1px solid rgba(56, 189, 248, 0.25)", borderRadius: "12px", padding: "14px 18px", fontSize: "13px", color: "#93c5fd", lineHeight: "1.5" }}>
              💡 <strong>Routing Decision:</strong> {telemetry.routing_reason}
            </div>
          )}

          {/* View Toggle Tabs */}
          <div style={{ display: "flex", gap: "8px", borderBottom: "1px solid #1e293b", paddingBottom: "8px" }}>
            <button
              onClick={() => setActiveView("assistant")}
              style={{
                background: activeView === "assistant" ? "rgba(56, 189, 248, 0.1)" : "none",
                border: "none",
                color: activeView === "assistant" ? "#38bdf8" : "#94a3b8",
                fontWeight: "600",
                fontSize: "13px",
                padding: "6px 14px",
                borderRadius: "6px",
                cursor: "pointer"
              }}
            >
              💬 Assistant Response
            </button>
            <button
              onClick={() => setActiveView("json")}
              style={{
                background: activeView === "json" ? "rgba(56, 189, 248, 0.1)" : "none",
                border: "none",
                color: activeView === "json" ? "#38bdf8" : "#94a3b8",
                fontWeight: "600",
                fontSize: "13px",
                padding: "6px 14px",
                borderRadius: "6px",
                cursor: "pointer"
              }}
            >
              📄 Raw JSON Response
            </button>
          </div>

          {activeView === "assistant" ? (
            <div style={{ background: "#020617", border: "1px solid #1e293b", borderRadius: "14px", padding: "20px", minHeight: "180px", fontSize: "14px", lineHeight: "1.7", color: "#e2e8f0", whiteSpace: "pre-wrap", flex: 1, overflowY: "auto", maxHeight: "320px" }}>
              {responseOutput || <span style={{ color: "#475569", fontStyle: "italic" }}>Type a prompt and click "Send to OptiLLM Gateway" to view the response output and live telemetry metrics.</span>}
            </div>
          ) : (
            <pre style={{ background: "#020617", border: "1px solid #1e293b", borderRadius: "14px", padding: "18px", fontSize: "12px", color: "#38bdf8", fontFamily: "'Fira Code', monospace", flex: 1, overflowX: "auto", maxHeight: "320px", margin: 0 }}>
              {rawJson ? JSON.stringify(rawJson, null, 2) : <span style={{ color: "#475569", fontStyle: "italic" }}>// Raw JSON payload will appear here after sending request.</span>}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
