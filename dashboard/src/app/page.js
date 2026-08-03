"use client";

import { useEffect, useState } from "react";

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/v1/analytics")
      .then((res) => res.json())
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div style={{ padding: "40px", fontFamily: "sans-serif", color: "#fff", background: "#0f172a", minHeight: "100vh" }}>
        <h2>Loading OptiLLM Gateway Analytics...</h2>
      </div>
    );
  }

  const summary = data?.summary || {};

  return (
    <div style={{ padding: "40px", fontFamily: "sans-serif", color: "#f8fafc", background: "#0f172a", minHeight: "100vh" }}>
      <header style={{ borderBottom: "1px solid #334155", paddingBottom: "20px", marginBottom: "30px" }}>
        <h1 style={{ fontSize: "28px", margin: 0, color: "#38bdf8" }}>⚡ OptiLLM Production Dashboard</h1>
        <p style={{ color: "#94a3b8", marginTop: "5px" }}>Real-time Gateway Observability & Optimization Analytics</p>
      </header>

      {/* KPI Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "20px", marginBottom: "40px" }}>
        <div style={{ background: "#1e293b", padding: "20px", borderRadius: "12px", border: "1px solid #334155" }}>
          <span style={{ color: "#94a3b8", fontSize: "14px" }}>Total Requests</span>
          <h2 style={{ fontSize: "32px", margin: "10px 0 0 0", color: "#f8fafc" }}>{summary.total_requests || 0}</h2>
        </div>
        <div style={{ background: "#1e293b", padding: "20px", borderRadius: "12px", border: "1px solid #334155" }}>
          <span style={{ color: "#94a3b8", fontSize: "14px" }}>Cache Hit Rate</span>
          <h2 style={{ fontSize: "32px", margin: "10px 0 0 0", color: "#34d399" }}>{((summary.cache_hit_rate || 0) * 100).toFixed(1)}%</h2>
        </div>
        <div style={{ background: "#1e293b", padding: "20px", borderRadius: "12px", border: "1px solid #334155" }}>
          <span style={{ color: "#94a3b8", fontSize: "14px" }}>Total Cost</span>
          <h2 style={{ fontSize: "32px", margin: "10px 0 0 0", color: "#fbbf24" }}>${(summary.total_cost || 0).toFixed(4)}</h2>
        </div>
        <div style={{ background: "#1e293b", padding: "20px", borderRadius: "12px", border: "1px solid #334155" }}>
          <span style={{ color: "#94a3b8", fontSize: "14px" }}>Total Savings</span>
          <h2 style={{ fontSize: "32px", margin: "10px 0 0 0", color: "#38bdf8" }}>${(summary.total_savings || 0).toFixed(4)}</h2>
        </div>
      </div>

      {/* Recent Activity Table */}
      <div style={{ background: "#1e293b", padding: "25px", borderRadius: "12px", border: "1px solid #334155" }}>
        <h3 style={{ marginTop: 0, color: "#cbd5e1" }}>Recent Gateway Requests</h3>
        <table style={{ width: "100%", borderCollapse: "collapse", color: "#e2e8f0" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #334155", textAlign: "left" }}>
              <th style={{ padding: "10px" }}>ID</th>
              <th style={{ padding: "10px" }}>Requested</th>
              <th style={{ padding: "10px" }}>Used</th>
              <th style={{ padding: "10px" }}>Latency</th>
              <th style={{ padding: "10px" }}>Cost</th>
              <th style={{ padding: "10px" }}>Flags</th>
            </tr>
          </thead>
          <tbody>
            {(data?.recent_requests || []).slice(0, 5).map((r) => (
              <tr key={r.id} style={{ borderBottom: "1px solid #1e293b" }}>
                <td style={{ padding: "10px", fontSize: "14px" }}>{r.id}</td>
                <td style={{ padding: "10px" }}>{r.model_requested}</td>
                <td style={{ padding: "10px", color: "#38bdf8" }}>{r.model_used}</td>
                <td style={{ padding: "10px" }}>{r.latency_ms}ms</td>
                <td style={{ padding: "10px", color: "#34d399" }}>${r.cost_usd.toFixed(6)}</td>
                <td style={{ padding: "10px", fontSize: "12px" }}>
                  {r.cache_hit && <span style={{ background: "#065f46", padding: "3px 8px", borderRadius: "4px", marginRight: "4px" }}>CACHE</span>}
                  {r.routed && <span style={{ background: "#1e40af", padding: "3px 8px", borderRadius: "4px" }}>ROUTED</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
