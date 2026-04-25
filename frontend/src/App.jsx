/**
 * AlphaLab Frontend — React Dashboard
 * Aesthetic: Bloomberg terminal meets modern quant desk.
 * Dark background, IBM Plex Mono for data, sharp signal colours.
 * 4 tabs: Factor Leaderboard | Factor Deep Dive | Portfolio View | Pipeline Monitor
 */

import { useState, useEffect, useCallback, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell,
} from "recharts";

// ─────────────────────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────────────────────

const API = "http://localhost:8000";

const COLORS = {
  bg:        "#07080d",
  surface:   "#0d0f1a",
  border:    "#1a1d2e",
  borderHi:  "#2a2d42",
  text:      "#c8ccd8",
  textDim:   "#5a5e72",
  textBright:"#e8ecf8",
  green:     "#00d68f",
  greenDim:  "#004d34",
  red:       "#ff4757",
  redDim:    "#4d0f17",
  amber:     "#ffb830",
  amberDim:  "#4d3800",
  blue:      "#4fc3f7",
  blueDim:   "#0a2d3d",
  purple:    "#a78bfa",
  chart1:    "#00d68f",
  chart2:    "#4fc3f7",
};

// ─────────────────────────────────────────────────────────────────────────────
// GLOBAL STYLES (injected once)
// ─────────────────────────────────────────────────────────────────────────────

const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html, body, #root {
    height: 100%;
    background: ${COLORS.bg};
    color: ${COLORS.text};
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }

  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: ${COLORS.bg}; }
  ::-webkit-scrollbar-thumb { background: ${COLORS.border}; border-radius: 2px; }

  .fade-in {
    animation: fadeIn 0.3s ease forwards;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .pulse {
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
  }

  .blink {
    animation: blink 1s step-end infinite;
  }
  @keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
  }

  table { border-collapse: collapse; width: 100%; }
  th {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: ${COLORS.textDim};
    padding: 6px 12px;
    border-bottom: 1px solid ${COLORS.border};
    text-align: left;
    white-space: nowrap;
  }
  td {
    padding: 7px 12px;
    border-bottom: 1px solid ${COLORS.border};
    white-space: nowrap;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,0.02); }

  button {
    cursor: pointer;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    border: none;
    outline: none;
  }

  .tag {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 2px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
`;

function InjectStyles() {
  useEffect(() => {
    const style = document.createElement("style");
    style.textContent = GLOBAL_CSS;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// API HOOKS
// ─────────────────────────────────────────────────────────────────────────────

function useApi(endpoint, deps = []) {
  const [data, setData]   = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetch_ = useCallback(async () => {
    if (!endpoint) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}${endpoint}`);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setData(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [endpoint, ...deps]);

  useEffect(() => { fetch_(); }, [fetch_]);
  return { data, loading, error, refetch: fetch_ };
}

// ─────────────────────────────────────────────────────────────────────────────
// SMALL PRIMITIVES
// ─────────────────────────────────────────────────────────────────────────────

function StatusTag({ status }) {
  const map = {
    active:   { bg: COLORS.greenDim, color: COLORS.green,  label: "ACTIVE"   },
    decaying: { bg: COLORS.amberDim, color: COLORS.amber,  label: "DECAYING" },
    weak:     { bg: COLORS.redDim,   color: COLORS.red,    label: "WEAK"     },
  };
  const s = map[status] || { bg: COLORS.border, color: COLORS.textDim, label: status?.toUpperCase() || "—" };
  return (
    <span className="tag" style={{ background: s.bg, color: s.color }}>
      {s.label}
    </span>
  );
}

function RecoTag({ reco }) {
  const map = {
    LONG:    { bg: COLORS.greenDim, color: COLORS.green },
    SHORT:   { bg: COLORS.redDim,   color: COLORS.red   },
    NEUTRAL: { bg: COLORS.border,   color: COLORS.textDim },
  };
  const s = map[reco] || map.NEUTRAL;
  return (
    <span className="tag" style={{ background: s.bg, color: s.color }}>
      {reco}
    </span>
  );
}

function Num({ v, pct, color }) {
  if (v === null || v === undefined) return <span style={{ color: COLORS.textDim }}>—</span>;
  const val = pct ? (v * 100).toFixed(2) + "%" : v.toFixed(4);
  const col = color || (v > 0 ? COLORS.green : v < 0 ? COLORS.red : COLORS.text);
  return <span style={{ color: col }}>{v > 0 && !pct ? "+" : ""}{val}</span>;
}

function Loader({ label = "FETCHING DATA" }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, color: COLORS.textDim, padding: "40px 0" }}>
      <span className="pulse" style={{ color: COLORS.blue, fontSize: 16 }}>◈</span>
      <span style={{ fontSize: 11, letterSpacing: "0.12em" }}>{label}</span>
    </div>
  );
}

function ErrorBox({ msg }) {
  return (
    <div style={{
      border: `1px solid ${COLORS.redDim}`,
      background: "rgba(255,71,87,0.06)",
      color: COLORS.red,
      padding: "14px 18px",
      borderRadius: 3,
      fontSize: 12,
      margin: "12px 0",
    }}>
      ⚠ {msg}
      <div style={{ marginTop: 6, color: COLORS.textDim, fontSize: 11 }}>
        Make sure the FastAPI server is running at {API}
      </div>
    </div>
  );
}

function Card({ children, style }) {
  return (
    <div style={{
      background: COLORS.surface,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 4,
      overflow: "hidden",
      ...style,
    }}>
      {children}
    </div>
  );
}

function CardHeader({ title, subtitle, right }) {
  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "12px 16px",
      borderBottom: `1px solid ${COLORS.border}`,
    }}>
      <div>
        <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.12em", color: COLORS.textBright, textTransform: "uppercase" }}>
          {title}
        </div>
        {subtitle && <div style={{ fontSize: 10, color: COLORS.textDim, marginTop: 2 }}>{subtitle}</div>}
      </div>
      {right && <div>{right}</div>}
    </div>
  );
}

function StatBox({ label, value, color }) {
  return (
    <div style={{
      padding: "14px 20px",
      borderRight: `1px solid ${COLORS.border}`,
      minWidth: 120,
    }}>
      <div style={{ fontSize: 10, color: COLORS.textDim, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 5 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 600, color: color || COLORS.textBright }}>{value}</div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB 1 — FACTOR LEADERBOARD
// ─────────────────────────────────────────────────────────────────────────────

function FactorLeaderboard() {
  const [statusFilter, setStatusFilter] = useState("all");
  const endpoint = statusFilter === "all" ? "/factors?limit=25" : `/factors?status=${statusFilter}&limit=25`;
  const { data, loading, error } = useApi(endpoint, [statusFilter]);

  const filters = ["all", "active", "decaying", "weak"];

  return (
    <div className="fade-in" style={{ padding: "24px 28px" }}>

      {/* Stats row */}
      {data && (
        <div style={{ display: "flex", marginBottom: 24, border: `1px solid ${COLORS.border}`, borderRadius: 4, background: COLORS.surface, overflow: "hidden" }}>
          <StatBox label="Total Factors"   value={data.total}          color={COLORS.textBright} />
          <StatBox label="Active"          value={data.active_count}   color={COLORS.green} />
          <StatBox label="Decaying"        value={data.decaying_count} color={COLORS.amber} />
          <StatBox label="Weak / No Edge"  value={data.total - data.active_count - data.decaying_count} color={COLORS.red} />
        </div>
      )}

      {/* Filter tabs */}
      <div style={{ display: "flex", gap: 4, marginBottom: 16 }}>
        {filters.map(f => (
          <button
            key={f}
            onClick={() => setStatusFilter(f)}
            style={{
              padding: "5px 14px",
              borderRadius: 2,
              background: statusFilter === f ? COLORS.blue : "transparent",
              color: statusFilter === f ? COLORS.bg : COLORS.textDim,
              border: `1px solid ${statusFilter === f ? COLORS.blue : COLORS.border}`,
              transition: "all 0.15s",
            }}
          >
            {f.toUpperCase()}
          </button>
        ))}
      </div>

      {loading && <Loader />}
      {error   && <ErrorBox msg={error} />}

      {data && (
        <Card>
          <table>
            <thead>
              <tr>
                <th style={{ width: 40 }}>#</th>
                <th>Factor</th>
                <th>Sharpe</th>
                <th>Ann. Return</th>
                <th>Max Drawdown</th>
                <th>Win Rate</th>
                <th>Factor Decay</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {data.factors.map(f => (
                <tr key={f.factor}>
                  <td style={{ color: COLORS.textDim }}>{f.rank}</td>
                  <td style={{ color: COLORS.textBright, fontWeight: 500 }}>{f.factor}</td>
                  <td><Num v={f.sharpe} /></td>
                  <td><Num v={f.annual_return} pct /></td>
                  <td><Num v={f.max_drawdown} pct /></td>
                  <td><Num v={f.win_rate} pct color={COLORS.text} /></td>
                  <td><Num v={f.factor_decay} /></td>
                  <td><StatusTag status={f.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB 2 — FACTOR DEEP DIVE
// ─────────────────────────────────────────────────────────────────────────────

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: COLORS.surface,
      border: `1px solid ${COLORS.borderHi}`,
      padding: "8px 12px",
      borderRadius: 3,
      fontSize: 11,
    }}>
      <div style={{ color: COLORS.textDim, marginBottom: 4 }}>{label}</div>
      {payload.map(p => (
        <div key={p.name} style={{ color: p.color }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(4) : p.value}
        </div>
      ))}
    </div>
  );
};

function FactorDeepDive() {
  const { data: factorsData } = useApi("/factors?limit=25");
  const { data: shapData }    = useApi("/factors/shap");
  const [selected, setSelected] = useState(null);
  const { data: btData, loading: btLoading, error: btError } = useApi(
    selected ? `/backtest/${selected}` : null, [selected]
  );

  // Pick first active factor as default
  useEffect(() => {
    if (factorsData && !selected) {
      const first = factorsData.factors.find(f => f.status === "active");
      if (first) setSelected(first.factor);
    }
  }, [factorsData, selected]);

  const pnlPoints = btData?.pnl_series?.map(p => ({
    date: p.date,
    pnl:  parseFloat((p.cumulative_pnl * 100).toFixed(4)),
  })) || [];

  return (
    <div className="fade-in" style={{ padding: "24px 28px", display: "grid", gridTemplateColumns: "220px 1fr", gap: 20 }}>

      {/* Factor selector sidebar */}
      <div>
        <Card>
          <CardHeader title="Factors" />
          <div style={{ maxHeight: 520, overflowY: "auto" }}>
            {factorsData?.factors.map(f => (
              <div
                key={f.factor}
                onClick={() => setSelected(f.factor)}
                style={{
                  padding: "9px 14px",
                  cursor: "pointer",
                  background: selected === f.factor ? COLORS.blueDim : "transparent",
                  borderLeft: `2px solid ${selected === f.factor ? COLORS.blue : "transparent"}`,
                  display: "flex",
                  flexDirection: "column",
                  gap: 3,
                  transition: "all 0.1s",
                }}
              >
                <span style={{
                  fontSize: 11,
                  color: selected === f.factor ? COLORS.textBright : COLORS.text,
                  fontWeight: selected === f.factor ? 600 : 400,
                }}>
                  {f.factor}
                </span>
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <StatusTag status={f.status} />
                  <span style={{ fontSize: 10, color: f.sharpe > 0 ? COLORS.green : COLORS.red }}>
                    SR {f.sharpe?.toFixed(2) ?? "—"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Right panel */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

        {/* Metrics row */}
        {btData && (
          <div style={{ display: "flex", border: `1px solid ${COLORS.border}`, borderRadius: 4, background: COLORS.surface, overflow: "hidden" }}>
            <StatBox label="Sharpe Ratio"  value={btData.sharpe?.toFixed(3) ?? "—"} color={btData.sharpe > 0 ? COLORS.green : COLORS.red} />
            <StatBox label="Ann. Return"   value={btData.annual_return !== null ? (btData.annual_return * 100).toFixed(1) + "%" : "—"} color={btData.annual_return > 0 ? COLORS.green : COLORS.red} />
            <StatBox label="Max Drawdown"  value={btData.max_drawdown !== null ? (btData.max_drawdown * 100).toFixed(1) + "%" : "—"} color={COLORS.red} />
            <StatBox label="Win Rate"      value={btData.win_rate !== null ? (btData.win_rate * 100).toFixed(1) + "%" : "—"} color={COLORS.text} />
            <StatBox label="Status"        value={<StatusTag status={btData.status} />} />
          </div>
        )}

        {/* P&L Chart */}
        <Card>
          <CardHeader
            title={selected ? `${selected} — Cumulative P&L` : "Select a Factor"}
            subtitle="Long/short equal-weight backtest · 21d rebalance"
          />
          <div style={{ padding: "16px 8px" }}>
            {btLoading && <Loader label="LOADING P&L SERIES" />}
            {btError   && <ErrorBox msg={btError} />}
            {!btLoading && pnlPoints.length === 0 && (
              <div style={{ color: COLORS.textDim, padding: "30px 20px", fontSize: 11 }}>
                No P&L time-series available. Run backtester.py with{" "}
                <code style={{ color: COLORS.amber }}>save_pnl=True</code> to generate per-factor parquets.
              </div>
            )}
            {pnlPoints.length > 0 && (
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={pnlPoints} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="2 4" stroke={COLORS.border} />
                  <XAxis dataKey="date" tick={{ fontSize: 9, fill: COLORS.textDim }} tickLine={false} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 9, fill: COLORS.textDim }} tickLine={false} tickFormatter={v => `${v}%`} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={0} stroke={COLORS.borderHi} strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="pnl" stroke={COLORS.green} strokeWidth={1.5} dot={false} name="Cum. P&L %" />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>

        {/* Global SHAP importance */}
        {shapData && (
          <Card>
            <CardHeader title="Global SHAP — Factor Importance in Composite Model" subtitle="Mean absolute SHAP value across all stocks" />
            <div style={{ padding: "16px 8px" }}>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={shapData} layout="vertical" margin={{ left: 80, right: 20 }}>
                  <CartesianGrid strokeDasharray="2 4" stroke={COLORS.border} horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 9, fill: COLORS.textDim }} tickLine={false} tickFormatter={v => v.toFixed(4)} />
                  <YAxis type="category" dataKey="feature" tick={{ fontSize: 10, fill: COLORS.text }} width={150} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="mean_abs_shap" radius={[0, 2, 2, 0]} name="Mean |SHAP|">
                    {shapData.map((_, i) => (
                      <Cell key={i} fill={i === 0 ? COLORS.green : COLORS.blue} fillOpacity={1 - i * 0.15} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB 3 — PORTFOLIO VIEW
// ─────────────────────────────────────────────────────────────────────────────

function ShapBar({ shap, max }) {
  const pct = Math.min(Math.abs(shap) / max, 1) * 100;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{
        width: 60,
        height: 4,
        background: COLORS.border,
        borderRadius: 2,
        overflow: "hidden",
      }}>
        <div style={{
          width: `${pct}%`,
          height: "100%",
          background: shap > 0 ? COLORS.green : COLORS.red,
          borderRadius: 2,
          transition: "width 0.3s",
        }} />
      </div>
      <span style={{ color: shap > 0 ? COLORS.green : COLORS.red, fontSize: 11 }}>
        {shap > 0 ? "+" : ""}{shap?.toFixed(4)}
      </span>
    </div>
  );
}

function PortfolioView() {
  const { data, loading, error } = useApi("/portfolio?top_n=10");
  const [leg, setLeg] = useState("long");
  const [expanded, setExpanded] = useState(null);

  const stocks = leg === "long"
    ? (data?.long_portfolio  || [])
    : (data?.short_portfolio || []);

  // Compute max shap for bar scaling
  const maxShap = Math.max(...stocks.map(s => Math.abs(s.shap_1 || 0)), 0.001);

  return (
    <div className="fade-in" style={{ padding: "24px 28px" }}>

      {/* Summary stats */}
      {data && (
        <div style={{ display: "flex", marginBottom: 24, border: `1px solid ${COLORS.border}`, borderRadius: 4, background: COLORS.surface, overflow: "hidden" }}>
          <StatBox label="Total Stocks"  value={data.total_stocks} />
          <StatBox label="Long Book"     value={data.long_portfolio.length}  color={COLORS.green} />
          <StatBox label="Short Book"    value={data.short_portfolio.length} color={COLORS.red} />
          <StatBox label="Generated"     value={data.generated_at?.slice(11, 19) + " UTC"} color={COLORS.textDim} />
        </div>
      )}

      {/* Leg toggle */}
      <div style={{ display: "flex", gap: 4, marginBottom: 16 }}>
        {["long", "short"].map(l => (
          <button
            key={l}
            onClick={() => setLeg(l)}
            style={{
              padding: "5px 18px",
              borderRadius: 2,
              background: leg === l ? (l === "long" ? COLORS.greenDim : COLORS.redDim) : "transparent",
              color: leg === l ? (l === "long" ? COLORS.green : COLORS.red) : COLORS.textDim,
              border: `1px solid ${leg === l ? (l === "long" ? COLORS.green : COLORS.red) : COLORS.border}`,
              fontWeight: leg === l ? 600 : 400,
              transition: "all 0.15s",
            }}
          >
            {l === "long" ? "▲ LONG" : "▼ SHORT"}
          </button>
        ))}
      </div>

      {loading && <Loader />}
      {error   && <ErrorBox msg={error} />}

      {stocks.length > 0 && (
        <Card>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Ticker</th>
                <th>Alpha Score</th>
                <th>Pred. Return</th>
                <th>Signal</th>
                <th>Top Driver</th>
                <th>SHAP Impact</th>
                <th>2nd Driver</th>
                <th>3rd Driver</th>
              </tr>
            </thead>
            <tbody>
              {stocks.map((s, i) => (
                <>
                  <tr
                    key={s.ticker}
                    onClick={() => setExpanded(expanded === s.ticker ? null : s.ticker)}
                    style={{ cursor: "pointer" }}
                  >
                    <td style={{ color: COLORS.textDim }}>{s.alpha_rank}</td>
                    <td style={{ color: COLORS.textBright, fontWeight: 600 }}>
                      {expanded === s.ticker ? "▾ " : "▸ "}{s.ticker}
                    </td>
                    <td>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <div style={{
                          width: 50, height: 5,
                          background: COLORS.border,
                          borderRadius: 3, overflow: "hidden",
                        }}>
                          <div style={{
                            width: `${(s.composite_alpha_score || 0) * 100}%`,
                            height: "100%",
                            background: leg === "long" ? COLORS.green : COLORS.red,
                            borderRadius: 3,
                          }} />
                        </div>
                        <span style={{ color: COLORS.textBright }}>{s.composite_alpha_score?.toFixed(3)}</span>
                      </div>
                    </td>
                    <td><Num v={s.predicted_return} pct /></td>
                    <td><RecoTag reco={s.recommendation} /></td>
                    <td style={{ color: COLORS.blue, fontSize: 11 }}>{s.top_factor_1 || "—"}</td>
                    <td><ShapBar shap={s.shap_1} max={maxShap} /></td>
                    <td style={{ color: COLORS.textDim, fontSize: 11 }}>{s.top_factor_2 || "—"}</td>
                    <td style={{ color: COLORS.textDim, fontSize: 11 }}>{s.top_factor_3 || "—"}</td>
                  </tr>
                  {expanded === s.ticker && (
                    <tr key={`${s.ticker}-exp`}>
                      <td colSpan={9} style={{ background: COLORS.blueDim, padding: "12px 20px" }}>
                        <div style={{ fontSize: 11, color: COLORS.textBright, marginBottom: 6, fontWeight: 500 }}>
                          SHAP EXPLANATION
                        </div>
                        <div style={{ color: COLORS.text, fontSize: 11, lineHeight: 1.7 }}>
                          {s.explanation || "No explanation available."}
                        </div>
                        <div style={{ display: "flex", gap: 24, marginTop: 10 }}>
                          {[
                            [s.top_factor_1, s.shap_1],
                            [s.top_factor_2, s.shap_2],
                            [s.top_factor_3, s.shap_3],
                          ].filter(([f]) => f).map(([factor, shap], j) => (
                            <div key={j} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                              <span style={{ color: COLORS.textDim, fontSize: 10, textTransform: "uppercase", letterSpacing: "0.1em" }}>
                                Driver {j + 1}
                              </span>
                              <span style={{ color: COLORS.blue, fontSize: 11 }}>{factor}</span>
                              <span style={{ color: shap > 0 ? COLORS.green : COLORS.red, fontSize: 13, fontWeight: 600 }}>
                                {shap > 0 ? "+" : ""}{shap?.toFixed(5)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB 4 — PIPELINE MONITOR
// ─────────────────────────────────────────────────────────────────────────────

function PipelineMonitor() {
  const { data: health, loading: hLoading } = useApi("/health");
  const { data: ready,  loading: rLoading } = useApi("/ready");
  const [lastRefresh, setLastRefresh] = useState(new Date());

  useEffect(() => {
    const t = setInterval(() => setLastRefresh(new Date()), 30000);
    return () => clearInterval(t);
  }, []);

  const pipeline = [
    { step: "01", name: "Data Ingestion",     file: "dag_stocks_ingestion.py",  output: "data/raw/YYYY-MM-DD.parquet",              desc: "Airflow DAG · 9AM UTC weekdays · 100 NSE stocks" },
    { step: "02", name: "Factor Engineering", file: "momentum / value / volatility / mean_reversion", output: "data/features/*.parquet", desc: "4 factor modules · ~51 columns" },
    { step: "03", name: "Factor Combination", file: "combine_factors.py",        output: "data/features/all_factors.parquet",         desc: "Merge + overall_composite · 100 stocks × 51 cols" },
    { step: "04", name: "Backtesting",        file: "backtester.py",             output: "data/backtest/factor_backtest_summary.parquet", desc: "L/S backtest · 25 factors · 21d rebalance" },
    { step: "05", name: "Factor Ranking",     file: "factor_ranker.py",          output: "data/backtest/factor_rankings.parquet",     desc: "Sharpe rank · active/decaying/weak status" },
    { step: "06", name: "Composite Model",    file: "composite_model.py",        output: "data/models/composite_model.json",          desc: "XGBoost · 4 active factors · 21d fwd target" },
    { step: "07", name: "SHAP Explainer",     file: "shap_explainer.py",         output: "data/models/shap_explanations.parquet",     desc: "TreeExplainer · top-3 drivers per stock" },
    { step: "08", name: "API",                file: "src/api/main.py",           output: "http://localhost:8000",                     desc: "FastAPI · 6 endpoints · Prometheus /metrics" },
  ];

  const serviceStatus = [
    { name: "FastAPI",    port: 8000, color: health ? COLORS.green : COLORS.red,  status: health ? "UP" : hLoading ? "..." : "DOWN" },
    { name: "MLflow",     port: 5000, color: COLORS.textDim, status: "CHECK" },
    { name: "Airflow",    port: 8080, color: COLORS.textDim, status: "CHECK" },
    { name: "Prometheus", port: 9090, color: COLORS.textDim, status: "CHECK" },
    { name: "Grafana",    port: 3001, color: COLORS.textDim, status: "CHECK" },
  ];

  const filesLoaded = ready?.files_loaded || {};

  return (
    <div className="fade-in" style={{ padding: "24px 28px", display: "grid", gridTemplateColumns: "1fr 280px", gap: 20 }}>

      {/* Left: Pipeline steps */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <Card>
          <CardHeader
            title="Pipeline Stages"
            subtitle={`Last refresh: ${lastRefresh.toLocaleTimeString()}`}
            right={
              <span style={{ fontSize: 10, color: COLORS.textDim, letterSpacing: "0.08em" }}>
                <span className="blink" style={{ color: COLORS.green }}>●</span> LIVE
              </span>
            }
          />
          <div>
            {pipeline.map((p, i) => (
              <div
                key={p.step}
                style={{
                  display: "grid",
                  gridTemplateColumns: "36px 1fr",
                  padding: "12px 16px",
                  borderBottom: i < pipeline.length - 1 ? `1px solid ${COLORS.border}` : "none",
                  gap: 12,
                }}
              >
                {/* Step number + connector */}
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", paddingTop: 2 }}>
                  <div style={{
                    width: 24, height: 24,
                    borderRadius: "50%",
                    background: COLORS.greenDim,
                    border: `1px solid ${COLORS.green}`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 9, fontWeight: 600, color: COLORS.green,
                    flexShrink: 0,
                  }}>
                    {p.step}
                  </div>
                  {i < pipeline.length - 1 && (
                    <div style={{ width: 1, flex: 1, background: COLORS.border, marginTop: 4 }} />
                  )}
                </div>

                {/* Content */}
                <div style={{ paddingBottom: i < pipeline.length - 1 ? 8 : 0 }}>
                  <div style={{ fontWeight: 600, color: COLORS.textBright, fontSize: 12, marginBottom: 3 }}>
                    {p.name}
                  </div>
                  <div style={{ color: COLORS.blue, fontSize: 11, marginBottom: 2 }}>{p.file}</div>
                  <div style={{ color: COLORS.textDim, fontSize: 10 }}>{p.desc}</div>
                  <div style={{
                    marginTop: 6,
                    display: "inline-block",
                    background: COLORS.border,
                    padding: "2px 8px",
                    borderRadius: 2,
                    fontSize: 10,
                    color: COLORS.textDim,
                    fontFamily: "IBM Plex Mono, monospace",
                  }}>
                    → {p.output}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Right column */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

        {/* Service status */}
        <Card>
          <CardHeader title="Services" />
          {serviceStatus.map(s => (
            <div key={s.name} style={{
              display: "flex", justifyContent: "space-between", alignItems: "center",
              padding: "9px 16px",
              borderBottom: `1px solid ${COLORS.border}`,
            }}>
              <div>
                <div style={{ color: COLORS.textBright, fontSize: 11, fontWeight: 500 }}>{s.name}</div>
                <div style={{ color: COLORS.textDim, fontSize: 10 }}>:{s.port}</div>
              </div>
              <span className="tag" style={{
                background: s.status === "UP" ? COLORS.greenDim : s.status === "DOWN" ? COLORS.redDim : COLORS.border,
                color: s.status === "UP" ? COLORS.green : s.status === "DOWN" ? COLORS.red : COLORS.textDim,
              }}>
                {s.status}
              </span>
            </div>
          ))}
        </Card>

        {/* Data files loaded */}
        <Card>
          <CardHeader title="Data Files" subtitle="/ready endpoint" />
          {Object.entries(filesLoaded).map(([file, loaded]) => (
            <div key={file} style={{
              display: "flex", justifyContent: "space-between", alignItems: "center",
              padding: "8px 16px",
              borderBottom: `1px solid ${COLORS.border}`,
            }}>
              <span style={{ fontSize: 11, color: COLORS.text }}>{file}</span>
              <span style={{ color: loaded ? COLORS.green : COLORS.red, fontSize: 11 }}>
                {loaded ? "✓ loaded" : "✗ missing"}
              </span>
            </div>
          ))}
          {Object.keys(filesLoaded).length === 0 && (
            <div style={{ padding: "12px 16px", color: COLORS.textDim, fontSize: 11 }}>
              {rLoading ? "Checking..." : "API not reachable"}
            </div>
          )}
        </Card>

        {/* Quick links */}
        <Card>
          <CardHeader title="Quick Links" />
          {[
            { label: "API Docs (Swagger)", href: "http://localhost:8000/docs" },
            { label: "MLflow Experiments", href: "http://localhost:5000" },
            { label: "Airflow DAGs",       href: "http://localhost:8080" },
            { label: "Prometheus",         href: "http://localhost:9090" },
            { label: "Grafana",            href: "http://localhost:3001" },
          ].map(l => (
            <a
              key={l.label}
              href={l.href}
              target="_blank"
              rel="noreferrer"
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "9px 16px",
                borderBottom: `1px solid ${COLORS.border}`,
                color: COLORS.blue,
                textDecoration: "none",
                fontSize: 11,
                transition: "background 0.1s",
              }}
              onMouseEnter={e => e.currentTarget.style.background = COLORS.blueDim}
              onMouseLeave={e => e.currentTarget.style.background = "transparent"}
            >
              {l.label}
              <span style={{ color: COLORS.textDim }}>↗</span>
            </a>
          ))}
        </Card>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TOP NAV + APP SHELL
// ─────────────────────────────────────────────────────────────────────────────

const TABS = [
  { id: "leaderboard", label: "01 FACTOR LEADERBOARD" },
  { id: "deepdive",    label: "02 FACTOR DEEP DIVE"   },
  { id: "portfolio",   label: "03 PORTFOLIO VIEW"     },
  { id: "monitor",     label: "04 PIPELINE MONITOR"   },
];

export default function App() {
  const [tab, setTab] = useState("leaderboard");
  const [clock, setClock] = useState(new Date());
  const { data: healthData } = useApi("/health");

  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <>
      <InjectStyles />
      <div style={{ height: "100vh", display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* Top bar */}
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 28px",
          height: 44,
          background: COLORS.surface,
          borderBottom: `1px solid ${COLORS.border}`,
          flexShrink: 0,
        }}>
          {/* Logo */}
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{
              width: 28, height: 28,
              background: COLORS.green,
              borderRadius: 3,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 13, fontWeight: 700, color: COLORS.bg,
            }}>
              α
            </div>
            <div>
              <span style={{ fontSize: 13, fontWeight: 600, color: COLORS.textBright, letterSpacing: "0.08em" }}>
                ALPHALAB
              </span>
              <span style={{ fontSize: 10, color: COLORS.textDim, marginLeft: 10 }}>
                NSE FACTOR RESEARCH · v1.0
              </span>
            </div>
          </div>

          {/* Nav tabs */}
          <div style={{ display: "flex", gap: 2 }}>
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                style={{
                  padding: "0 16px",
                  height: 44,
                  background: "transparent",
                  color: tab === t.id ? COLORS.textBright : COLORS.textDim,
                  borderBottom: tab === t.id ? `2px solid ${COLORS.green}` : "2px solid transparent",
                  fontSize: 10,
                  letterSpacing: "0.1em",
                  fontWeight: tab === t.id ? 600 : 400,
                  transition: "all 0.15s",
                }}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Status + clock */}
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10 }}>
              <span className={healthData ? "blink" : ""} style={{ color: healthData ? COLORS.green : COLORS.red }}>
                ●
              </span>
              <span style={{ color: COLORS.textDim }}>
                {healthData ? "API LIVE" : "API DOWN"}
              </span>
            </div>
            <div style={{ fontSize: 11, color: COLORS.textDim, fontVariantNumeric: "tabular-nums" }}>
              {clock.toLocaleTimeString("en-GB")} IST
            </div>
          </div>
        </div>

        {/* Scrollable content */}
        <div style={{ flex: 1, overflowY: "auto" }}>
          {tab === "leaderboard" && <FactorLeaderboard />}
          {tab === "deepdive"    && <FactorDeepDive    />}
          {tab === "portfolio"   && <PortfolioView     />}
          {tab === "monitor"     && <PipelineMonitor   />}
        </div>

      </div>
    </>
  );
}