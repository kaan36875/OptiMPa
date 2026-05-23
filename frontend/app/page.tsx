"use client";

import { useState, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

interface ConcreteFeatures {
  cement: number;
  slag: number;
  fly_ash: number;
  water: number;
  superplasticizer: number;
  coarse_agg: number;
  fine_agg: number;
  age: number;
}

interface PredictionResult {
  strength_mpa: number;
  strength_grade: string;
  input_summary: ConcreteFeatures;
}

// ─── Slider configs ───────────────────────────────────────────────────────────

const SLIDERS = [
  { key: "cement"          as const, label: "Cement",             unit: "kg/m³", min: 100, max: 700, step: 5,   def: 350, note: "Primary binder — drives early strength" },
  { key: "slag"            as const, label: "Blast Furnace Slag", unit: "kg/m³", min: 0,   max: 360, step: 5,   def: 0,   note: "Latent hydraulic binder — boosts long-term strength" },
  { key: "fly_ash"         as const, label: "Fly Ash",            unit: "kg/m³", min: 0,   max: 200, step: 5,   def: 0,   note: "Pozzolanic filler — reduces heat of hydration" },
  { key: "water"           as const, label: "Water",              unit: "kg/m³", min: 120, max: 250, step: 1,   def: 175, note: "Lower w/c ratio → denser paste → higher strength" },
  { key: "superplasticizer"as const, label: "Superplasticizer",   unit: "kg/m³", min: 0,   max: 32,  step: 0.5, def: 6,   note: "Maintains workability at low water content" },
  { key: "coarse_agg"      as const, label: "Coarse Aggregate",   unit: "kg/m³", min: 800, max: 1150,step: 5,   def: 1040,note: "Structural skeleton — crushed stone or gravel" },
  { key: "fine_agg"        as const, label: "Fine Aggregate",     unit: "kg/m³", min: 550, max: 1000,step: 5,   def: 755, note: "Sand — fills voids between coarse particles" },
  { key: "age"             as const, label: "Curing Age",         unit: "days",  min: 1,   max: 365, step: 1,   def: 28,  note: "28 days = standard reference per EN 206" },
];

const DEFAULTS: ConcreteFeatures = {
  cement: 350,
  slag: 0,
  fly_ash: 0,
  water: 175,
  superplasticizer: 6,
  coarse_agg: 1040,
  fine_agg: 755,
  age: 28,
};

// ─── Grade colour ─────────────────────────────────────────────────────────────

function gradeColor(grade: string) {
  const n = parseInt(grade.replace("C", "").split("/")[0] ?? "0");
  if (n <= 20) return { color: "#f6ad55", label: "Low Strength" };
  if (n <= 35) return { color: "#68d391", label: "Normal Strength" };
  if (n <= 55) return { color: "#63b3ed", label: "High Strength" };
  return         { color: "#b794f4", label: "Ultra-High Strength" };
}

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ─── Component helpers ────────────────────────────────────────────────────────

/** A single labelled slider row */
function SliderRow({
  cfg,
  value,
  onChange,
}: {
  cfg: (typeof SLIDERS)[number];
  value: number;
  onChange: (k: keyof ConcreteFeatures, v: number) => void;
}) {
  const pct = ((value - cfg.min) / (cfg.max - cfg.min)) * 100;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {/* Label / value row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-3)" }}>
          {cfg.label}
        </span>
        <span style={{ fontSize: 13, fontWeight: 700, color: "var(--accent)", fontVariantNumeric: "tabular-nums" }}>
          {value}
          <span style={{ fontSize: 10, fontWeight: 400, color: "var(--text-3)", marginLeft: 3 }}>{cfg.unit}</span>
        </span>
      </div>

      {/* Slider track with accent fill */}
      <div style={{ position: "relative", height: 20, display: "flex", alignItems: "center" }}>
        {/* Filled portion */}
        <div
          style={{
            position: "absolute",
            left: 0,
            width: `${pct}%`,
            height: 4,
            borderRadius: 99,
            background: "linear-gradient(90deg, rgba(99,179,237,0.35), var(--accent))",
            pointerEvents: "none",
            zIndex: 1,
          }}
        />
        <input
          type="range"
          id={`slider-${cfg.key}`}
          min={cfg.min}
          max={cfg.max}
          step={cfg.step}
          value={value}
          onChange={(e) => onChange(cfg.key, parseFloat(e.target.value))}
          style={{ position: "relative", zIndex: 2, width: "100%", margin: 0 }}
          aria-label={cfg.label}
        />
      </div>

      {/* Note / range */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
        <span style={{ fontSize: 11, color: "var(--text-3)", lineHeight: 1.45, flex: 1 }}>{cfg.note}</span>
        <span style={{ fontSize: 10, color: "var(--text-3)", whiteSpace: "nowrap", paddingTop: 1 }}>
          {cfg.min}–{cfg.max}
        </span>
      </div>
    </div>
  );
}

/** Card wrapper */
function Card({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div
      style={{
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.07)",
        borderRadius: 20,
        backdropFilter: "blur(16px)",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function Home() {
  const [values, setValues]   = useState<ConcreteFeatures>(DEFAULTS);
  const [result, setResult]   = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const handleChange = useCallback((k: keyof ConcreteFeatures, v: number) => {
    setValues((p) => ({ ...p, [k]: v }));
  }, []);

  const handleReset = useCallback(() => {
    setValues(DEFAULTS);
    setResult(null);
    setError(null);
  }, []);

  const handlePredict = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error((e as { detail?: string }).detail ?? `Error ${res.status}`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message.includes("fetch")
            ? "Cannot reach the API — make sure FastAPI is running on port 8000."
            : e.message
          : "Unknown error"
      );
    } finally {
      setLoading(false);
    }
  }, [values]);

  const gc = result ? gradeColor(result.strength_grade) : null;
  const wc = result
    ? (result.input_summary.water / result.input_summary.cement).toFixed(3)
    : null;

  return (
    <>
      {/* ── Topnav ── */}
      <nav
        style={{
          position: "sticky", top: 0, zIndex: 50,
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "0 32px", height: 60,
          background: "rgba(8,8,15,0.85)",
          backdropFilter: "blur(20px)",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div
            style={{
              width: 30, height: 30, borderRadius: 9,
              background: "linear-gradient(135deg,#63b3ed,#76e4f7)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 14, fontWeight: 900, color: "#08080f",
            }}
          >
            Ω
          </div>
          <span style={{ fontWeight: 800, fontSize: 15, color: "var(--text-1)", letterSpacing: "-0.02em" }}>
            OptiMPa
          </span>
          <span
            style={{
              fontSize: 10, fontWeight: 700, letterSpacing: "0.08em",
              padding: "3px 10px", borderRadius: 99,
              background: "var(--accent-dim)",
              color: "var(--accent)",
              border: "1px solid var(--border-accent)",
            }}
          >
            RF Model
          </span>
        </div>
        <span style={{ fontSize: 12, color: "var(--text-3)", display: "none" }} className="sm-show">
          Concrete Compressive Strength Predictor
        </span>
      </nav>

      {/* ── Hero ── */}
      <div
        className="anim-fadeUp"
        style={{ textAlign: "center", padding: "52px 24px 36px", position: "relative", zIndex: 1 }}
      >
        <h1
          style={{
            fontSize: "clamp(2rem, 5vw, 3.25rem)",
            fontWeight: 900, letterSpacing: "-0.03em",
            lineHeight: 1.1, color: "var(--text-1)", marginBottom: 16,
          }}
        >
          Predict Concrete{" "}
          <span
            style={{
              background: "linear-gradient(90deg,#63b3ed,#76e4f7)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            Strength
          </span>
        </h1>
        <p style={{ fontSize: 15, color: "var(--text-2)", maxWidth: 520, margin: "0 auto", lineHeight: 1.65 }}>
          Dial in your concrete mix design. A Random Forest trained on the UCI
          Concrete dataset returns the predicted compressive strength and EN&nbsp;206
          grade instantly.
        </p>
      </div>

      {/* ── Two-column layout ── */}
      <div
        className="two-col-grid"
        style={{
          position: "relative", zIndex: 1,
          maxWidth: 1100,
          margin: "0 auto",
          padding: "0 24px 80px",
          display: "grid",
          gridTemplateColumns: "1fr 360px",
          gap: 24,
          alignItems: "start",
        }}
      >
        {/* ── LEFT: Sliders ── */}
        <Card style={{ padding: 32 }}>
          {/* Header */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 32 }}>
            <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.18em", textTransform: "uppercase", color: "var(--text-2)" }}>
              Mix Design Parameters
            </span>
            <button
              id="reset-btn"
              onClick={handleReset}
              style={{
                fontSize: 11, fontWeight: 600,
                padding: "6px 14px", borderRadius: 10,
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                color: "var(--text-3)", cursor: "pointer",
                transition: "all 0.18s",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.color = "var(--text-1)";
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.16)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.color = "var(--text-3)";
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
              }}
            >
              Reset defaults
            </button>
          </div>

          {/* Slider list */}
          <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
            {SLIDERS.map((cfg) => (
              <SliderRow
                key={cfg.key}
                cfg={cfg}
                value={values[cfg.key]}
                onChange={handleChange}
              />
            ))}
          </div>
        </Card>

        {/* ── RIGHT: Sticky panel ── */}
        <div className="sticky-panel" style={{ position: "sticky", top: 80, display: "flex", flexDirection: "column", gap: 16 }}>

          {/* Predict button */}
          <button
            id="predict-btn"
            onClick={handlePredict}
            disabled={loading}
            style={{
              width: "100%", padding: "16px 0",
              borderRadius: 16, border: "none",
              fontWeight: 800, fontSize: 15, letterSpacing: "-0.01em",
              cursor: loading ? "not-allowed" : "pointer",
              background: loading
                ? "rgba(255,255,255,0.05)"
                : "linear-gradient(135deg,#63b3ed,#76e4f7)",
              color: loading ? "var(--text-3)" : "#08080f",
              boxShadow: loading ? "none" : "0 0 32px rgba(99,179,237,0.25)",
              transition: "all 0.2s",
              display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
            }}
          >
            {loading ? (
              <>
                <span className="spinner" />
                Predicting…
              </>
            ) : (
              "Predict Strength"
            )}
          </button>

          {/* Error */}
          {error && (
            <Card style={{ padding: "18px 20px", borderColor: "rgba(252,129,129,0.25)" }}>
              <p style={{ fontSize: 11, fontWeight: 700, color: "#fc8181", marginBottom: 5, textTransform: "uppercase", letterSpacing: "0.1em" }}>
                Connection Error
              </p>
              <p style={{ fontSize: 12, color: "var(--text-2)", lineHeight: 1.5 }}>{error}</p>
            </Card>
          )}

          {/* Result */}
          {result && gc && !error && (
            <Card
              style={{ padding: "32px 28px", borderColor: `${gc.color}28` }}
              key={result.strength_mpa}
            >
              <div className="anim-popIn">
                {/* Label */}
                <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.2em", textTransform: "uppercase", color: "var(--text-3)", textAlign: "center", marginBottom: 10 }}>
                  Predicted Compressive Strength
                </p>

                {/* MPa number */}
                <div style={{ textAlign: "center", marginBottom: 18 }}>
                  <span
                    style={{
                      fontSize: 76, fontWeight: 900, lineHeight: 1,
                      color: gc.color,
                      textShadow: `0 0 48px ${gc.color}55`,
                      fontVariantNumeric: "tabular-nums",
                      display: "inline-block",
                    }}
                  >
                    {result.strength_mpa.toFixed(1)}
                  </span>
                  <span style={{ fontSize: 22, fontWeight: 300, color: "var(--text-2)", marginLeft: 6, verticalAlign: "bottom", lineHeight: 1, display: "inline-block", paddingBottom: 6 }}>
                    MPa
                  </span>
                </div>

                {/* Grade badge */}
                <div style={{ display: "flex", justifyContent: "center", marginBottom: 24 }}>
                  <div
                    style={{
                      display: "inline-flex", alignItems: "center", gap: 10,
                      padding: "8px 20px", borderRadius: 99,
                      background: `${gc.color}14`,
                      border: `1px solid ${gc.color}38`,
                    }}
                  >
                    <div style={{ width: 7, height: 7, borderRadius: "50%", background: gc.color, boxShadow: `0 0 8px ${gc.color}` }} />
                    <span style={{ fontWeight: 800, fontSize: 13, color: gc.color }}>{result.strength_grade}</span>
                    <span style={{ fontSize: 11, color: "var(--text-2)", fontWeight: 500 }}>{gc.label}</span>
                  </div>
                </div>

                {/* Stats */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, paddingTop: 18, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                  {[
                    { label: "w/c Ratio", val: wc,   sub: "≤0.45 durable" },
                    { label: "Cement",    val: `${result.input_summary.cement}`, sub: "kg/m³" },
                    { label: "Age",       val: `${result.input_summary.age}d`,   sub: result.input_summary.age === 28 ? "standard" : "custom" },
                  ].map(({ label, val, sub }) => (
                    <div key={label} style={{ textAlign: "center" }}>
                      <p style={{ fontSize: 10, color: "var(--text-3)", marginBottom: 4, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.1em" }}>{label}</p>
                      <p style={{ fontSize: 14, fontWeight: 800, color: "var(--text-1)", fontVariantNumeric: "tabular-nums" }}>{val}</p>
                      <p style={{ fontSize: 10, color: "var(--text-3)" }}>{sub}</p>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          )}

          {/* Empty state */}
          {!result && !error && !loading && (
            <Card style={{ padding: "40px 28px", textAlign: "center" }}>
              <div
                style={{
                  width: 52, height: 52, borderRadius: 16, margin: "0 auto 14px",
                  background: "var(--accent-dim)", border: "1px solid var(--border-accent)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 22, color: "var(--accent)",
                }}
              >
                ◈
              </div>
              <p style={{ fontSize: 13, fontWeight: 700, color: "var(--text-2)", marginBottom: 8 }}>Ready to predict</p>
              <p style={{ fontSize: 12, color: "var(--text-3)", lineHeight: 1.6 }}>
                Set your mix parameters and press{" "}
                <span style={{ color: "var(--accent)", fontWeight: 600 }}>Predict Strength</span>.
              </p>
            </Card>
          )}

          {/* EN 206 legend */}
          <Card style={{ padding: "18px 20px" }}>
            <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.18em", textTransform: "uppercase", color: "var(--text-3)", marginBottom: 14 }}>
              EN 206 Grade Reference
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {[
                { range: "C8–C20",  color: "#f6ad55", desc: "Low Strength  (< 25 MPa)" },
                { range: "C25–C35", color: "#68d391", desc: "Normal Structural (25–40 MPa)" },
                { range: "C40–C55", color: "#63b3ed", desc: "High Strength (40–60 MPa)" },
                { range: "C60+",    color: "#b794f4", desc: "Ultra-High Strength" },
              ].map(({ range, color, desc }) => (
                <div key={range} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{ width: 7, height: 7, borderRadius: "50%", background: color, flexShrink: 0 }} />
                  <span style={{ fontSize: 11, fontWeight: 700, color, fontFamily: "monospace", minWidth: 52 }}>{range}</span>
                  <span style={{ fontSize: 11, color: "var(--text-3)" }}>{desc}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      {/* ── Footer ── */}
      <footer
        style={{
          position: "relative", zIndex: 1,
          textAlign: "center", padding: "24px 24px",
          borderTop: "1px solid rgba(255,255,255,0.05)",
        }}
      >
        <p style={{ fontSize: 11, color: "var(--text-3)" }}>
          OptiMPa · Random Forest trained on UCI Concrete Compressive Strength (I-Cheng Yeh, 1998)
        </p>
      </footer>

      {/* Responsive: collapse to single column below 768px */}
      <style>{`
        @media (max-width: 768px) {
          .two-col-grid {
            grid-template-columns: 1fr !important;
          }
          .sticky-panel {
            position: static !important;
          }
        }
      `}</style>
    </>
  );
}
