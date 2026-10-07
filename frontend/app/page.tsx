"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { BarRow, Card, Label, Note, Stat, fmt } from "./components/ui";
import { CARBON_MATERIALS, mixCarbon, referenceMix } from "./lib/carbon";
import { DEFAULT_MIX, FEATURE_LABELS, SLIDERS, saveMix } from "./lib/mixes";
import { en206Grade, explain, outOfRange, predict, type Mix } from "./lib/model";
import { MODEL } from "./lib/modelData";

// ─── Grade colour ─────────────────────────────────────────────────────────────

function gradeColor(grade: string) {
  // "C30/37" → 30 (fck). Below-class results ("< C8/10") fall into Low Strength.
  const n = grade.startsWith("<") ? 0 : parseInt(grade.replace("C", "").split("/")[0] ?? "0");
  if (n <= 20) return { color: "#f6ad55", label: "Low Strength" };
  if (n <= 35) return { color: "#68d391", label: "Normal Strength" };
  if (n <= 55) return { color: "#63b3ed", label: "High Strength" };
  return         { color: "#b794f4", label: "Ultra-High Strength" };
}

function featureValue(key: string, mix: Mix) {
  if (key in mix) {
    const unit = SLIDERS.find((s) => s.key === key)?.unit ?? "";
    return `${mix[key as keyof Mix]} ${unit}`;
  }
  const binder = mix.cement + mix.slag + mix.fly_ash;
  const ratios: Record<string, number> = {
    wc_ratio: mix.water / mix.cement,
    wb_ratio: mix.water / binder,
    fine_coarse_ratio: mix.fine_agg / mix.coarse_agg,
    slag_cement_ratio: mix.slag / mix.cement,
    fly_ash_cement_ratio: mix.fly_ash / mix.cement,
  };
  return key === "binder_total" ? `${binder} kg/m³` : (ratios[key] ?? 0).toFixed(3);
}

// ─── Slider row ───────────────────────────────────────────────────────────────

function SliderRow({
  cfg, value, onChange,
}: {
  cfg: (typeof SLIDERS)[number];
  value: number;
  onChange: (k: keyof Mix, v: number) => void;
}) {
  const pct = ((value - cfg.min) / (cfg.max - cfg.min)) * 100;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <label htmlFor={`slider-${cfg.key}`} style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-2)" }}>
          {cfg.label}
        </label>
        <span style={{ fontSize: 13, fontWeight: 700, color: "var(--accent)", fontVariantNumeric: "tabular-nums" }}>
          {value}
          <span style={{ fontSize: 10, fontWeight: 400, color: "var(--text-3)", marginLeft: 3 }}>{cfg.unit}</span>
        </span>
      </div>
      <div style={{ position: "relative", height: 20, display: "flex", alignItems: "center" }}>
        <div
          style={{
            position: "absolute", left: 0, width: `${pct}%`, height: 4, borderRadius: 99,
            background: "linear-gradient(90deg, rgba(99,179,237,0.35), var(--accent))",
            pointerEvents: "none", zIndex: 1,
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
        />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
        <span style={{ fontSize: 11, color: "var(--text-3)", lineHeight: 1.45, flex: 1 }}>{cfg.note}</span>
        <span style={{ fontSize: 10, color: "var(--text-3)", whiteSpace: "nowrap", paddingTop: 1 }}>
          {cfg.min}–{cfg.max}
        </span>
      </div>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function Home() {
  const [mix, setMix] = useState<Mix>(DEFAULT_MIX);

  // Remember the mix so the BIM page can use it
  useEffect(() => saveMix(mix), [mix]);

  const handleChange = useCallback((k: keyof Mix, v: number) => {
    setMix((p) => ({ ...p, [k]: v }));
  }, []);

  const exp = useMemo(() => explain(MODEL, mix), [mix]);
  const strength = Math.max(exp.prediction, 0);
  const grade = en206Grade(strength);
  const gc = gradeColor(grade);
  const carbon = useMemo(() => mixCarbon(mix), [mix]);
  const extrapolating = outOfRange(MODEL, mix);

  const ref = useMemo(() => {
    const m = referenceMix(mix);
    const s = Math.max(predict(MODEL, m), 0);
    return { mix: m, strength: s, grade: en206Grade(s), carbon: mixCarbon(m).total, outOfRange: outOfRange(MODEL, m) };
  }, [mix]);
  const usesScm = mix.slag + mix.fly_ash > 0;

  const shapRows = Object.entries(exp.shap)
    .map(([key, value]) => ({ key, label: FEATURE_LABELS[key] ?? key, val: featureValue(key, mix), value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const maxShap = Math.max(...shapRows.map((r) => Math.abs(r.value)), 1);

  const carbonRows = CARBON_MATERIALS
    .map((k) => ({ key: k, label: FEATURE_LABELS[k], value: carbon.byMaterial[k] }))
    .sort((a, b) => b.value - a.value);
  const maxCarbon = Math.max(...carbonRows.map((r) => r.value), 1);

  const wc = mix.water / mix.cement;
  const wb = mix.water / (mix.cement + mix.slag + mix.fly_ash);

  return (
    <>
      {/* ── Hero ── */}
      <div className="anim-fadeUp" style={{ textAlign: "center", padding: "48px 24px 32px" }}>
        <h1 style={{ fontSize: "clamp(2rem, 5vw, 3.1rem)", fontWeight: 900, letterSpacing: "-0.03em", lineHeight: 1.1, color: "var(--text-1)", marginBottom: 14 }}>
          Predict Concrete{" "}
          <span style={{ background: "linear-gradient(90deg,#63b3ed,#76e4f7)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            Strength
          </span>{" "}
          &amp; Carbon
        </h1>
        <p style={{ fontSize: 15, color: "var(--text-2)", maxWidth: 560, margin: "0 auto", lineHeight: 1.65 }}>
          Set the mix design and see the predicted compressive strength, its EN&nbsp;206 class, what drives it,
          and the mix&apos;s embodied carbon. Everything is calculated in your browser.
        </p>
      </div>

      {/* ── Row 1: sliders + results ── */}
      <div className="two-col-grid">
        <Card style={{ padding: 32 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 32 }}>
            <Label style={{ fontSize: 11, color: "var(--text-2)" }}>Mix Design Parameters</Label>
            <button
              id="reset-btn"
              onClick={() => setMix(DEFAULT_MIX)}
              style={{
                fontSize: 11, fontWeight: 600, padding: "6px 14px", borderRadius: 10,
                background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
                color: "var(--text-2)", cursor: "pointer",
              }}
            >
              Reset defaults
            </button>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
            {SLIDERS.map((cfg) => (
              <SliderRow key={cfg.key} cfg={cfg} value={mix[cfg.key]} onChange={handleChange} />
            ))}
          </div>
        </Card>

        <div className="sticky-panel" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Strength */}
          <Card style={{ padding: "28px 26px", borderColor: `${gc.color}28` }}>
            <Label style={{ textAlign: "center", marginBottom: 10 }}>Predicted compressive strength</Label>
            <div style={{ textAlign: "center", marginBottom: 16 }}>
              <span id="strength-value" style={{ fontSize: 72, fontWeight: 900, lineHeight: 1, color: gc.color, textShadow: `0 0 48px ${gc.color}55` }}>
                {strength.toFixed(1)}
              </span>
              <span style={{ fontSize: 22, fontWeight: 300, color: "var(--text-2)", marginLeft: 6 }}>MPa</span>
            </div>
            <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
              <div
                style={{
                  display: "inline-flex", alignItems: "center", gap: 10, padding: "8px 18px", borderRadius: 99,
                  background: `${gc.color}14`, border: `1px solid ${gc.color}38`,
                }}
              >
                <div style={{ width: 7, height: 7, borderRadius: "50%", background: gc.color }} />
                <span style={{ fontWeight: 800, fontSize: 13, color: "var(--text-1)" }}>{grade}</span>
                <span style={{ fontSize: 11, color: "var(--text-2)", fontWeight: 500 }}>{gc.label}</span>
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, paddingTop: 16, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
              <Stat label="w/c" value={wc.toFixed(2)} sub="≤0.45 durable" />
              <Stat label="w/b" value={wb.toFixed(2)} sub={usesScm ? "incl. SCM" : "= w/c"} />
              <Stat label="Age" value={`${mix.age}d`} sub={mix.age === 28 ? "standard" : "custom"} />
            </div>
            {extrapolating.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <Note tone="warning">
                  {extrapolating.map((k) => FEATURE_LABELS[k]).join(", ")} outside the range of the training data —
                  the prediction is an extrapolation.
                </Note>
              </div>
            )}
          </Card>

          {/* Carbon */}
          <Card style={{ padding: "24px 26px" }}>
            <Label style={{ marginBottom: 12 }}>Embodied carbon (A1–A3)</Label>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, marginBottom: 18 }}>
              <div>
                <span id="carbon-value" style={{ fontSize: 34, fontWeight: 900, color: "var(--text-1)" }}>{fmt(carbon.total, 0)}</span>
                <span style={{ fontSize: 13, color: "var(--text-2)", marginLeft: 6 }}>kg CO₂e/m³</span>
              </div>
              <div style={{ textAlign: "right" }}>
                <span style={{ fontSize: 16, fontWeight: 800, color: "var(--text-1)" }}>{strength > 0 ? fmt(carbon.total / strength, 1) : "–"}</span>
                <span style={{ fontSize: 11, color: "var(--text-2)", marginLeft: 4 }}>kg CO₂e per MPa</span>
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {carbonRows.filter((r) => r.value > 0.05).map((r) => (
                <BarRow
                  key={r.key}
                  label={r.label}
                  value={`${fmt(r.value, 1)} kg · ${Math.round((r.value / carbon.total) * 100)}%`}
                  fraction={r.value / maxCarbon}
                  title={`${r.label}: ${fmt(r.value, 1)} kg CO₂e/m³`}
                />
              ))}
            </div>
            <p style={{ fontSize: 10, color: "var(--text-3)", lineHeight: 1.5, marginTop: 14 }}>
              Generic emission factors (mostly ICE v3.0), cradle to gate, concrete only.{" "}
              <Link href="/model#carbon">How it&apos;s calculated</Link>
            </p>
          </Card>
        </div>
      </div>

      {/* ── Row 2: SHAP + reference ── */}
      <div className="half-grid" style={{ marginTop: 24 }}>
        <Card style={{ padding: "26px 24px" }}>
          <Label style={{ marginBottom: 10 }}>What drives this prediction (SHAP)</Label>
          <p style={{ fontSize: 12, color: "var(--text-2)", marginBottom: 14, lineHeight: 1.5 }}>
            How each input moves the prediction from the dataset average (<strong>{exp.baseValue.toFixed(1)} MPa</strong>) to this
            mix (<strong>{strength.toFixed(1)} MPa</strong>).
          </p>
          <div style={{ display: "flex", gap: 16, fontSize: 11, color: "var(--text-2)", marginBottom: 16 }}>
            <span><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 3, background: "var(--positive)", marginRight: 6 }} />raises strength</span>
            <span><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 3, background: "var(--negative)", marginRight: 6 }} />lowers strength</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
            {shapRows.map(({ key, label, val, value }) => {
              const pos = value >= 0;
              const pct = (Math.abs(value) / maxShap) * 50;
              return (
                <div key={key} title={`${label}: ${pos ? "+" : ""}${value.toFixed(2)} MPa`} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", fontSize: 11 }}>
                    <span style={{ fontWeight: 600, color: "var(--text-1)" }}>{label}</span>
                    <span style={{ color: "var(--text-3)", fontSize: 10, fontVariantNumeric: "tabular-nums" }}>{val}</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{ flex: 1, position: "relative", height: 8, background: "rgba(255,255,255,0.03)", borderRadius: 4 }}>
                      <div style={{ position: "absolute", left: "50%", top: -2, bottom: -2, width: 1, background: "rgba(255,255,255,0.18)" }} />
                      <div
                        style={{
                          position: "absolute", height: "100%",
                          left: pos ? "50%" : `${50 - pct}%`, width: `${pct}%`,
                          background: pos ? "var(--positive)" : "var(--negative)",
                          borderRadius: pos ? "0 4px 4px 0" : "4px 0 0 4px",
                        }}
                      />
                    </div>
                    <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-1)", fontVariantNumeric: "tabular-nums", width: 52, textAlign: "right" }}>
                      {pos ? "+" : "−"}{Math.abs(value).toFixed(2)}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Reference comparison */}
          <Card style={{ padding: "26px 24px" }}>
            <Label style={{ marginBottom: 10 }}>Compared with a Portland-cement-only mix</Label>
            <p style={{ fontSize: 12, color: "var(--text-2)", marginBottom: 16, lineHeight: 1.5 }}>
              Reference: the same mix with all binder as Portland cement (slag and fly ash replaced 1:1), so the
              binder content and w/b ratio stay the same.
            </p>
            {usesScm ? (
              <table className="data" id="reference-table">
                <thead>
                  <tr>
                    <th></th>
                    <th className="num">Your mix</th>
                    <th className="num">Reference</th>
                    <th className="num">Change</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Strength</td>
                    <td className="num">{strength.toFixed(1)} MPa</td>
                    <td className="num">{ref.strength.toFixed(1)} MPa</td>
                    <td className="num">{change(strength, ref.strength)}</td>
                  </tr>
                  <tr>
                    <td>EN 206 class</td>
                    <td className="num">{grade}</td>
                    <td className="num">{ref.grade}</td>
                    <td className="num"></td>
                  </tr>
                  <tr className="strong">
                    <td>kg CO₂e/m³</td>
                    <td className="num">{fmt(carbon.total, 0)}</td>
                    <td className="num">{fmt(ref.carbon, 0)}</td>
                    <td className="num">{change(carbon.total, ref.carbon)}</td>
                  </tr>
                  <tr>
                    <td>kg CO₂e per MPa</td>
                    <td className="num">{strength > 0 ? fmt(carbon.total / strength, 1) : "–"}</td>
                    <td className="num">{ref.strength > 0 ? fmt(ref.carbon / ref.strength, 1) : "–"}</td>
                    <td className="num">{strength > 0 && ref.strength > 0 ? change(carbon.total / strength, ref.carbon / ref.strength) : ""}</td>
                  </tr>
                </tbody>
              </table>
            ) : (
              <Note>
                This mix uses only Portland cement, so it is its own reference. Add slag or fly ash to see how much
                carbon they save and what they do to the predicted strength.
              </Note>
            )}
            {usesScm && ref.outOfRange.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <Note tone="warning">
                  The reference mix has {ref.mix.cement} kg/m³ cement, above the training data ({MODEL.data_ranges.cement[1]} kg/m³),
                  so its predicted strength is an extrapolation.
                </Note>
              </div>
            )}
          </Card>

          {/* EN 206 legend */}
          <Card style={{ padding: "18px 20px" }}>
            <Label style={{ marginBottom: 14 }}>EN 206 grade reference</Label>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {[
                { range: "C8–C20",  color: "#f6ad55", desc: "Low Strength (fck ≤ 20 MPa)" },
                { range: "C25–C35", color: "#68d391", desc: "Normal Structural (fck 25–35)" },
                { range: "C40–C55", color: "#63b3ed", desc: "High Strength (fck 40–55)" },
                { range: "C60+",    color: "#b794f4", desc: "Ultra-High Strength (fck ≥ 60)" },
              ].map(({ range, color, desc }) => (
                <div key={range} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{ width: 7, height: 7, borderRadius: "50%", background: color, flexShrink: 0 }} />
                  <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-1)", fontFamily: "monospace", minWidth: 60 }}>{range}</span>
                  <span style={{ fontSize: 11, color: "var(--text-2)" }}>{desc}</span>
                </div>
              ))}
            </div>
            <p style={{ fontSize: 10, color: "var(--text-3)", lineHeight: 1.5, marginTop: 14 }}>
              Class estimated from fck = predicted mean − 8 MPa (EN 1992-1-1).
            </p>
          </Card>
        </div>
      </div>
    </>
  );
}

/** "+12%" / "−34%" relative to the reference */
function change(value: number, reference: number) {
  if (!reference) return "";
  const pct = ((value - reference) / reference) * 100;
  if (Math.abs(pct) < 0.5) return "0%";
  return `${pct > 0 ? "+" : "−"}${Math.abs(pct).toFixed(0)}%`;
}
