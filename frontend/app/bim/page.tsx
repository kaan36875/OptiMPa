"use client";

import { useMemo, useRef, useState } from "react";
import Link from "next/link";
import type { IfcAPI } from "web-ifc";
import { BarRow, Card, Label, Note, fmt } from "../components/ui";
import { mixCarbon } from "../lib/carbon";
import type { IfcResult } from "../lib/ifc";
import { CLASS_MIXES, DEFAULT_MIX, loadMix } from "../lib/mixes";
import { en206Grade, predict, type Mix } from "../lib/model";
import { MODEL } from "../lib/modelData";

const NO_CLASS = "No class";
type MixSource = "example" | "predictor";

// web-ifc (~6 MB JS + WASM) is loaded only when a file is opened
let apiPromise: Promise<IfcAPI> | null = null;
function getIfcApi() {
  apiPromise ??= (async () => {
    const { IfcAPI, LogLevel } = await import("web-ifc");
    const api = new IfcAPI();
    api.SetWasmPath("/wasm/", true);
    await api.Init(undefined, true); // single-threaded: no special server headers needed
    api.SetLogLevel(LogLevel.LOG_LEVEL_OFF);
    return api;
  })();
  return apiPromise;
}

const classFck = (grade: string) => {
  const m = /C(\d+)\//.exec(grade);
  return m && !grade.startsWith("<") ? Number(m[1]) : 0;
};

export default function BimPage() {
  const [result, setResult] = useState<IfcResult | null>(null);
  const [fileName, setFileName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [sources, setSources] = useState<Record<string, MixSource>>({});
  const inputRef = useRef<HTMLInputElement>(null);

  // The mix last set on the predictor page (read once a model is open — client only)
  const predictorMix: Mix = useMemo(() => (result ? loadMix() : DEFAULT_MIX), [result]);

  async function analyse(name: string, data: Uint8Array) {
    setBusy(true);
    setError(null);
    setFileName(name);
    try {
      const [api, { analyseIfc }] = await Promise.all([getIfcApi(), import("../lib/ifc")]);
      setResult(analyseIfc(api, data));
      setSources({});
    } catch (e) {
      setResult(null);
      setError(`Could not read this file: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  async function openFile(file: File | undefined) {
    if (!file) return;
    await analyse(file.name, new Uint8Array(await file.arrayBuffer()));
  }

  async function openSample() {
    const res = await fetch("/samples/sample_building.ifc");
    await analyse("sample_building.ifc", new Uint8Array(await res.arrayBuffer()));
  }

  // ── Aggregate by strength class, with the mix chosen for each class ──
  const groups = useMemo(() => {
    if (!result) return [];
    const byClass = new Map<string, { count: number; volume: number }>();
    for (const e of result.elements) {
      const key = e.strengthClass ?? NO_CLASS;
      const g = byClass.get(key) ?? { count: 0, volume: 0 };
      g.count++;
      g.volume += e.volume;
      byClass.set(key, g);
    }
    return [...byClass.entries()]
      .map(([cls, g]) => {
        const hasExample = cls in CLASS_MIXES;
        const source: MixSource = sources[cls] ?? (hasExample ? "example" : "predictor");
        const mix = source === "example" && hasExample ? CLASS_MIXES[cls] : predictorMix;
        const strength = Math.max(predict(MODEL, mix), 0);
        const perM3 = mixCarbon(mix).total;
        return {
          cls, ...g, hasExample, source, strength,
          grade: en206Grade(strength),
          meets: cls === NO_CLASS ? null : classFck(en206Grade(strength)) >= classFck(cls),
          perM3, carbon: g.volume * perM3,
        };
      })
      .sort((a, b) => b.carbon - a.carbon);
  }, [result, sources, predictorMix]);

  const perM3ByClass = useMemo(() => Object.fromEntries(groups.map((g) => [g.cls, g.perM3])), [groups]);
  const totalVolume = groups.reduce((s, g) => s + g.volume, 0);
  const totalCarbon = groups.reduce((s, g) => s + g.carbon, 0);

  const byIfcClass = useMemo(() => {
    const m = new Map<string, { volume: number; carbon: number }>();
    for (const e of result?.elements ?? []) {
      const g = m.get(e.ifcClass) ?? { volume: 0, carbon: 0 };
      g.volume += e.volume;
      g.carbon += e.volume * (perM3ByClass[e.strengthClass ?? NO_CLASS] ?? 0);
      m.set(e.ifcClass, g);
    }
    return [...m.entries()].sort((a, b) => b[1].carbon - a[1].carbon);
  }, [result, perM3ByClass]);
  const maxIfcCarbon = Math.max(...byIfcClass.map(([, g]) => g.carbon), 1);

  function downloadCsv() {
    if (!result) return;
    const header = ["global_id", "ifc_class", "name", "material", "strength_class", "volume_m3", "volume_source", "note", "kgco2e_per_m3", "kgco2e"];
    const esc = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;
    const rows = result.elements.map((e) => {
      const perM3 = perM3ByClass[e.strengthClass ?? NO_CLASS] ?? 0;
      return [e.globalId, e.ifcClass, e.name, e.material, e.strengthClass ?? "", e.volume.toFixed(4), e.volumeSource, e.note, perM3.toFixed(1), (e.volume * perM3).toFixed(1)];
    });
    const csv = [header, ...rows].map((r) => r.map(esc).join(",")).join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${fileName.replace(/\.ifc$/i, "") || "model"}-concrete.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "48px 24px 0", display: "flex", flexDirection: "column", gap: 24 }}>
      <div style={{ textAlign: "center" }}>
        <h1 style={{ fontSize: "clamp(1.8rem, 4vw, 2.8rem)", fontWeight: 900, letterSpacing: "-0.03em", color: "var(--text-1)", marginBottom: 12 }}>
          Embodied carbon from a BIM model
        </h1>
        <p style={{ fontSize: 15, color: "var(--text-2)", maxWidth: 620, margin: "0 auto", lineHeight: 1.65 }}>
          Open an IFC file to get the volume of every concrete element and the embodied carbon of the concrete. The file is
          read in your browser and is not uploaded anywhere.
        </p>
      </div>

      {/* ── File picker ── */}
      <Card style={{ padding: 0, borderStyle: "dashed", borderColor: dragging ? "var(--accent)" : "rgba(255,255,255,0.12)" }}>
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); openFile(e.dataTransfer.files[0]); }}
          style={{ padding: "36px 24px", textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}
        >
          <p style={{ fontSize: 14, color: "var(--text-1)", fontWeight: 600 }}>
            {busy ? `Reading ${fileName}…` : "Drop an .ifc file here"}
          </p>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", justifyContent: "center" }}>
            <button
              id="choose-ifc"
              disabled={busy}
              onClick={() => inputRef.current?.click()}
              style={{
                padding: "10px 20px", borderRadius: 12, border: "none", fontWeight: 800, fontSize: 13, cursor: busy ? "wait" : "pointer",
                background: "linear-gradient(135deg,#63b3ed,#76e4f7)", color: "#08080f",
              }}
            >
              Choose IFC file
            </button>
            <button
              id="sample-ifc"
              disabled={busy}
              onClick={openSample}
              style={{
                padding: "10px 20px", borderRadius: 12, fontWeight: 600, fontSize: 13, cursor: busy ? "wait" : "pointer",
                background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.12)", color: "var(--text-1)",
              }}
            >
              Try the sample building
            </button>
          </div>
          <input ref={inputRef} type="file" accept=".ifc" hidden onChange={(e) => openFile(e.target.files?.[0])} />
          <p style={{ fontSize: 11, color: "var(--text-3)", maxWidth: 560, lineHeight: 1.5 }}>
            Concrete = material name, category or description containing &quot;concrete&quot;, &quot;beton&quot; or a class like C30/37. Volumes
            come from the quantity sets (NetVolume / GrossVolume) or, if missing, from the 3D geometry.
          </p>
        </div>
      </Card>

      {error && <Note tone="warning">{error}</Note>}

      {result && !busy && (
        <>
          {/* ── Headline ── */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16 }}>
            <Card style={{ padding: "22px 24px" }}>
              <Label style={{ marginBottom: 8 }}>Embodied carbon of the concrete</Label>
              <span id="bim-total-carbon" style={{ fontSize: 48, fontWeight: 900, color: "var(--text-1)" }}>{fmt(totalCarbon / 1000, 1)}</span>
              <span style={{ fontSize: 15, color: "var(--text-2)", marginLeft: 6 }}>t CO₂e</span>
            </Card>
            <Card style={{ padding: "22px 24px" }}>
              <Label style={{ marginBottom: 8 }}>Concrete volume</Label>
              <span style={{ fontSize: 30, fontWeight: 800, color: "var(--text-1)" }}>{fmt(totalVolume, 2)}</span>
              <span style={{ fontSize: 13, color: "var(--text-2)", marginLeft: 6 }}>m³</span>
              <p style={{ fontSize: 11, color: "var(--text-3)", marginTop: 4 }}>
                {result.elements.length} elements · {totalVolume > 0 ? fmt(totalCarbon / totalVolume, 0) : "–"} kg CO₂e/m³ on average
              </p>
            </Card>
            <Card style={{ padding: "22px 24px" }}>
              <Label style={{ marginBottom: 8 }}>File</Label>
              <p style={{ fontSize: 14, fontWeight: 700, color: "var(--text-1)", wordBreak: "break-all" }}>{fileName}</p>
              <p style={{ fontSize: 11, color: "var(--text-3)", marginTop: 4 }}>
                {result.schema} · {result.review.length ? `${result.review.length} element(s) to review` : "nothing to review"}
              </p>
            </Card>
          </div>

          {result.elements.length === 0 ? (
            <Note tone="warning">No concrete elements were found. Check that the concrete materials are named so they can be recognised (see the rule above).</Note>
          ) : (
            <>
              {/* ── By class / mix ── */}
              <Card style={{ padding: "24px 24px 16px" }}>
                <Label style={{ marginBottom: 8 }}>By strength class</Label>
                <p style={{ fontSize: 12, color: "var(--text-2)", marginBottom: 14, lineHeight: 1.5 }}>
                  Each class gets a mix: an example mix for that class, or the mix currently set on the{" "}
                  <Link href="/">predictor</Link>. The strength model checks whether the mix reaches the class.
                </p>
                <div style={{ overflowX: "auto" }}>
                  <table className="data" id="bim-class-table">
                    <thead>
                      <tr>
                        <th>Class</th><th className="num">Elements</th><th className="num">Volume m³</th><th>Mix</th>
                        <th className="num">Mix strength</th><th>Check</th><th className="num">kg CO₂e/m³</th><th className="num">t CO₂e</th>
                      </tr>
                    </thead>
                    <tbody>
                      {groups.map((g) => (
                        <tr key={g.cls}>
                          <td style={{ color: "var(--text-1)", fontWeight: 700 }}>{g.cls}</td>
                          <td className="num">{g.count}</td>
                          <td className="num">{fmt(g.volume, 2)}</td>
                          <td>
                            {g.hasExample ? (
                              <select
                                value={g.source}
                                onChange={(e) => setSources((s) => ({ ...s, [g.cls]: e.target.value as MixSource }))}
                                style={{ background: "#14141f", color: "var(--text-1)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "4px 6px", fontSize: 12 }}
                              >
                                <option value="example">Example {g.cls} mix</option>
                                <option value="predictor">Predictor mix</option>
                              </select>
                            ) : (
                              "Predictor mix"
                            )}
                          </td>
                          <td className="num">{g.strength.toFixed(1)} MPa · {g.grade}</td>
                          <td>
                            {g.meets === null ? (
                              <span style={{ color: "var(--text-3)" }}>–</span>
                            ) : g.meets ? (
                              <span style={{ color: "var(--text-1)" }}><span aria-hidden style={{ color: "#0ca30c", fontWeight: 900 }}>✓</span> meets</span>
                            ) : (
                              <span style={{ color: "var(--text-1)" }}><span aria-hidden style={{ color: "#d03b3b", fontWeight: 900 }}>✕</span> below class</span>
                            )}
                          </td>
                          <td className="num">{fmt(g.perM3, 0)}</td>
                          <td className="num" style={{ color: "var(--text-1)", fontWeight: 700 }}>{fmt(g.carbon / 1000, 2)}</td>
                        </tr>
                      ))}
                      <tr className="strong">
                        <td>Total</td><td className="num">{result.elements.length}</td><td className="num">{fmt(totalVolume, 2)}</td>
                        <td></td><td></td><td></td><td></td><td className="num">{fmt(totalCarbon / 1000, 2)}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </Card>

              <div className="half-grid" style={{ padding: 0 }}>
                {/* ── By element type ── */}
                <Card style={{ padding: "24px" }}>
                  <Label style={{ marginBottom: 14 }}>Carbon by element type</Label>
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {byIfcClass.map(([cls, g]) => (
                      <BarRow
                        key={cls}
                        label={cls.replace(/^Ifc/, "")}
                        value={`${fmt(g.carbon / 1000, 2)} t · ${fmt(g.volume, 1)} m³`}
                        fraction={g.carbon / maxIfcCarbon}
                        title={`${cls}: ${fmt(g.carbon / 1000, 2)} t CO₂e, ${fmt(g.volume, 2)} m³`}
                      />
                    ))}
                  </div>
                </Card>

                {/* ── Review ── */}
                <Card style={{ padding: "24px" }}>
                  <Label style={{ marginBottom: 14 }}>Needs review ({result.review.length})</Label>
                  {result.review.length === 0 ? (
                    <p style={{ fontSize: 12, color: "var(--text-2)" }}>Every concrete element had a volume and a clear concrete share.</p>
                  ) : (
                    <>
                      <p style={{ fontSize: 12, color: "var(--text-2)", marginBottom: 10, lineHeight: 1.5 }}>
                        These elements look like concrete but were left out of the total, because their share or volume
                        couldn&apos;t be determined.
                      </p>
                      <ul style={{ display: "flex", flexDirection: "column", gap: 8, listStyle: "none" }}>
                        {result.review.map((r) => (
                          <li key={r.globalId} style={{ fontSize: 12, color: "var(--text-2)" }}>
                            <strong style={{ color: "var(--text-1)" }}>{r.name || r.globalId}</strong> ({r.ifcClass}) — {r.issue}
                          </li>
                        ))}
                      </ul>
                    </>
                  )}
                </Card>
              </div>

              {/* ── Elements ── */}
              <Card style={{ padding: "24px 24px 16px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, gap: 12 }}>
                  <Label>Concrete elements ({result.elements.length})</Label>
                  <button
                    id="download-csv"
                    onClick={downloadCsv}
                    style={{
                      fontSize: 11, fontWeight: 600, padding: "6px 14px", borderRadius: 10, cursor: "pointer",
                      background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.12)", color: "var(--text-1)",
                    }}
                  >
                    Download CSV
                  </button>
                </div>
                <div style={{ maxHeight: 420, overflow: "auto" }}>
                  <table className="data">
                    <thead>
                      <tr><th>Element</th><th>Type</th><th>Material</th><th>Class</th><th className="num">m³</th><th>Volume from</th><th className="num">kg CO₂e</th></tr>
                    </thead>
                    <tbody>
                      {result.elements.map((e) => (
                        <tr key={e.globalId || e.expressId}>
                          <td style={{ color: "var(--text-1)" }}>{e.name || e.globalId}</td>
                          <td>{e.ifcClass.replace(/^Ifc/, "")}</td>
                          <td>{e.material}{e.note && <span style={{ color: "var(--text-3)" }}> · {e.note}</span>}</td>
                          <td>{e.strengthClass ?? "–"}</td>
                          <td className="num">{e.volume.toFixed(3)}</td>
                          <td>{e.volumeSource === "geometry" ? "geometry" : "quantity set"}</td>
                          <td className="num">{fmt(e.volume * (perM3ByClass[e.strengthClass ?? NO_CLASS] ?? 0), 0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              <Note>
                Concrete only (A1–A3, generic emission factors). Reinforcement, formwork and transport are not included, and
                the result is only as complete as the model&apos;s materials. <Link href="/model#carbon">How carbon is calculated</Link>
              </Note>
            </>
          )}
        </>
      )}
    </div>
  );
}
