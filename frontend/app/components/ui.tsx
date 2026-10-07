import type { CSSProperties, ReactNode } from "react";

/** Glass card used throughout the site */
export function Card({ children, style }: { children: ReactNode; style?: CSSProperties }) {
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

/** Small uppercase section label */
export function Label({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <p
      style={{
        fontSize: 10, fontWeight: 700, letterSpacing: "0.18em", textTransform: "uppercase",
        color: "var(--text-3)", ...style,
      }}
    >
      {children}
    </p>
  );
}

/** Label / value / caption stack used in stat rows */
export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div style={{ textAlign: "center" }}>
      <p style={{ fontSize: 10, color: "var(--text-3)", marginBottom: 4, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.1em" }}>
        {label}
      </p>
      <p style={{ fontSize: 14, fontWeight: 800, color: "var(--text-1)" }}>{value}</p>
      {sub && <p style={{ fontSize: 10, color: "var(--text-3)" }}>{sub}</p>}
    </div>
  );
}

/**
 * One row of a single-series horizontal bar list: label + value on top, bar below.
 * Text stays in text colours; the bar alone carries the series colour.
 */
export function BarRow({
  label, value, fraction, title, color = "var(--series-1)",
}: {
  label: ReactNode; value: ReactNode; fraction: number; title?: string; color?: string;
}) {
  return (
    <div title={title} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", fontSize: 11, gap: 8 }}>
        <span style={{ fontWeight: 600, color: "var(--text-1)" }}>{label}</span>
        <span style={{ color: "var(--text-2)", fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap" }}>{value}</span>
      </div>
      <div style={{ height: 8, background: "rgba(255,255,255,0.04)", borderRadius: 4 }}>
        <div
          style={{
            width: `${Math.max(0, Math.min(1, fraction)) * 100}%`,
            minWidth: fraction > 0 ? 3 : 0,
            height: "100%",
            background: color,
            borderRadius: "0 4px 4px 0",
          }}
        />
      </div>
    </div>
  );
}

/** Grey info note */
export function Note({ children, tone = "muted" }: { children: ReactNode; tone?: "muted" | "warning" }) {
  const warning = tone === "warning";
  return (
    <p
      style={{
        fontSize: 11, lineHeight: 1.5, padding: "10px 12px", borderRadius: 10,
        color: warning ? "var(--text-1)" : "var(--text-2)",
        background: warning ? "rgba(250,178,25,0.08)" : "rgba(255,255,255,0.03)",
        border: `1px solid ${warning ? "rgba(250,178,25,0.35)" : "rgba(255,255,255,0.06)"}`,
      }}
    >
      {warning && <span aria-hidden style={{ color: "#fab219", fontWeight: 800, marginRight: 6 }}>!</span>}
      {children}
    </p>
  );
}

export const fmt = (x: number, digits = 1) =>
  x.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
