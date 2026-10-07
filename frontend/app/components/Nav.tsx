"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Predictor" },
  { href: "/bim", label: "BIM carbon" },
  { href: "/model", label: "Model" },
];

export default function Nav() {
  const pathname = usePathname();
  return (
    <nav
      style={{
        position: "sticky", top: 0, zIndex: 50,
        display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16,
        padding: "0 24px", height: 60,
        background: "rgba(8,8,15,0.85)",
        backdropFilter: "blur(20px)",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      <Link href="/" style={{ display: "flex", alignItems: "center", gap: 12, textDecoration: "none" }}>
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
        <span style={{ fontWeight: 800, fontSize: 15, color: "var(--text-1)", letterSpacing: "-0.02em" }}>OptiMPa</span>
      </Link>
      <div style={{ display: "flex", gap: 4 }}>
        {LINKS.map(({ href, label }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              style={{
                fontSize: 12, fontWeight: 600, padding: "7px 12px", borderRadius: 10, textDecoration: "none",
                color: active ? "var(--accent)" : "var(--text-2)",
                background: active ? "var(--accent-dim)" : "transparent",
                border: `1px solid ${active ? "var(--border-accent)" : "transparent"}`,
                whiteSpace: "nowrap",
              }}
            >
              {label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
