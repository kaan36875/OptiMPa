import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Nav from "./components/Nav";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "OptiMPa — Concrete Strength & Carbon",
  description:
    "Predict concrete compressive strength from the mix design, see what drives it (SHAP), estimate the mix's embodied carbon and the carbon of the concrete in a BIM (IFC) model.",
  keywords: ["concrete", "compressive strength", "embodied carbon", "mix design", "BIM", "IFC", "machine learning"],
  authors: [{ name: "OptiMPa" }],
  openGraph: {
    title: "OptiMPa — Concrete Strength & Carbon",
    description: "Concrete strength prediction, embodied carbon and BIM quantities in the browser.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="antialiased">
        <Nav />
        <main style={{ position: "relative", zIndex: 1, paddingBottom: 64 }}>{children}</main>
        <footer
          style={{
            position: "relative", zIndex: 1,
            textAlign: "center", padding: "24px",
            borderTop: "1px solid rgba(255,255,255,0.05)",
          }}
        >
          <p style={{ fontSize: 11, color: "var(--text-3)" }}>
            OptiMPa · XGBoost model trained on UCI Concrete Compressive Strength (I-Cheng Yeh, 1998) · runs entirely in your browser ·{" "}
            <a href="https://github.com/kaan36875/OptiMPa">source on GitHub</a>
          </p>
        </footer>
      </body>
    </html>
  );
}
