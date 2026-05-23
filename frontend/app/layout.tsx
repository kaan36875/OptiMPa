import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "OptiMPa — Concrete Strength Predictor",
  description:
    "AI-powered concrete compressive strength prediction. Enter your mix design parameters and get an instant MPa estimate powered by a Random Forest model trained on UCI data.",
  keywords: ["concrete", "compressive strength", "MPa", "mix design", "civil engineering", "machine learning"],
  authors: [{ name: "OptiMPa" }],
  openGraph: {
    title: "OptiMPa — Concrete Strength Predictor",
    description: "Predict concrete compressive strength from mix design parameters.",
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
      <body className="antialiased">{children}</body>
    </html>
  );
}
