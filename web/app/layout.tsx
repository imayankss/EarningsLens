import type { Metadata } from "next";
import type { Viewport } from "next";

import "./globals.css";

const vercelHost = process.env.VERCEL_PROJECT_PRODUCTION_URL ?? process.env.VERCEL_URL;
const siteUrl = vercelHost ? `https://${vercelHost}` : "http://localhost:3000";

export const metadata: Metadata = {
  applicationName: "EarningsLens",
  metadataBase: new URL(siteUrl),
  title: {
    default: "EarningsLens | Financial NLP & Event-Study Analytics",
    template: "%s | EarningsLens",
  },
  description:
    "A deployment-safe financial NLP dashboard for verified FinBERT, Loughran–McDonald, transcript intelligence, and event-study outputs.",
  authors: [{ name: "Mayank Suryavanshi" }],
  creator: "Mayank Suryavanshi",
  category: "technology",
  keywords: [
    "financial NLP",
    "earnings calls",
    "FinBERT",
    "Loughran-McDonald",
    "event study",
    "machine learning",
  ],
  openGraph: {
    type: "website",
    siteName: "EarningsLens",
    title: "EarningsLens | Financial NLP & Event-Study Analytics",
    description:
      "Verified earnings-call sentiment and market-reaction outputs from a static, deployment-safe research pipeline.",
  },
  twitter: {
    card: "summary_large_image",
    title: "EarningsLens | Financial NLP & Event-Study Analytics",
    description:
      "Verified FinBERT, Loughran–McDonald, transcript, and event-study artifacts in a polished research dashboard.",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export const viewport: Viewport = {
  colorScheme: "dark",
  themeColor: "#050816",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
