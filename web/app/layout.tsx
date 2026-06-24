import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "EarningsLens | Earnings Call Sentiment Analyzer",
  description:
    "Static-first financial NLP dashboard for FinBERT, Loughran-McDonald sentiment analysis, speaker-aware transcript intelligence, and event-study market reaction analytics.",
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
