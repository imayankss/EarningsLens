import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft, BrainCircuit, Database, LineChart, ShieldCheck, Workflow } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export const metadata: Metadata = {
  title: "Methodology",
  description:
    "How EarningsLens separates offline financial NLP and event-study processing from its static Vercel presentation layer.",
};

const methods = [
  {
    title: "Transcript preparation",
    description:
      "The Python pipeline cleans earnings-call text, preserves transcript metadata, and creates model-safe chunks before scoring.",
    icon: Database,
  },
  {
    title: "Two sentiment lenses",
    description:
      "FinBERT produces contextual positive, neutral, and negative probabilities. Loughran–McDonald contributes transparent financial-word counts and a tone score.",
    icon: BrainCircuit,
  },
  {
    title: "Market alignment",
    description:
      "Earnings events are aligned with observed market windows to calculate raw returns, abnormal returns, and cumulative abnormal returns.",
    icon: LineChart,
  },
  {
    title: "Guarded modeling",
    description:
      "Training and statistical inference are skipped when the sample is too small. The checked n=1 snapshot demonstrates the data contract, not predictive validity.",
    icon: ShieldCheck,
  },
];

export default function MethodologyPage() {
  return (
    <main className="min-h-screen bg-[#050816]">
      <header className="border-b border-white/10 bg-slate-950/55">
        <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-4 sm:px-6">
          <Link href="/" className="font-semibold text-white">EarningsLens</Link>
          <Button asChild size="sm" variant="outline" className="border-white/15 bg-white/5">
            <Link href="/">
              <ArrowLeft aria-hidden className="size-4" />
              Dashboard
            </Link>
          </Button>
        </div>
      </header>

      <section className="relative overflow-hidden border-b border-white/10 px-4 py-20 sm:px-6">
        <div aria-hidden className="market-grid absolute inset-0 opacity-45" />
        <div className="relative mx-auto max-w-5xl">
          <Badge variant="outline" className="border-cyan-300/25 bg-cyan-300/[0.08] font-mono text-cyan-100">Research methodology</Badge>
          <h1 className="mt-6 max-w-4xl text-4xl font-semibold tracking-tight text-white sm:text-6xl">
            A transparent boundary between research compute and the live product.
          </h1>
          <p className="mt-6 max-w-3xl text-lg leading-8 text-slate-300">
            EarningsLens preserves the original analytics pipeline while deploying only small, versioned, non-sensitive output artifacts to Vercel.
          </p>
        </div>
      </section>

      <section className="mx-auto max-w-5xl px-4 py-16 sm:px-6">
        <div className="grid gap-4 md:grid-cols-2">
          {methods.map(({ title, description, icon: Icon }) => (
            <Card key={title} className="h-full">
              <CardContent className="p-6">
                <span className="flex size-10 items-center justify-center rounded-lg border border-cyan-300/20 bg-cyan-300/[0.08]">
                  <Icon aria-hidden className="size-5 text-cyan-100" />
                </span>
                <h2 className="mt-5 text-xl font-semibold text-white">{title}</h2>
                <p className="mt-3 text-sm leading-7 text-slate-400">{description}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        <Card className="mt-8 overflow-hidden border-emerald-300/20">
          <div className="h-1 bg-gradient-to-r from-cyan-300 via-emerald-300 to-amber-300" />
          <CardContent className="p-6 sm:p-8">
            <div className="flex items-center gap-3">
              <Workflow aria-hidden className="size-6 text-emerald-200" />
              <h2 className="text-xl font-semibold text-white">Deployment contract</h2>
            </div>
            <ol className="mt-6 grid gap-4 text-sm leading-7 text-slate-300 sm:grid-cols-2">
              <li className="rounded-lg border border-white/10 bg-white/[0.035] p-4"><strong className="text-white">1. Run offline analysis.</strong><br />Transcript ingestion, FinBERT inference, financial dictionary scoring, and market alignment run in Python outside web requests.</li>
              <li className="rounded-lg border border-white/10 bg-white/[0.035] p-4"><strong className="text-white">2. Export checked JSON.</strong><br /><code className="font-mono text-cyan-100">python3 scripts/export_web_data.py</code> serializes existing artifacts and rejects non-finite JSON values.</li>
              <li className="rounded-lg border border-white/10 bg-white/[0.035] p-4"><strong className="text-white">3. Validate at build time.</strong><br />The Next.js build checks the JSON schema before producing the static application.</li>
              <li className="rounded-lg border border-white/10 bg-white/[0.035] p-4"><strong className="text-white">4. Serve on Vercel.</strong><br />Normal requests load static JavaScript, CSS, and JSON—never transformer weights or training jobs.</li>
            </ol>
          </CardContent>
        </Card>

        <div className="mt-8 rounded-lg border border-amber-300/20 bg-amber-300/[0.06] p-5 text-sm leading-7 text-slate-300">
          <strong className="text-amber-100">Current evidence limit:</strong> the repository contains one verified AAPL transcript/event. Speaker labels are absent from the checked speaker summary, and no predictive model is trained. The dashboard exposes those limitations instead of creating substitute values.
        </div>
      </section>
    </main>
  );
}
