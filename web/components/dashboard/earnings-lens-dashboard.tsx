"use client";

import { motion } from "framer-motion";
import {
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Code2,
  Cpu,
  Database,
  FileText,
  GitBranch,
  Layers3,
  LibraryBig,
  LineChart as LineChartIcon,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Workflow,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { SectionShell } from "@/components/dashboard/section-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { pipelineStages, repositoryLinks } from "@/data/dashboard-data";
import { cn } from "@/lib/utils";
import type { AnalysisResult, DashboardData } from "@/types/dashboard";

const DATA_URL = "/data/dashboard.json";

const chartColors = {
  cyan: "#38bdf8",
  emerald: "#22c55e",
  rose: "#f43f5e",
  slate: "#94a3b8",
  amber: "#f59e0b",
  violet: "#8b5cf6",
};

type TooltipPayload = {
  color?: string;
  name?: string;
  value?: number | string;
};

function isDashboardData(value: unknown): value is DashboardData {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<DashboardData>;
  return (
    candidate.schemaVersion === 1 &&
    Array.isArray(candidate.analyses) &&
    Boolean(candidate.dataset) &&
    Boolean(candidate.modeling)
  );
}

function formatInteger(value: number | null | undefined): string {
  return value === null || value === undefined
    ? "Not available"
    : new Intl.NumberFormat("en-US").format(value);
}

function formatPercent(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined
    ? "Not available"
    : new Intl.NumberFormat("en-US", {
        style: "percent",
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      }).format(value);
}

function formatScore(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined ? "Not available" : value.toFixed(digits);
}

function formatDate(value: string | null | undefined): string {
  if (!value) return "Not available";
  const dateOnly = value.slice(0, 10);
  const date = new Date(`${dateOnly}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(date);
}

function titleCase(value: string | null | undefined): string {
  if (!value) return "Not available";
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function ChartTooltip({
  active,
  payload,
  label,
  suffix = "",
}: {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string | number;
  suffix?: string;
}) {
  if (!active || !payload?.length) return null;

  return (
    <div className="rounded-lg border border-white/10 bg-slate-950/95 px-3 py-2 text-xs shadow-glow backdrop-blur-xl">
      {label !== undefined ? <p className="mb-1 font-medium text-slate-200">{label}</p> : null}
      <div className="space-y-1">
        {payload.map((item) => {
          const value =
            typeof item.value === "number" ? `${item.value.toFixed(2)}${suffix}` : item.value;
          return (
            <div key={`${item.name}-${item.value}`} className="flex items-center gap-2">
              <span className="size-2 rounded-full" style={{ backgroundColor: item.color }} />
              <span className="text-slate-400">{item.name}</span>
              <span className="font-mono text-white">{value}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ChartPanel({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.15 }}
      transition={{ duration: 0.45 }}
      className="h-full"
    >
      <Card className="h-full overflow-hidden">
        <div className="h-1 bg-gradient-to-r from-cyan-400 via-violet-400 to-emerald-400" />
        <CardHeader>
          <CardTitle className="text-white">{title}</CardTitle>
          <CardDescription className="leading-6">{description}</CardDescription>
        </CardHeader>
        <CardContent>{children}</CardContent>
      </Card>
    </motion.div>
  );
}

function MetricCard({
  label,
  value,
  detail,
  icon: Icon,
  tone = "cyan",
}: {
  label: string;
  value: string;
  detail: string;
  icon: typeof FileText;
  tone?: "cyan" | "emerald" | "amber" | "slate";
}) {
  const styles = {
    cyan: "border-cyan-400/20 bg-cyan-400/[0.06] text-cyan-200",
    emerald: "border-emerald-400/20 bg-emerald-400/[0.06] text-emerald-200",
    amber: "border-amber-400/20 bg-amber-400/[0.06] text-amber-200",
    slate: "border-slate-400/15 bg-slate-400/[0.05] text-slate-200",
  };

  return (
    <Card className={cn("h-full", styles[tone])}>
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-slate-400">{label}</p>
            <p className="mt-3 text-2xl font-semibold text-white">{value}</p>
          </div>
          <Icon aria-hidden className="size-5 shrink-0" />
        </div>
        <p className="mt-3 text-sm leading-6 text-slate-400">{detail}</p>
      </CardContent>
    </Card>
  );
}

function EmptyPanel({ title, message }: { title: string; message: string }) {
  return (
    <div className="flex min-h-64 flex-col items-center justify-center rounded-lg border border-dashed border-white/15 bg-white/[0.025] p-8 text-center">
      <CircleAlert aria-hidden className="size-7 text-amber-300" />
      <p className="mt-4 font-medium text-white">{title}</p>
      <p className="mt-2 max-w-md text-sm leading-6 text-slate-400">{message}</p>
    </div>
  );
}

function SiteHeader() {
  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[#050816]/90 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <a href="#top" className="flex items-center gap-2 font-semibold text-white">
          <span className="flex size-8 items-center justify-center rounded-lg border border-cyan-300/30 bg-cyan-300/10">
            <Sparkles aria-hidden className="size-4 text-cyan-200" />
          </span>
          EarningsLens
        </a>
        <nav aria-label="Primary navigation" className="hidden items-center gap-6 text-sm text-slate-300 md:flex">
          <a className="transition hover:text-white" href="#results">Results</a>
          <a className="transition hover:text-white" href="#market-reaction">Market reaction</a>
          <a className="transition hover:text-white" href="#architecture">Architecture</a>
          <Link className="transition hover:text-white" href="/methodology">Methodology</Link>
        </nav>
        <Button asChild size="sm" variant="outline" className="border-white/15 bg-white/5">
          <a href="https://github.com/imayankss/EarningsLens" target="_blank" rel="noreferrer">
            <GitBranch aria-hidden className="size-4" />
            <span className="hidden sm:inline">Source</span>
          </a>
        </Button>
      </div>
    </header>
  );
}

function LoadingDashboard() {
  return (
    <main className="min-h-screen bg-[#050816]">
      <SiteHeader />
      <div className="mx-auto flex min-h-[70vh] max-w-7xl items-center px-4 sm:px-6 lg:px-8" role="status" aria-live="polite">
        <div className="w-full rounded-2xl border border-white/10 bg-slate-950/55 p-8">
          <div className="h-4 w-36 animate-pulse rounded bg-cyan-300/15" />
          <div className="mt-6 h-12 max-w-2xl animate-pulse rounded bg-white/10" />
          <div className="mt-4 h-5 max-w-xl animate-pulse rounded bg-white/[0.06]" />
          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[0, 1, 2, 3].map((item) => (
              <div key={item} className="h-36 animate-pulse rounded-lg border border-white/10 bg-white/[0.04]" />
            ))}
          </div>
          <span className="sr-only">Loading verified EarningsLens data.</span>
        </div>
      </div>
    </main>
  );
}

function ErrorDashboard({ message, retry }: { message: string; retry: () => void }) {
  return (
    <main className="min-h-screen bg-[#050816]">
      <SiteHeader />
      <div className="mx-auto flex min-h-[70vh] max-w-3xl items-center px-4 text-center">
        <Card className="w-full border-rose-400/20">
          <CardContent className="p-8">
            <CircleAlert aria-hidden className="mx-auto size-9 text-rose-300" />
            <h1 className="mt-5 text-2xl font-semibold text-white">The verified data snapshot could not be loaded.</h1>
            <p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-slate-400">{message}</p>
            <Button className="mt-6" onClick={retry}>
              <RefreshCw aria-hidden className="size-4" />
              Retry data request
            </Button>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}

function Hero({ data, analysis }: { data: DashboardData; analysis: AnalysisResult }) {
  const transcript = analysis.transcript;
  const agreement = analysis.comparison.directionalAgreement;

  return (
    <section id="top" className="relative overflow-hidden border-b border-white/10 px-4 pb-14 pt-12 sm:px-6 lg:px-8 lg:pb-20 lg:pt-20">
      <div aria-hidden className="market-grid absolute inset-0 opacity-65" />
      <div aria-hidden className="absolute inset-x-0 top-0 h-56 bg-gradient-to-b from-cyan-400/10 to-transparent" />
      <div className="relative mx-auto grid max-w-7xl gap-10 lg:grid-cols-[1.08fr_0.92fr] lg:items-center">
        <div>
          <div className="flex flex-wrap gap-3">
            <Badge variant="outline" className="border-amber-300/30 bg-amber-300/10 font-mono text-amber-100">
              Verified {data.dataset.kind} dataset · n={data.dataset.observationCount}
            </Badge>
            <Badge variant="outline" className="border-emerald-300/30 bg-emerald-300/10 font-mono text-emerald-100">
              Precomputed pipeline output
            </Badge>
          </div>
          <motion.h1
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-7 max-w-4xl text-5xl font-semibold leading-[0.98] tracking-tight text-white sm:text-6xl lg:text-7xl"
          >
            Earnings-call language, translated into market-aware signals.
          </motion.h1>
          <p className="mt-6 max-w-3xl text-base leading-8 text-slate-300 sm:text-lg">
            EarningsLens connects FinBERT, the Loughran–McDonald financial dictionary, advanced transcript features, and event-study outputs in a deployment-safe research dashboard.
          </p>
          <div className="mt-7 rounded-lg border border-amber-300/20 bg-amber-300/[0.07] p-4 text-sm leading-6 text-amber-50">
            <strong className="font-semibold">Dataset scope:</strong> {data.dataset.notice}
          </div>
          <div className="mt-8 flex flex-wrap gap-3">
            <Button asChild size="lg" className="bg-cyan-300 text-slate-950 hover:bg-cyan-200">
              <a href="#results">
                Explore verified results
                <ArrowRight aria-hidden className="size-4" />
              </a>
            </Button>
            <Button asChild size="lg" variant="outline" className="border-white/15 bg-white/5">
              <Link href="/methodology">Read methodology</Link>
            </Button>
          </div>
        </div>

        <motion.div initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.12 }}>
          <Card className="overflow-hidden border-cyan-300/20 bg-slate-950/75">
            <div className="h-1 bg-gradient-to-r from-cyan-300 via-emerald-300 to-amber-300" />
            <CardHeader>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="font-mono text-xs uppercase tracking-[0.16em] text-cyan-200">Verified analysis</p>
                  <CardTitle className="mt-2 text-2xl text-white">
                    {transcript.companyName ?? transcript.ticker ?? transcript.id}
                  </CardTitle>
                  <CardDescription className="mt-2">
                    {transcript.ticker} · Q{transcript.fiscalQuarter} {transcript.fiscalYear} · {formatDate(transcript.earningsDate)}
                  </CardDescription>
                </div>
                <span className="sentiment-pulse rounded-full border border-emerald-300/25 bg-emerald-300/10 p-3">
                  <BrainCircuit aria-hidden className="size-6 text-emerald-200" />
                </span>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 sm:grid-cols-3">
                <div className="rounded-lg border border-white/10 bg-white/[0.04] p-4">
                  <p className="font-mono text-[11px] uppercase text-slate-400">FinBERT</p>
                  <p className="mt-2 text-xl font-semibold text-emerald-200">{titleCase(analysis.finbert.direction)}</p>
                  <p className="mt-1 text-xs text-slate-500">score {formatScore(analysis.finbert.score)}</p>
                </div>
                <div className="rounded-lg border border-white/10 bg-white/[0.04] p-4">
                  <p className="font-mono text-[11px] uppercase text-slate-400">LM label</p>
                  <p className="mt-2 text-xl font-semibold text-amber-200">{titleCase(analysis.loughranMcDonald.label)}</p>
                  <p className="mt-1 text-xs text-slate-500">tone {formatScore(analysis.loughranMcDonald.toneScore)}</p>
                </div>
                <div className="rounded-lg border border-white/10 bg-white/[0.04] p-4">
                  <p className="font-mono text-[11px] uppercase text-slate-400">Agreement</p>
                  <p className="mt-2 text-xl font-semibold text-cyan-200">
                    {agreement === null ? "Unavailable" : agreement ? "Aligned" : "Divergent"}
                  </p>
                  <p className="mt-1 text-xs text-slate-500">directional labels</p>
                </div>
              </div>
              <div className="mt-5 flex items-center gap-2 rounded-lg border border-emerald-300/15 bg-emerald-300/[0.05] p-3 text-sm text-slate-300">
                <ShieldCheck aria-hidden className="size-4 shrink-0 text-emerald-200" />
                Values are loaded from the checked JSON export—not generated in the browser.
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      <div className="relative mx-auto mt-10 grid max-w-7xl gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="Verified events" value={formatInteger(data.dataset.observationCount)} detail={`${data.dataset.companyCount} company in the current artifact snapshot`} icon={Database} tone="cyan" />
        <MetricCard label="Transcript chunks" value={formatInteger(transcript.chunkCount)} detail="Model-safe units aggregated into transcript-level sentiment" icon={Layers3} tone="emerald" />
        <MetricCard label="Clean words" value={formatInteger(transcript.cleanWordCount)} detail="Words retained by the checked preprocessing pipeline" icon={FileText} tone="slate" />
        <MetricCard label="Mean confidence" value={formatPercent(analysis.finbert.meanConfidence)} detail="Average confidence across the exported FinBERT analysis" icon={CheckCircle2} tone="amber" />
      </div>
    </section>
  );
}

function Results({ analysis }: { analysis: AnalysisResult }) {
  const sentimentDistribution = [
    { name: "Positive", value: analysis.finbert.positiveProbability, color: chartColors.emerald },
    { name: "Neutral", value: analysis.finbert.neutralProbability, color: chartColors.slate },
    { name: "Negative", value: analysis.finbert.negativeProbability, color: chartColors.rose },
  ].flatMap((item) => (item.value === null ? [] : [{ ...item, value: item.value * 100 }]));

  const lmCounts = [
    { name: "Positive", value: analysis.loughranMcDonald.positiveCount ?? 0, fill: chartColors.emerald },
    { name: "Negative", value: analysis.loughranMcDonald.negativeCount ?? 0, fill: chartColors.rose },
  ];

  return (
    <SectionShell
      eyebrow="Verified model output"
      title="Contextual sentiment and dictionary tone, side by side."
      description="Every number in this section is exported from the checked AAPL transcript artifact. The two methods agree on direction, while preserving their distinct scoring systems."
      className="scroll-mt-20"
    >
      <div id="results" className="grid scroll-mt-24 gap-6 lg:grid-cols-2">
        <ChartPanel title="FinBERT probability mix" description="Transcript-level positive, neutral, and negative probabilities.">
          {sentimentDistribution.length ? (
            <div className="h-72" role="img" aria-label="FinBERT sentiment probability donut chart">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={sentimentDistribution} dataKey="value" nameKey="name" innerRadius={68} outerRadius={104} paddingAngle={4}>
                    {sentimentDistribution.map((item) => <Cell key={item.name} fill={item.color} />)}
                  </Pie>
                  <Tooltip content={<ChartTooltip suffix="%" />} />
                  <Legend iconType="circle" />
                </PieChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyPanel title="FinBERT probabilities unavailable" message="Regenerate the web data after producing transcript-level sentiment artifacts." />}
        </ChartPanel>

        <ChartPanel title="Loughran–McDonald polarity counts" description="Explainable finance-dictionary matches in the same verified transcript.">
          <div className="h-72" role="img" aria-label="Loughran-McDonald positive and negative word-count bar chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={lmCounts}>
                <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" vertical={false} />
                <XAxis dataKey="name" stroke="#94a3b8" tickLine={false} axisLine={false} />
                <YAxis stroke="#64748b" tickLine={false} axisLine={false} allowDecimals={false} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="value" name="Word matches" radius={[7, 7, 0, 0]}>
                  {lmCounts.map((item) => <Cell key={item.name} fill={item.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartPanel>
      </div>

      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="FinBERT score" value={formatScore(analysis.finbert.score)} detail="Positive probability minus negative probability" icon={BrainCircuit} tone="cyan" />
        <MetricCard label="LM tone score" value={formatScore(analysis.loughranMcDonald.toneScore)} detail={`${formatInteger(analysis.loughranMcDonald.positiveCount)} positive and ${formatInteger(analysis.loughranMcDonald.negativeCount)} negative matches`} icon={LibraryBig} tone="amber" />
        <MetricCard label="Score gap" value={formatScore(analysis.comparison.absoluteDifference)} detail="Absolute difference between the two transcript-level scores" icon={BarChart3} tone="slate" />
        <MetricCard label="Directional result" value={analysis.comparison.directionalAgreement ? "Agreement" : "No agreement"} detail="Both exported labels are positive for this transcript" icon={CheckCircle2} tone="emerald" />
      </div>
    </SectionShell>
  );
}

function TranscriptIntelligence({ analysis }: { analysis: AnalysisResult }) {
  const topics = analysis.nlp.topics
    .filter((topic) => (topic.count ?? 0) > 0)
    .map((topic) => ({ ...topic, count: topic.count ?? 0 }));
  const aggregate = analysis.speakerAnalysis.aggregate;

  return (
    <SectionShell
      eyebrow="Transcript intelligence"
      title="What the call discussed—and what the current export cannot claim."
      description="Topic counts, uncertainty, and keywords come from the real advanced-NLP artifact. Speaker-level labels are surfaced only when the pipeline actually exports them."
    >
      <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <ChartPanel title="Finance topic frequency" description="Matched topic terms across the verified transcript; categories can overlap.">
          {topics.length ? (
            <div className="h-80" role="img" aria-label="Financial topic-frequency horizontal bar chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topics} layout="vertical" margin={{ left: 16 }}>
                  <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" horizontal={false} />
                  <XAxis type="number" stroke="#64748b" tickLine={false} axisLine={false} allowDecimals={false} />
                  <YAxis type="category" dataKey="name" width={128} stroke="#94a3b8" tickLine={false} axisLine={false} tick={{ fontSize: 12 }} />
                  <Tooltip content={<ChartTooltip />} />
                  <Bar dataKey="count" name="Term matches" fill={chartColors.cyan} radius={[0, 7, 7, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyPanel title="Topic data unavailable" message="The current JSON export contains no topic-frequency rows." />}
        </ChartPanel>

        <Card className="h-full">
          <CardHeader>
            <div className="flex items-center justify-between gap-4">
              <div>
                <CardTitle className="text-white">Language fingerprint</CardTitle>
                <CardDescription className="mt-2">Checked advanced-NLP features.</CardDescription>
              </div>
              <Code2 aria-hidden className="size-6 text-cyan-200" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
              <div className="rounded-lg border border-white/10 bg-white/[0.035] p-4">
                <p className="font-mono text-xs uppercase text-slate-400">Tokens analyzed</p>
                <p className="mt-2 text-2xl font-semibold text-white">{formatInteger(analysis.nlp.totalTokens)}</p>
              </div>
              <div className="rounded-lg border border-white/10 bg-white/[0.035] p-4">
                <p className="font-mono text-xs uppercase text-slate-400">Uncertainty</p>
                <p className="mt-2 text-2xl font-semibold text-amber-200">{formatInteger(analysis.nlp.uncertaintyCount)} hits</p>
                <p className="mt-1 text-xs text-slate-500">{formatPercent(analysis.nlp.uncertaintyRatio, 2)} of tokens</p>
              </div>
            </div>
            <div className="mt-6">
              <p className="text-sm font-medium text-white">Top exported keywords</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {analysis.nlp.topKeywords.length ? analysis.nlp.topKeywords.map((keyword) => (
                  <Badge key={keyword} variant="outline" className="border-cyan-300/20 bg-cyan-300/[0.07] font-mono text-cyan-100">
                    {keyword}
                  </Badge>
                )) : <span className="text-sm text-slate-400">No keyword artifact available.</span>}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="mt-6 border-amber-300/20">
        <CardContent className="grid gap-6 p-6 md:grid-cols-[1fr_auto] md:items-center">
          <div>
            <div className="flex items-center gap-2">
              <CircleAlert aria-hidden className="size-5 text-amber-200" />
              <h3 className="font-semibold text-white">Speaker-level availability</h3>
            </div>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">{analysis.speakerAnalysis.message}</p>
            <p className="mt-2 text-sm text-slate-500">EarningsLens does not infer CEO, CFO, analyst, or section values when labels are absent.</p>
          </div>
          {aggregate ? (
            <div className="grid min-w-72 grid-cols-2 gap-3">
              <div className="rounded-lg border border-white/10 bg-white/[0.035] p-3">
                <p className="font-mono text-[11px] uppercase text-slate-500">Aggregate score</p>
                <p className="mt-2 text-xl font-semibold text-white">{formatScore(aggregate.averageSentimentScore)}</p>
              </div>
              <div className="rounded-lg border border-white/10 bg-white/[0.035] p-3">
                <p className="font-mono text-[11px] uppercase text-slate-500">Chunk mix</p>
                <p className="mt-2 text-sm font-medium text-white">
                  {formatInteger(aggregate.positiveChunks)} positive · {formatInteger(aggregate.neutralChunks)} neutral
                </p>
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </SectionShell>
  );
}

function MarketReaction({ analysis }: { analysis: AnalysisResult }) {
  const marketData = analysis.marketReaction.windows.map((window) => ({
    label: window.label,
    "Raw return": window.rawReturn === null ? null : window.rawReturn * 100,
    "Abnormal return": window.abnormalReturn === null ? null : window.abnormalReturn * 100,
  }));

  return (
    <SectionShell
      eyebrow="Event study"
      title="Post-earnings returns are shown as measured windows, not a prediction."
      description="Raw and abnormal returns are exported from the market-alignment layer. CAR values are displayed separately because they use the event-study pipeline’s additive definition."
      className="scroll-mt-20"
    >
      <div id="market-reaction" className="grid scroll-mt-24 gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <ChartPanel title="Return horizons" description="Observed forward-return and abnormal-return values for the verified event.">
          {marketData.length ? (
            <div className="h-80" role="img" aria-label="Raw and abnormal return by post-earnings horizon bar chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={marketData}>
                  <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" vertical={false} />
                  <XAxis dataKey="label" stroke="#94a3b8" tickLine={false} axisLine={false} />
                  <YAxis stroke="#64748b" tickLine={false} axisLine={false} tickFormatter={(value) => `${value}%`} />
                  <Tooltip content={<ChartTooltip suffix="%" />} />
                  <Legend />
                  <Bar dataKey="Raw return" fill={chartColors.slate} radius={[6, 6, 0, 0]} />
                  <Bar dataKey="Abnormal return" fill={chartColors.cyan} radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyPanel title="Market windows unavailable" message="No return horizons were present in the checked master dataset." />}
        </ChartPanel>

        <Card>
          <CardHeader>
            <CardTitle className="text-white">Event-study snapshot</CardTitle>
            <CardDescription>Verified event and cumulative abnormal returns.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-lg border border-white/10 bg-white/[0.035] p-4">
              <p className="font-mono text-xs uppercase text-slate-500">Aligned market event</p>
              <p className="mt-2 text-xl font-semibold text-white">{formatDate(analysis.marketReaction.eventDate)}</p>
              <p className="mt-2 text-xs text-slate-500">
                Source window {formatDate(analysis.marketReaction.marketWindowStart)} – {formatDate(analysis.marketReaction.marketWindowEnd)}
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
              {analysis.marketReaction.cumulativeAbnormalReturns.map((item) => (
                <div key={item.label} className="rounded-lg border border-emerald-300/20 bg-emerald-300/[0.06] p-4">
                  <p className="font-mono text-xs uppercase text-emerald-100">{item.label}</p>
                  <p className="mt-2 text-2xl font-semibold text-white">{formatPercent(item.value, 2)}</p>
                </div>
              ))}
            </div>
            <div className="rounded-lg border border-amber-300/20 bg-amber-300/[0.06] p-4 text-sm leading-6 text-slate-300">
              A single event cannot establish statistical significance, causality, or an investable relationship. These values demonstrate the pipeline contract only.
            </div>
          </CardContent>
        </Card>
      </div>
    </SectionShell>
  );
}

function ArchitectureAndModeling({ data }: { data: DashboardData }) {
  return (
    <SectionShell
      eyebrow="Deployment architecture"
      title="Heavy research runs offline; the web request stays lightweight."
      description="The Vercel application serves a static Next.js interface and a small JSON artifact. FinBERT inference, transcript processing, market downloads, and training remain explicit Python batch jobs."
      className="scroll-mt-20"
    >
      <div id="architecture" className="grid scroll-mt-24 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {pipelineStages.map((stage, index) => (
          <motion.div key={stage.stage} initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: index * 0.04 }}>
            <Card className="h-full">
              <CardContent className="p-5">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs text-cyan-200">{stage.stage}</span>
                  <Badge variant="outline" className="border-white/10 bg-white/[0.04] font-mono text-slate-300">{stage.runtime}</Badge>
                </div>
                <h3 className="mt-5 font-semibold text-white">{stage.title}</h3>
                <p className="mt-3 text-sm leading-6 text-slate-400">{stage.description}</p>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-[0.82fr_1.18fr]">
        <Card className="border-violet-300/20">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-white">Model readiness</CardTitle>
                <CardDescription className="mt-2">Guarded by the actual sample size.</CardDescription>
              </div>
              <Cpu aria-hidden className="size-6 text-violet-200" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border border-white/10 bg-white/[0.035] p-4">
                <p className="font-mono text-[11px] uppercase text-slate-500">Features</p>
                <p className="mt-2 text-2xl font-semibold text-white">{formatInteger(data.modeling.featureCount)}</p>
              </div>
              <div className="rounded-lg border border-white/10 bg-white/[0.035] p-4">
                <p className="font-mono text-[11px] uppercase text-slate-500">Models trained</p>
                <p className="mt-2 text-2xl font-semibold text-white">{data.modeling.modelsTrained}</p>
              </div>
            </div>
            <div className="mt-4 rounded-lg border border-amber-300/20 bg-amber-300/[0.06] p-4">
              <p className="font-medium text-amber-100">{titleCase(data.modeling.status)}</p>
              <p className="mt-2 text-sm leading-6 text-slate-300">{data.modeling.message}</p>
              <p className="mt-2 text-xs text-slate-500">{data.modeling.modelsAttempted} guarded model/target combinations recorded.</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between gap-4">
              <div>
                <CardTitle className="text-white">Artifact provenance</CardTitle>
                <CardDescription className="mt-2">Repository sources used by the deployment export.</CardDescription>
              </div>
              <Workflow aria-hidden className="size-6 text-cyan-200" />
            </div>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Layer</TableHead>
                  <TableHead>Checked source</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.provenance.map((item) => (
                  <TableRow key={item.path}>
                    <TableCell className="font-medium text-white">{item.label}</TableCell>
                    <TableCell className="break-all font-mono text-xs text-slate-400">{item.path}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </SectionShell>
  );
}

function Footer() {
  return (
    <footer className="border-t border-white/10 bg-slate-950/60">
      <div className="mx-auto grid max-w-7xl gap-8 px-4 py-12 sm:px-6 md:grid-cols-[1fr_auto] md:items-end lg:px-8">
        <div>
          <div className="flex items-center gap-2 text-lg font-semibold text-white">
            <LineChartIcon aria-hidden className="size-5 text-cyan-200" />
            EarningsLens
          </div>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400">
            A financial NLP research platform. Demonstration outputs are descriptive and are not investment advice.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          {repositoryLinks.map((link) => (
            <a key={link.href} href={link.href} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-sm text-slate-300 transition hover:text-white">
              {link.label}
              <ArrowUpRight aria-hidden className="size-3.5" />
            </a>
          ))}
        </div>
      </div>
    </footer>
  );
}

function DashboardContent({ data }: { data: DashboardData }) {
  const [selectedId, setSelectedId] = useState(data.analyses[0]?.transcript.id ?? "");
  const analysis = useMemo(
    () => data.analyses.find((item) => item.transcript.id === selectedId) ?? data.analyses[0],
    [data.analyses, selectedId],
  );

  if (!analysis) {
    return <ErrorDashboard message="The JSON artifact is valid but contains no analysis rows." retry={() => window.location.reload()} />;
  }

  return (
    <main className="min-h-screen overflow-hidden bg-[#050816]">
      <SiteHeader />
      {data.analyses.length > 1 ? (
        <div className="mx-auto max-w-7xl px-4 pt-6 sm:px-6 lg:px-8">
          <label htmlFor="analysis-selector" className="text-sm font-medium text-slate-200">Analysis</label>
          <select id="analysis-selector" value={selectedId} onChange={(event) => setSelectedId(event.target.value)} className="ml-3 rounded-md border border-white/15 bg-slate-950 px-3 py-2 text-sm text-white">
            {data.analyses.map((item) => <option key={item.transcript.id} value={item.transcript.id}>{item.transcript.ticker ?? item.transcript.id}</option>)}
          </select>
        </div>
      ) : null}
      <Hero data={data} analysis={analysis} />
      <Results analysis={analysis} />
      <TranscriptIntelligence analysis={analysis} />
      <MarketReaction analysis={analysis} />
      <ArchitectureAndModeling data={data} />
      <section className="mx-auto max-w-7xl px-4 pb-16 sm:px-6 lg:px-8">
        <Card className="overflow-hidden border-cyan-300/20">
          <CardContent className="grid gap-6 p-6 md:grid-cols-[1fr_auto] md:items-center lg:p-8">
            <div>
              <Badge variant="outline" className="border-cyan-300/25 bg-cyan-300/[0.08] font-mono text-cyan-100">Inspect the implementation</Badge>
              <h2 className="mt-4 text-2xl font-semibold text-white">Research code, deployment contract, and methodology are public.</h2>
              <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">Regenerate the JSON after an offline pipeline run, commit the artifact, and let the connected Vercel project publish the updated dashboard.</p>
            </div>
            <Button asChild size="lg" className="bg-cyan-300 text-slate-950 hover:bg-cyan-200">
              <a href="https://github.com/imayankss/EarningsLens" target="_blank" rel="noreferrer">
                View repository
                <ChevronRight aria-hidden className="size-4" />
              </a>
            </Button>
          </CardContent>
        </Card>
      </section>
      <Footer />
    </main>
  );
}

export function EarningsLensDashboard() {
  const [requestKey, setRequestKey] = useState(0);
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadData() {
      try {
        const response = await fetch(DATA_URL, { signal: controller.signal, cache: "force-cache" });
        if (!response.ok) throw new Error(`Data request returned HTTP ${response.status}.`);
        const payload: unknown = await response.json();
        if (!isDashboardData(payload)) throw new Error("The dashboard data contract is invalid or unsupported.");
        setData(payload);
      } catch (caught) {
        if (controller.signal.aborted) return;
        setError(caught instanceof Error ? caught.message : "An unexpected data-loading error occurred.");
      }
    }

    void loadData();
    return () => controller.abort();
  }, [requestKey]);

  function retry() {
    setData(null);
    setError(null);
    setRequestKey((value) => value + 1);
  }

  if (error) return <ErrorDashboard message={error} retry={retry} />;
  if (!data) return <LoadingDashboard />;
  return <DashboardContent data={data} />;
}
