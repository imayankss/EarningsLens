"use client";

import { motion } from "framer-motion";
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  Blocks,
  BrainCircuit,
  CandlestickChart,
  Code2,
  Cpu,
  DatabaseZap,
  ExternalLink,
  FileText,
  GitBranch,
  LibraryBig,
  LineChart as LineChartIcon,
  Presentation,
  Radar,
  ShieldCheck,
  Sparkles,
  UsersRound,
  Workflow,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { FinanceTicker } from "@/components/dashboard/finance-ticker";
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
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  ctaLinks,
  demoDrivers,
  eventStudyData,
  heroMetrics,
  insightCards,
  lmCategories,
  modelComparison,
  pipelineStages,
  predictiveCards,
  sentimentCards,
  sentimentDistribution,
  speakerInsights,
  tickerMovements,
  transcriptMoments,
} from "@/data/dashboard-data";
import { cn } from "@/lib/utils";
import type { SentimentTone } from "@/types/dashboard";

const toneStyles: Record<
  SentimentTone,
  { bg: string; border: string; text: string; soft: string; line: string }
> = {
  positive: {
    bg: "bg-emerald-400/10",
    border: "border-emerald-400/25",
    text: "text-emerald-300",
    soft: "from-emerald-400/20 to-cyan-400/5",
    line: "#22c55e",
  },
  negative: {
    bg: "bg-rose-400/10",
    border: "border-rose-400/25",
    text: "text-rose-300",
    soft: "from-rose-400/20 to-slate-400/5",
    line: "#f43f5e",
  },
  neutral: {
    bg: "bg-slate-400/10",
    border: "border-slate-400/20",
    text: "text-slate-300",
    soft: "from-slate-400/20 to-cyan-400/5",
    line: "#94a3b8",
  },
  warning: {
    bg: "bg-amber-400/10",
    border: "border-amber-400/25",
    text: "text-amber-300",
    soft: "from-amber-400/20 to-cyan-400/5",
    line: "#f59e0b",
  },
};

type TooltipPayload = {
  color?: string;
  name?: string;
  value?: number | string;
};

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string | number;
}) {
  if (!active || !payload?.length) {
    return null;
  }

  return (
    <div className="rounded-md border border-white/10 bg-slate-950/95 px-3 py-2 font-mono text-xs shadow-glow backdrop-blur-xl">
      {label ? <div className="mb-1 text-slate-300">{label}</div> : null}
      <div className="space-y-1">
        {payload.map((item) => (
          <div key={`${item.name}-${item.value}`} className="flex items-center gap-2">
            <span
              className="size-2 rounded-full"
              style={{ backgroundColor: item.color ?? "#38bdf8" }}
            />
            <span className="text-slate-400">{item.name}</span>
            <span className="text-white">{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ChartPanel({
  title,
  description,
  children,
  className,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 22 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.18 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className={className}
    >
      <Card className="h-full overflow-hidden">
        <div className="h-1 bg-gradient-to-r from-cyan-400 via-violet-400 to-emerald-400" />
        <CardHeader>
          <CardTitle className="text-white">{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent>{children}</CardContent>
      </Card>
    </motion.div>
  );
}

function DashboardIcon({
  name,
  className,
}: {
  name: string;
  className?: string;
}) {
  switch (name) {
    case "BarChart3":
      return <BarChart3 className={className} />;
    case "Blocks":
      return <Blocks className={className} />;
    case "BrainCircuit":
      return <BrainCircuit className={className} />;
    case "CandlestickChart":
      return <CandlestickChart className={className} />;
    case "Code2":
      return <Code2 className={className} />;
    case "Cpu":
      return <Cpu className={className} />;
    case "DatabaseZap":
      return <DatabaseZap className={className} />;
    case "FileText":
      return <FileText className={className} />;
    case "Github":
      return <GitBranch className={className} />;
    case "LibraryBig":
      return <LibraryBig className={className} />;
    case "LineChart":
      return <LineChartIcon className={className} />;
    case "Presentation":
      return <Presentation className={className} />;
    case "Radar":
      return <Radar className={className} />;
    case "ShieldCheck":
      return <ShieldCheck className={className} />;
    case "Sparkles":
      return <Sparkles className={className} />;
    case "UsersRound":
      return <UsersRound className={className} />;
    case "Workflow":
      return <Workflow className={className} />;
    default:
      return <Activity className={className} />;
  }
}

function HeroMetricCard({
  metric,
  index,
}: {
  metric: (typeof heroMetrics)[number];
  index: number;
}) {
  const tone = toneStyles[metric.tone];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.48 + index * 0.08 }}
    >
      <Card className={cn("h-full overflow-hidden", tone.border)}>
        <CardContent className="p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="font-mono text-[11px] uppercase text-slate-400">
                {metric.label}
              </p>
              <p className={cn("mt-2 text-2xl font-semibold", tone.text)}>
                {metric.value}
              </p>
            </div>
            <div className={cn("rounded-md border p-2", tone.bg, tone.border)}>
              <DashboardIcon name={metric.icon} className={cn("size-5", tone.text)} />
            </div>
          </div>
          <p className="mt-3 text-sm text-slate-400">{metric.detail}</p>
        </CardContent>
      </Card>
    </motion.div>
  );
}

function SignalCard({
  label,
  value,
  description,
  tone,
}: {
  label: string;
  value: string;
  description: string;
  tone: SentimentTone;
}) {
  const style = toneStyles[tone];

  return (
    <Card className={cn("overflow-hidden", style.border)}>
      <CardContent className="p-5">
        <div className={cn("mb-4 h-1 w-14 rounded-full", style.bg)} />
        <div className={cn("font-mono text-2xl font-semibold", style.text)}>
          {value}
        </div>
        <div className="mt-2 text-sm font-medium text-white">{label}</div>
        <p className="mt-2 text-sm leading-6 text-slate-400">{description}</p>
      </CardContent>
    </Card>
  );
}

function HeroSection() {
  return (
    <section className="relative overflow-hidden px-4 pb-12 pt-8 sm:px-6 lg:px-8">
      <div aria-hidden className="absolute inset-0 market-grid opacity-70" />
      <div
        aria-hidden
        className="absolute inset-x-0 top-0 h-36 bg-gradient-to-b from-cyan-400/10 to-transparent"
      />
      <div
        aria-hidden
        className="scanline absolute left-0 right-0 top-0 h-24 bg-gradient-to-b from-transparent via-cyan-300/10 to-transparent"
      />

      <div className="relative mx-auto grid min-h-[760px] max-w-7xl items-center gap-10 py-10 lg:grid-cols-[1.08fr_0.92fr] lg:py-16">
        <div>
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55 }}
            className="mb-5 flex flex-wrap gap-3"
          >
            <Badge
              variant="outline"
              className="border-cyan-400/30 bg-cyan-400/10 font-mono text-cyan-200"
            >
              Static-first Next.js dashboard
            </Badge>
            <Badge
              variant="outline"
              className="border-emerald-400/30 bg-emerald-400/10 font-mono text-emerald-200"
            >
              Python NLP pipeline unchanged
            </Badge>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.65, delay: 0.08 }}
            className="max-w-5xl text-6xl font-semibold leading-none text-white sm:text-7xl lg:text-8xl"
          >
            EarningsLens
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.16 }}
            className="mt-5 text-2xl font-medium text-cyan-100 sm:text-3xl"
          >
            Earnings Call Sentiment Intelligence Platform
          </motion.p>
          <motion.p
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.24 }}
            className="mt-6 max-w-3xl text-base leading-8 text-slate-300 sm:text-lg"
          >
            AI-powered financial NLP platform analyzing earnings call transcripts
            using FinBERT, Loughran-McDonald sentiment analysis,
            speaker-aware transcript processing, and event-study pipelines.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.32 }}
            className="mt-8 flex flex-wrap gap-3"
          >
            <Button asChild size="lg" className="bg-cyan-300 text-slate-950 hover:bg-cyan-200">
              <a
                href="https://github.com/imayankss/earnings-call-sentiment-analyzer"
                target="_blank"
                rel="noreferrer"
              >
                View GitHub Repository
                <ArrowUpRight className="size-4" />
              </a>
            </Button>
            <Button asChild size="lg" variant="outline" className="border-white/15 bg-white/5">
              <a href="#interactive-demo">Open Demo Signal</a>
            </Button>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.22 }}
          className="relative"
        >
          <Card className="relative overflow-hidden border-cyan-400/20 bg-slate-950/70">
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-300 to-transparent" />
            <CardHeader>
              <div className="flex items-center justify-between gap-3">
                <div>
                  <CardTitle className="text-white">AAPL Q4 2020 Signal</CardTitle>
                  <CardDescription>
                    Demo-ready financial NLP cockpit
                  </CardDescription>
                </div>
                <div className="sentiment-pulse rounded-full border border-emerald-400/30 bg-emerald-400/10 p-4">
                  <Radar className="size-7 text-emerald-300" />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-3">
                <div className="rounded-md border border-white/10 bg-white/[0.04] p-4">
                  <p className="font-mono text-xs text-slate-400">Overall sentiment</p>
                  <p className="mt-2 text-2xl font-semibold text-emerald-300">Positive</p>
                </div>
                <div className="rounded-md border border-white/10 bg-white/[0.04] p-4">
                  <p className="font-mono text-xs text-slate-400">Confidence</p>
                  <p className="mt-2 text-2xl font-semibold text-white">87%</p>
                </div>
                <div className="rounded-md border border-white/10 bg-white/[0.04] p-4">
                  <p className="font-mono text-xs text-slate-400">Market signal</p>
                  <p className="mt-2 text-2xl font-semibold text-cyan-200">Bullish</p>
                </div>
              </div>

              <div className="mt-6 h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={transcriptMoments}>
                    <defs>
                      <linearGradient id="heroSentiment" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.48} />
                        <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" vertical={false} />
                    <XAxis dataKey="chunk" stroke="#64748b" tickLine={false} axisLine={false} />
                    <YAxis hide domain={[0, 0.7]} />
                    <Tooltip content={<ChartTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="sentiment"
                      stroke="#38bdf8"
                      strokeWidth={3}
                      fill="url(#heroSentiment)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      <div className="relative mx-auto grid max-w-7xl gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {heroMetrics.map((metric, index) => (
          <HeroMetricCard key={metric.label} metric={metric} index={index} />
        ))}
      </div>
    </section>
  );
}

function ProjectOverview() {
  return (
    <SectionShell
      eyebrow="Project Overview"
      title="A pipeline that turns raw earnings calls into market-aware sentiment features."
      description="The dashboard mirrors the existing architecture: transcript ingestion, cleaning, chunking, FinBERT scoring, Loughran-McDonald dictionary signals, speaker analysis, event-study alignment, and predictive modeling hooks."
    >
      <div className="relative overflow-hidden rounded-lg border border-white/10 bg-slate-950/55 p-4 backdrop-blur-xl">
        <div className="absolute left-8 right-8 top-1/2 hidden h-px bg-gradient-to-r from-cyan-400/10 via-cyan-300/70 to-emerald-400/10 lg:block" />
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {pipelineStages.map((stage, index) => (
            <motion.div
              key={stage.stage}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.2 }}
              transition={{ duration: 0.45, delay: index * 0.04 }}
              className="relative rounded-md border border-white/10 bg-slate-950/85 p-4"
            >
              <div className="mb-4 flex items-center justify-between">
                <span className="font-mono text-xs text-cyan-300">{stage.stage}</span>
                <span className="size-2 rounded-full bg-cyan-300 shadow-glow" />
              </div>
              <h3 className="text-base font-semibold text-white">{stage.title}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-400">{stage.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </SectionShell>
  );
}

function SentimentEngine() {
  return (
    <SectionShell
      eyebrow="Sentiment Engine"
      title="Contextual model scores and dictionary tone in one analyst view."
      description="FinBERT captures earnings-call language in context, while Loughran-McDonald surfaces auditable financial tone categories."
    >
      <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <ChartPanel
          title="FinBERT Sentiment Distribution"
          description="Positive, neutral, and negative probability mix from the verified baseline call."
        >
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={sentimentDistribution}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={70}
                  outerRadius={104}
                  paddingAngle={4}
                >
                  {sentimentDistribution.map((slice) => (
                    <Cell key={slice.name} fill={slice.color} />
                  ))}
                </Pie>
                <Tooltip content={<ChartTooltip />} />
                <Legend iconType="circle" />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </ChartPanel>

        <ChartPanel
          title="Loughran-McDonald Signal Categories"
          description="Dictionary-based financial tone counts, mixing verified and clearly marked demo values."
        >
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={lmCategories}>
                <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" vertical={false} />
                <XAxis dataKey="name" stroke="#94a3b8" tickLine={false} axisLine={false} />
                <YAxis stroke="#64748b" tickLine={false} axisLine={false} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {lmCategories.map((slice) => (
                    <Cell key={slice.name} fill={slice.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartPanel>
      </div>

      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {sentimentCards.map((card) => (
          <SignalCard key={card.label} {...card} />
        ))}
      </div>
    </SectionShell>
  );
}

function ModelComparison() {
  return (
    <SectionShell
      eyebrow="FinBERT vs Loughran-McDonald"
      title="Two sentiment lenses: contextual intelligence and auditable financial dictionaries."
      description="The dashboard keeps both views visible because institutional analysis benefits from model nuance and transparent category counts."
    >
      <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <ChartPanel
          title="Signal Capability Comparison"
          description="Static scoring rubric for how each method contributes to a financial NLP product."
        >
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelComparison}>
                <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" vertical={false} />
                <XAxis dataKey="category" stroke="#94a3b8" tickLine={false} axisLine={false} />
                <YAxis stroke="#64748b" tickLine={false} axisLine={false} domain={[0, 100]} />
                <Tooltip content={<ChartTooltip />} />
                <Legend />
                <Bar dataKey="finbert" name="FinBERT" fill="#38bdf8" radius={[6, 6, 0, 0]} />
                <Bar dataKey="lm" name="LM Dictionary" fill="#f59e0b" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartPanel>

        <Card className="h-full">
          <CardHeader>
            <CardTitle className="text-white">Comparison Notes</CardTitle>
            <CardDescription>
              What each engine contributes to the research workflow.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="finbert">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="finbert">FinBERT</TabsTrigger>
                <TabsTrigger value="lm">LM</TabsTrigger>
                <TabsTrigger value="bridge">Bridge</TabsTrigger>
              </TabsList>
              <TabsContent value="finbert" className="rounded-md border border-cyan-400/20 bg-cyan-400/10 p-4">
                <BrainCircuit className="mb-3 size-6 text-cyan-200" />
                <p className="text-sm leading-6 text-slate-200">
                  FinBERT captures contextual financial language, including tone
                  that depends on sentence structure, forward-looking statements,
                  and earnings-call phrasing.
                </p>
              </TabsContent>
              <TabsContent value="lm" className="rounded-md border border-amber-400/20 bg-amber-400/10 p-4">
                <LibraryBig className="mb-3 size-6 text-amber-200" />
                <p className="text-sm leading-6 text-slate-200">
                  Loughran-McDonald captures dictionary-based financial tone,
                  making uncertainty, litigation, constraint, positive, and
                  negative words easy to inspect.
                </p>
              </TabsContent>
              <TabsContent value="bridge" className="rounded-md border border-emerald-400/20 bg-emerald-400/10 p-4">
                <Workflow className="mb-3 size-6 text-emerald-200" />
                <p className="text-sm leading-6 text-slate-200">
                  Together they create richer features for event studies and
                  downstream market reaction modeling.
                </p>
              </TabsContent>
            </Tabs>
            <Separator className="my-6 bg-white/10" />
            <div className="grid gap-3 font-mono text-sm">
              <div className="flex items-center justify-between rounded-md bg-white/[0.04] p-3">
                <span className="text-slate-400">FinBERT score</span>
                <span className="text-cyan-200">0.432</span>
              </div>
              <div className="flex items-center justify-between rounded-md bg-white/[0.04] p-3">
                <span className="text-slate-400">LM tone score</span>
                <span className="text-amber-200">0.492</span>
              </div>
              <div className="flex items-center justify-between rounded-md bg-white/[0.04] p-3">
                <span className="text-slate-400">Label agreement</span>
                <span className="text-emerald-200">Positive</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </SectionShell>
  );
}

function TranscriptIntelligence() {
  const sectionLabels = ["Prepared Remarks", "Financial Results", "Guidance", "Q&A"];

  return (
    <SectionShell
      eyebrow="Transcript Intelligence"
      title="Chunk-level sentiment reveals how tone changes across the call."
      description="The timeline separates prepared remarks, financial results, guidance, and Q&A so the dashboard can show where optimism or caution enters the call."
    >
      <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <ChartPanel
          title="Sentiment by Transcript Chunk"
          description="Static demo timeline built for future transcript-level interactivity."
        >
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={transcriptMoments}>
                <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" vertical={false} />
                <XAxis dataKey="chunk" stroke="#94a3b8" tickLine={false} axisLine={false} />
                <YAxis stroke="#64748b" tickLine={false} axisLine={false} domain={[0, 0.7]} />
                <Tooltip content={<ChartTooltip />} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="sentiment"
                  name="Sentiment"
                  stroke="#22c55e"
                  strokeWidth={3}
                  dot={{ r: 4, fill: "#22c55e" }}
                />
                <Line
                  type="monotone"
                  dataKey="confidence"
                  name="Confidence"
                  stroke="#38bdf8"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </ChartPanel>

        <Card>
          <CardHeader>
            <CardTitle className="text-white">Call Sections</CardTitle>
            <CardDescription>
              Storytelling labels for transcript progression.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {sectionLabels.map((section, index) => (
              <motion.div
                key={section}
                initial={{ opacity: 0, x: 18 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, amount: 0.3 }}
                transition={{ duration: 0.45, delay: index * 0.08 }}
                className="rounded-md border border-white/10 bg-white/[0.04] p-4"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-white">{section}</span>
                  <Badge variant="outline" className="border-cyan-400/25 bg-cyan-400/10 font-mono text-cyan-200">
                    phase {index + 1}
                  </Badge>
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-400">
                  {index === 0
                    ? "Management frames the quarter and sets the strategic tone."
                    : index === 1
                      ? "Revenue, margin, and product performance language enters."
                      : index === 2
                        ? "Forward-looking language drives uncertainty and signal strength."
                        : "Analyst scrutiny tests whether positive tone survives questions."}
                </p>
              </motion.div>
            ))}
          </CardContent>
        </Card>
      </div>
    </SectionShell>
  );
}

function SpeakerAwareInsights() {
  return (
    <SectionShell
      eyebrow="Speaker-Aware Insights"
      title="Management tone and analyst pressure are separated into research-ready signals."
      description="Speaker-level cards compare CEO sentiment, CFO sentiment, analyst Q&A sentiment, and aggregate management tone."
    >
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {speakerInsights.map((speaker) => {
          const style = toneStyles[speaker.tone];

          return (
            <Card key={speaker.speaker} className={cn("overflow-hidden", style.border)}>
              <CardContent className="p-5">
                <div className="mb-4 flex items-center justify-between">
                  <Badge variant="outline" className={cn(style.border, style.bg, style.text)}>
                    {speaker.role}
                  </Badge>
                  <UsersRound className={cn("size-5", style.text)} />
                </div>
                <h3 className="text-xl font-semibold text-white">{speaker.speaker}</h3>
                <div className="mt-5 space-y-4">
                  <div>
                    <div className="mb-2 flex justify-between font-mono text-xs text-slate-400">
                      <span>sentiment</span>
                      <span className={style.text}>{speaker.sentiment}%</span>
                    </div>
                    <Progress value={speaker.sentiment} />
                  </div>
                  <div>
                    <div className="mb-2 flex justify-between font-mono text-xs text-slate-400">
                      <span>confidence</span>
                      <span className="text-cyan-200">{speaker.confidence}%</span>
                    </div>
                    <Progress value={speaker.confidence} />
                  </div>
                </div>
                <p className="mt-5 text-sm leading-6 text-slate-400">{speaker.note}</p>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </SectionShell>
  );
}

function EventStudy() {
  return (
    <SectionShell
      eyebrow="Event Study / Market Reaction"
      title="Sentiment becomes more useful when it is compared with post-earnings movement."
      description="The event-study view aligns abnormal return and cumulative abnormal return windows around the earnings event."
    >
      <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <ChartPanel
          title="Abnormal Return and CAR Window"
          description="Sample static data showing how a bullish call could be evaluated against market reaction."
        >
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={eventStudyData}>
                <defs>
                  <linearGradient id="carFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.38} />
                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" vertical={false} />
                <XAxis dataKey="day" stroke="#94a3b8" tickLine={false} axisLine={false} />
                <YAxis stroke="#64748b" tickLine={false} axisLine={false} />
                <Tooltip content={<ChartTooltip />} />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="car"
                  name="CAR %"
                  stroke="#22c55e"
                  strokeWidth={3}
                  fill="url(#carFill)"
                />
                <Line
                  type="monotone"
                  dataKey="abnormalReturn"
                  name="Abnormal return %"
                  stroke="#38bdf8"
                  strokeWidth={2}
                  dot={{ r: 3, fill: "#38bdf8" }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </ChartPanel>

        <Card>
          <CardHeader>
            <CardTitle className="text-white">Market Interpretation</CardTitle>
            <CardDescription>
              How a sentiment dashboard supports financial research.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-sm leading-6 text-slate-300">
            <div className="rounded-md border border-emerald-400/20 bg-emerald-400/10 p-4">
              <p className="font-medium text-emerald-200">Positive language plus positive CAR</p>
              <p className="mt-2 text-slate-300">
                A constructive management tone can be compared with abnormal
                returns to evaluate whether the market rewarded the narrative.
              </p>
            </div>
            <div className="rounded-md border border-cyan-400/20 bg-cyan-400/10 p-4">
              <p className="font-medium text-cyan-200">Feature bridge</p>
              <p className="mt-2 text-slate-300">
                FinBERT scores, LM tone, uncertainty ratios, and speaker
                signals can become inputs for post-earnings direction models.
              </p>
            </div>
            <div className="rounded-md border border-amber-400/20 bg-amber-400/10 p-4">
              <p className="font-medium text-amber-200">Current caveat</p>
              <p className="mt-2 text-slate-300">
                Verified artifacts currently show n=1, so the dashboard uses
                static demo visuals for recruiter-friendly product storytelling.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </SectionShell>
  );
}

function PredictiveModeling() {
  return (
    <SectionShell
      eyebrow="Predictive Modeling"
      title="Modeling cards show the intended ML contract without overstating current sample size."
      description="The project includes feature engineering and model metric surfaces, with evaluation guarded until more observations are available."
    >
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {predictiveCards.map((card, index) => (
          <motion.div
            key={card.label}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.25 }}
            transition={{ duration: 0.45, delay: index * 0.06 }}
          >
            <Card className="h-full overflow-hidden">
              <CardContent className="p-5">
                <div className="mb-5 flex items-center justify-between">
                  <Cpu className="size-5 text-cyan-200" />
                  <Badge variant="outline" className="border-slate-400/20 bg-slate-400/10 font-mono text-slate-300">
                    {card.status}
                  </Badge>
                </div>
                <p className="text-sm text-slate-400">{card.label}</p>
                <p className="mt-2 text-2xl font-semibold text-white">{card.value}</p>
                <p className="mt-4 text-sm leading-6 text-slate-400">{card.detail}</p>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </SectionShell>
  );
}

function InteractiveDemo() {
  return (
    <SectionShell
      eyebrow="Interactive Earnings Call Demo"
      title="A polished Apple Q4 2020 signal card for interview walkthroughs."
      description="This static card demonstrates what a future interactive call analysis experience could feel like once more transcripts are loaded."
      className="scroll-mt-8"
    >
      <Card id="interactive-demo" className="overflow-hidden border-cyan-400/20">
        <div className="h-1 bg-gradient-to-r from-cyan-300 via-emerald-300 to-amber-300" />
        <CardContent className="grid gap-8 p-6 lg:grid-cols-[0.82fr_1.18fr] lg:p-8">
          <div>
            <Badge variant="outline" className="border-cyan-400/30 bg-cyan-400/10 font-mono text-cyan-200">
              Apple Inc. · Q4 2020
            </Badge>
            <h3 className="mt-5 text-3xl font-semibold text-white">Overall Sentiment: Positive</h3>
            <p className="mt-4 text-sm leading-7 text-slate-300">
              Confidence is high, uncertainty language is low, and the market
              reaction signal is presented as bullish for demo purposes.
            </p>

            <div className="mt-8 grid gap-4 sm:grid-cols-2">
              <div className="rounded-md border border-emerald-400/20 bg-emerald-400/10 p-4">
                <p className="font-mono text-xs text-emerald-200">Confidence</p>
                <p className="mt-2 text-3xl font-semibold text-white">87%</p>
              </div>
              <div className="rounded-md border border-cyan-400/20 bg-cyan-400/10 p-4">
                <p className="font-mono text-xs text-cyan-200">Market Signal</p>
                <p className="mt-2 text-3xl font-semibold text-white">Bullish</p>
              </div>
            </div>
          </div>

          <div className="rounded-md border border-white/10 bg-white/[0.04] p-4">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-white">Key Drivers</p>
                <p className="text-xs text-slate-400">Static driver table for the demo experience</p>
              </div>
              <ShieldCheck className="size-5 text-emerald-300" />
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Driver</TableHead>
                  <TableHead>Impact</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {demoDrivers.map((driver) => (
                  <TableRow key={driver.driver}>
                    <TableCell className="font-medium text-white">{driver.driver}</TableCell>
                    <TableCell className="font-mono text-emerald-200">{driver.impact}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </SectionShell>
  );
}

function InsightsAndCta() {
  return (
    <SectionShell
      eyebrow="Insights Summary"
      title="What EarningsLens proves as a portfolio project."
      description="The product story highlights financial NLP architecture, event-study framing, speaker-aware analytics, and investment research UX."
    >
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        {insightCards.map((insight) => {
          return (
            <Card key={insight.title} className="h-full">
              <CardContent className="p-5">
                <div className="mb-4 flex size-10 items-center justify-center rounded-md border border-cyan-400/25 bg-cyan-400/10">
                  <DashboardIcon name={insight.icon} className="size-5 text-cyan-200" />
                </div>
                <h3 className="text-base font-semibold text-white">{insight.title}</h3>
                <p className="mt-3 text-sm leading-6 text-slate-400">{insight.description}</p>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <Card className="mt-8 overflow-hidden border-emerald-400/20">
        <CardContent className="grid gap-6 p-6 lg:grid-cols-[1fr_auto] lg:items-center">
          <div>
            <Badge variant="outline" className="border-emerald-400/30 bg-emerald-400/10 font-mono text-emerald-200">
              GitHub CTA
            </Badge>
            <h3 className="mt-4 text-2xl font-semibold text-white">Explore the project source and methodology.</h3>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
              The web dashboard is a static-first presentation layer over the
              existing Python NLP and event-study workflow.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            {ctaLinks.map((link) => {
              return (
                <Button key={link.href} asChild variant="outline" className="border-white/15 bg-white/5">
                  <a href={link.href} target="_blank" rel="noreferrer">
                    <DashboardIcon name={link.icon} className="size-4" />
                    {link.label}
                    <ExternalLink className="size-3.5" />
                  </a>
                </Button>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </SectionShell>
  );
}

export function EarningsLensDashboard() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-[#050816]">
      <HeroSection />
      <FinanceTicker items={tickerMovements} />
      <ProjectOverview />
      <SentimentEngine />
      <ModelComparison />
      <TranscriptIntelligence />
      <SpeakerAwareInsights />
      <EventStudy />
      <PredictiveModeling />
      <InteractiveDemo />
      <InsightsAndCta />
    </main>
  );
}
