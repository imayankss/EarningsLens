import type {
  CtaLink,
  DemoDriver,
  EventStudyPoint,
  HeroMetric,
  InsightCard,
  ModelComparisonPoint,
  PipelineStage,
  PredictiveCard,
  SentimentCard,
  SentimentSlice,
  SpeakerInsight,
  TickerMovement,
  TranscriptMoment,
} from "@/types/dashboard";

export const heroMetrics: HeroMetric[] = [
  {
    label: "Transcripts Processed",
    value: "1 verified",
    detail: "AAPL Q4 2020 baseline run",
    tone: "positive",
    icon: "FileText",
  },
  {
    label: "Transcript Chunks",
    value: "31",
    detail: "speaker-aware analysis units",
    tone: "neutral",
    icon: "Blocks",
  },
  {
    label: "FinBERT Sentiment",
    value: "46.1%",
    detail: "positive probability snapshot",
    tone: "positive",
    icon: "BrainCircuit",
  },
  {
    label: "LM Dictionary Signals",
    value: "232",
    detail: "positive and negative polarity hits",
    tone: "warning",
    icon: "LibraryBig",
  },
  {
    label: "Event Study Ready",
    value: "1D-5D",
    detail: "post-earnings return windows",
    tone: "positive",
    icon: "LineChart",
  },
];

export const tickerMovements: TickerMovement[] = [
  {
    symbol: "AAPL",
    company: "Apple Inc.",
    sentiment: "Positive",
    move: "+1.8%",
    tone: "positive",
  },
  {
    symbol: "MSFT",
    company: "Microsoft",
    sentiment: "Neutral",
    move: "+0.4%",
    tone: "neutral",
  },
  {
    symbol: "TSLA",
    company: "Tesla",
    sentiment: "Negative",
    move: "-2.1%",
    tone: "negative",
  },
  {
    symbol: "NVDA",
    company: "NVIDIA",
    sentiment: "Positive",
    move: "+3.2%",
    tone: "positive",
  },
];

export const pipelineStages: PipelineStage[] = [
  {
    stage: "01",
    title: "Transcript Ingestion",
    description: "Load earnings call text, metadata, fiscal quarter, ticker, and event date.",
  },
  {
    stage: "02",
    title: "Cleaning",
    description: "Normalize call text, remove artifacts, and preserve finance language.",
  },
  {
    stage: "03",
    title: "Chunking",
    description: "Split long calls into model-safe windows for robust sentiment scoring.",
  },
  {
    stage: "04",
    title: "FinBERT",
    description: "Score contextual positive, neutral, and negative financial sentiment.",
  },
  {
    stage: "05",
    title: "Loughran-McDonald",
    description: "Extract domain dictionary signals such as tone, uncertainty, and constraint.",
  },
  {
    stage: "06",
    title: "Speaker Analysis",
    description: "Compare management tone, prepared remarks, CFO updates, and analyst Q&A.",
  },
  {
    stage: "07",
    title: "Event Study",
    description: "Align sentiment with abnormal returns and cumulative market reaction.",
  },
  {
    stage: "08",
    title: "Predictive Modeling",
    description: "Package sentiment and market features for direction prediction experiments.",
  },
];

export const sentimentDistribution: SentimentSlice[] = [
  { name: "Positive", value: 46.1, color: "#22c55e" },
  { name: "Neutral", value: 51.1, color: "#94a3b8" },
  { name: "Negative", value: 2.8, color: "#f43f5e" },
];

export const lmCategories: SentimentSlice[] = [
  { name: "Positive", value: 172, color: "#22c55e" },
  { name: "Negative", value: 60, color: "#f43f5e" },
  { name: "Uncertainty", value: 46, color: "#f59e0b" },
  { name: "Litigious", value: 18, color: "#8b5cf6" },
  { name: "Constraining", value: 12, color: "#38bdf8" },
];

export const sentimentCards: SentimentCard[] = [
  {
    label: "Positive sentiment",
    value: "46.1%",
    description: "FinBERT positive probability on the verified call.",
    tone: "positive",
  },
  {
    label: "Negative sentiment",
    value: "2.8%",
    description: "Low negative probability across the baseline transcript.",
    tone: "negative",
  },
  {
    label: "Neutral sentiment",
    value: "51.1%",
    description: "Balanced corporate language and factual reporting.",
    tone: "neutral",
  },
  {
    label: "Uncertainty",
    value: "46 hits",
    description: "Dictionary-driven uncertainty mentions from LM scoring.",
    tone: "warning",
  },
  {
    label: "Litigious",
    value: "18 demo",
    description: "Static demo value for legal and risk language tracking.",
    tone: "neutral",
  },
  {
    label: "Constraining",
    value: "12 demo",
    description: "Static demo value for operational constraint language.",
    tone: "warning",
  },
];

export const modelComparison: ModelComparisonPoint[] = [
  { category: "Context", finbert: 92, lm: 44 },
  { category: "Finance Tone", finbert: 82, lm: 88 },
  { category: "Explainability", finbert: 68, lm: 94 },
  { category: "Chunk Fit", finbert: 90, lm: 78 },
  { category: "Market Feature", finbert: 84, lm: 81 },
];

export const transcriptMoments: TranscriptMoment[] = [
  { chunk: 1, section: "Prepared Remarks", sentiment: 0.31, confidence: 0.76 },
  { chunk: 4, section: "Prepared Remarks", sentiment: 0.43, confidence: 0.79 },
  { chunk: 7, section: "Financial Results", sentiment: 0.58, confidence: 0.84 },
  { chunk: 10, section: "Financial Results", sentiment: 0.49, confidence: 0.78 },
  { chunk: 13, section: "Guidance", sentiment: 0.37, confidence: 0.74 },
  { chunk: 16, section: "Guidance", sentiment: 0.52, confidence: 0.81 },
  { chunk: 19, section: "Q&A", sentiment: 0.29, confidence: 0.71 },
  { chunk: 22, section: "Q&A", sentiment: 0.41, confidence: 0.77 },
  { chunk: 25, section: "Q&A", sentiment: 0.48, confidence: 0.82 },
  { chunk: 28, section: "Q&A", sentiment: 0.44, confidence: 0.76 },
  { chunk: 31, section: "Q&A", sentiment: 0.55, confidence: 0.8 },
];

export const speakerInsights: SpeakerInsight[] = [
  {
    speaker: "CEO",
    role: "Management tone",
    sentiment: 84,
    confidence: 87,
    tone: "positive",
    note: "Optimistic demand language and resilient services commentary.",
  },
  {
    speaker: "CFO",
    role: "Financial framing",
    sentiment: 72,
    confidence: 82,
    tone: "positive",
    note: "Strong gross margin language with controlled uncertainty.",
  },
  {
    speaker: "Analyst Q&A",
    role: "Market scrutiny",
    sentiment: 58,
    confidence: 74,
    tone: "neutral",
    note: "Questions introduce caution around guidance and product cycles.",
  },
  {
    speaker: "Management",
    role: "Aggregate signal",
    sentiment: 79,
    confidence: 85,
    tone: "positive",
    note: "Prepared remarks carry a stronger bullish tone than Q&A.",
  },
];

export const eventStudyData: EventStudyPoint[] = [
  { day: "-5", abnormalReturn: -0.8, car: -0.8 },
  { day: "-4", abnormalReturn: 0.3, car: -0.5 },
  { day: "-3", abnormalReturn: 0.6, car: 0.1 },
  { day: "-2", abnormalReturn: -0.2, car: -0.1 },
  { day: "-1", abnormalReturn: 0.4, car: 0.3 },
  { day: "0", abnormalReturn: 0.0, car: 0.3 },
  { day: "+1", abnormalReturn: 1.13, car: 1.43 },
  { day: "+2", abnormalReturn: 3.86, car: 5.29 },
  { day: "+3", abnormalReturn: 9.19, car: 14.18 },
  { day: "+4", abnormalReturn: 2.6, car: 16.78 },
  { day: "+5", abnormalReturn: 15.14, car: 29.32 },
];

export const predictiveCards: PredictiveCard[] = [
  {
    label: "Direction Prediction",
    value: "Prototype",
    status: "Insufficient sample size",
    detail: "Model hooks exist; current verified dataset is n=1.",
  },
  {
    label: "Sentiment Features",
    value: "25",
    status: "Ready for expansion",
    detail: "FinBERT, LM tone, uncertainty, keywords, and return features.",
  },
  {
    label: "Market Reaction Signal",
    value: "Bullish demo",
    status: "Static signal",
    detail: "Demonstrates how sentiment can map to post-earnings movement.",
  },
  {
    label: "Evaluation Metrics",
    value: "Pending",
    status: "Guardrailed",
    detail: "Accuracy, precision, recall, F1, and ROC AUC need more observations.",
  },
];

export const demoDrivers: DemoDriver[] = [
  { driver: "strong revenue language", impact: "positive" },
  { driver: "positive guidance", impact: "positive" },
  { driver: "low uncertainty tone", impact: "risk reduced" },
  { driver: "optimistic management remarks", impact: "bullish" },
];

export const insightCards: InsightCard[] = [
  {
    title: "Financial NLP pipeline design",
    description:
      "The project separates ingestion, preprocessing, sentiment scoring, event-study analysis, and presentation into clear layers.",
    icon: "Workflow",
  },
  {
    title: "Domain-specific sentiment analysis",
    description:
      "FinBERT and Loughran-McDonald are presented together so contextual and dictionary signals can be compared.",
    icon: "BrainCircuit",
  },
  {
    title: "Event-study thinking",
    description:
      "The dashboard frames language signals beside abnormal returns and CAR windows instead of treating sentiment as an isolated score.",
    icon: "CandlestickChart",
  },
  {
    title: "Speaker-aware analytics",
    description:
      "Transcript sections and speaker roles make the analysis useful for management tone, CFO framing, and analyst pressure.",
    icon: "UsersRound",
  },
  {
    title: "Investment research dashboard",
    description:
      "The interface reads like an analyst cockpit: ticker motion, model comparison, signal cards, and clear caveats.",
    icon: "Presentation",
  },
];

export const ctaLinks: CtaLink[] = [
  {
    label: "View GitHub Repository",
    href: "https://github.com/imayankss/earnings-call-sentiment-analyzer",
    icon: "Github",
  },
  {
    label: "View NLP Pipeline",
    href: "https://github.com/imayankss/earnings-call-sentiment-analyzer/tree/main/src",
    icon: "Code2",
  },
  {
    label: "View Project Report",
    href: "https://github.com/imayankss/earnings-call-sentiment-analyzer/blob/main/docs/PROJECT_REPORT.md",
    icon: "FileText",
  },
];
