export type SentimentTone = "positive" | "negative" | "neutral" | "warning";

export type TickerMovement = {
  symbol: string;
  company: string;
  sentiment: "Positive" | "Neutral" | "Negative";
  move: string;
  tone: SentimentTone;
};

export type PipelineStage = {
  stage: string;
  title: string;
  description: string;
  runtime: string;
};

export type RepositoryLink = {
  label: string;
  href: string;
};

export type TranscriptSummary = {
  id: string;
  ticker: string | null;
  companyName: string | null;
  earningsDate: string | null;
  fiscalQuarter: number | null;
  fiscalYear: number | null;
  cleanWordCount: number | null;
  chunkCount: number | null;
};

export type FinbertResult = {
  score: number | null;
  positiveProbability: number | null;
  negativeProbability: number | null;
  neutralProbability: number | null;
  meanConfidence: number | null;
  direction: string | null;
};

export type LoughranMcDonaldResult = {
  toneScore: number | null;
  positiveCount: number | null;
  negativeCount: number | null;
  scoredWordCount: number | null;
  label: string | null;
};

export type ModelComparison = {
  directionalAgreement: boolean | null;
  scoreDifference: number | null;
  absoluteDifference: number | null;
};

export type MarketWindow = {
  label: string;
  horizonDays: number;
  rawReturn: number | null;
  abnormalReturn: number | null;
};

export type CumulativeAbnormalReturn = {
  label: string;
  horizonDays: number;
  value: number;
};

export type TopicResult = {
  name: string;
  count: number | null;
  ratio: number | null;
};

export type SpeakerGroup = {
  label: string | null;
  chunkCount: number | null;
  averageSentimentScore: number | null;
  sentimentStdDev: number | null;
  positiveChunks: number | null;
  negativeChunks: number | null;
  neutralChunks: number | null;
};

export type AnalysisResult = {
  transcript: TranscriptSummary;
  finbert: FinbertResult;
  loughranMcDonald: LoughranMcDonaldResult;
  comparison: ModelComparison;
  marketReaction: {
    eventDate: string | null;
    windows: MarketWindow[];
    cumulativeAbnormalReturns: CumulativeAbnormalReturn[];
    marketWindowStart: string | null;
    marketWindowEnd: string | null;
  };
  nlp: {
    totalTokens: number | null;
    uncertaintyCount: number | null;
    uncertaintyRatio: number | null;
    topKeywords: string[];
    topics: TopicResult[];
  };
  speakerAnalysis: {
    available: boolean;
    groups: SpeakerGroup[];
    aggregate: SpeakerGroup | null;
    message: string;
  };
};

export type DashboardData = {
  schemaVersion: number;
  dataset: {
    kind: "demonstration" | "research";
    label: string;
    observationCount: number;
    companyCount: number;
    isLimited: boolean;
    notice: string;
  };
  analyses: AnalysisResult[];
  modeling: {
    observationCount: number | null;
    featureCount: number | null;
    modelsAttempted: number;
    modelsTrained: number;
    status: string;
    message: string;
  };
  provenance: Array<{
    label: string;
    path: string;
  }>;
};
