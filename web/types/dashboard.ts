import type { LucideIcon } from "lucide-react";

export type SentimentTone = "positive" | "negative" | "neutral" | "warning";

export type HeroMetric = {
  label: string;
  value: string;
  detail: string;
  tone: SentimentTone;
  icon: string;
};

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
};

export type SentimentSlice = {
  name: string;
  value: number;
  color: string;
};

export type SentimentCard = {
  label: string;
  value: string;
  description: string;
  tone: SentimentTone;
};

export type ModelComparisonPoint = {
  category: string;
  finbert: number;
  lm: number;
};

export type TranscriptMoment = {
  chunk: number;
  section: string;
  sentiment: number;
  confidence: number;
};

export type SpeakerInsight = {
  speaker: string;
  role: string;
  sentiment: number;
  confidence: number;
  tone: SentimentTone;
  note: string;
};

export type EventStudyPoint = {
  day: string;
  abnormalReturn: number;
  car: number;
};

export type PredictiveCard = {
  label: string;
  value: string;
  status: string;
  detail: string;
};

export type DemoDriver = {
  driver: string;
  impact: string;
};

export type InsightCard = {
  title: string;
  description: string;
  icon: string;
};

export type CtaLink = {
  label: string;
  href: string;
  icon: string;
};

export type IconMap = Record<string, LucideIcon>;
