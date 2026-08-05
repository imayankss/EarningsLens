import type { PipelineStage, RepositoryLink } from "@/types/dashboard";

export const pipelineStages: PipelineStage[] = [
  {
    stage: "01",
    title: "Transcript ingestion",
    description: "Load earnings-call text and preserve company, quarter, date, and transcript identifiers.",
    runtime: "Offline Python",
  },
  {
    stage: "02",
    title: "Preprocessing",
    description: "Clean transcript text and split long calls into model-safe, metadata-aware chunks.",
    runtime: "Offline Python",
  },
  {
    stage: "03",
    title: "FinBERT + LM",
    description: "Generate contextual probabilities and explainable finance-dictionary tone features.",
    runtime: "Offline ML",
  },
  {
    stage: "04",
    title: "Market alignment",
    description: "Join earnings events with return, abnormal-return, and CAR research windows.",
    runtime: "Offline batch",
  },
  {
    stage: "05",
    title: "Web export",
    description: "Serialize checked, non-sensitive artifacts into a small versioned JSON contract.",
    runtime: "Build input",
  },
  {
    stage: "06",
    title: "Static dashboard",
    description: "Render the exported snapshot in Next.js without loading models or calling data vendors.",
    runtime: "Vercel",
  },
];

export const repositoryLinks: RepositoryLink[] = [
  {
    label: "GitHub repository",
    href: "https://github.com/imayankss/EarningsLens",
  },
  {
    label: "Python analytics layer",
    href: "https://github.com/imayankss/EarningsLens/tree/main/src",
  },
  {
    label: "Research methodology",
    href: "https://github.com/imayankss/EarningsLens/blob/main/docs/METHODOLOGY.md",
  },
];
