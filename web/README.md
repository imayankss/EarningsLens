# EarningsLens Web Dashboard

Static-first Next.js dashboard for the Earnings Call Sentiment Analyzer project.

## Local Run

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000.

## Verification

```bash
cd web
npm run lint
npm run build
```

The dashboard uses local TypeScript data from `web/data/` and does not call a backend or modify the Python NLP pipeline.
