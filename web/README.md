# EarningsLens Web Dashboard

Static-first Next.js presentation layer for checked EarningsLens pipeline artifacts.

## Architecture

The browser never loads FinBERT, downloads transcripts or market data, or trains a model. The offline Python pipeline produces research artifacts; `scripts/export_web_data.py` converts those artifacts into `public/data/dashboard.json`; Next.js serves the JSON and static UI on Vercel.

The checked repository snapshot contains one AAPL transcript/event. The application labels it as a demonstration dataset and does not fabricate missing company, speaker, section, or model metrics.

## Local development

From the repository root, regenerate the deployment data:

```bash
python3 scripts/export_web_data.py
```

Then run the frontend:

```bash
cd web
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Verification

```bash
npm run validate:data
npm run lint
npm run typecheck
npm test
npm run build
```

No frontend environment variables are required for the current static deployment.

## Vercel

Use `web/` as the Vercel Root Directory. Framework, install command, and build command should be detected as Next.js, `npm install`/`npm ci`, and `npm run build`. Large-scale pipeline jobs remain outside Vercel builds and requests.
