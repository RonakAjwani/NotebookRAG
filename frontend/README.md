# Frontend - Hybrid RAG Dashboard

Single-page dashboard for the Hybrid RAG API: ask a question, get a grounded
answer with clickable verified citations, a confidence breakdown, and the
ranked retrieved chunks behind it. Includes a hybrid-vs-dense side-by-side
compare mode and an ingest panel for uploading and managing documents.

Built with React 18, TypeScript, Vite, shadcn/ui, and Tailwind CSS.

## Setup

```bash
npm install
npm run dev        # http://localhost:8080
```

The backend API URL comes from `VITE_API_URL` (see `.env.example`, defaults to
`http://localhost:8000`). Start the backend first; see the repo root README.

## Scripts

- `npm run dev` - development server
- `npm run build` - production build (tsc is not part of the build; run
  `npx tsc --noEmit` to typecheck)
- `npm run lint` - ESLint

## Structure

- `src/pages/Dashboard.tsx` - the app: query input, mode toggle, results
- `src/components/AnswerView.tsx` - answer with `[n]` citations and verified badges
- `src/components/ConfidenceBreakdown.tsx` - retrieval / coverage / completeness
- `src/components/RetrievedChunkList.tsx` - ranked chunks with scores
- `src/components/IngestPanel.tsx` - upload and document management
- `src/lib/api.ts` - `/v1` API client; `src/lib/types.ts` mirrors the backend
  response schemas
