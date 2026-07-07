// Mirrors backend/app/models/schemas.py response shapes.

export type RetrievalMode = "dense" | "sparse" | "hybrid";

export interface Chunk {
  id: string;
  text: string;
  source: string;
  doc_id: string;
  chunk_index: number;
  section: string | null;
  page: number | null;
  strategy: string;
  char_count: number;
}

export interface RetrievedChunk {
  chunk: Chunk;
  dense_score: number | null;
  sparse_score: number | null;
  fused_score: number | null;
  rerank_score: number | null;
}

export interface Citation {
  marker: number;
  doc_id: string;
  source: string;
  section: string | null;
  page: number | null;
  text: string;
  verified: boolean | null;
}

export interface ConfidenceBreakdown {
  retrieval: number;
  citation_coverage: number;
  completeness: number;
  composite: number;
}

export interface AnswerResponse {
  question: string;
  answer: string;
  answered: boolean;
  mode: RetrievalMode;
  citations: Citation[];
  confidence: ConfidenceBreakdown | null;
  retrieved: RetrievedChunk[];
  suggested_sources: string[];
  metadata: Record<string, unknown>;
}

export interface DocumentInfo {
  doc_id: string;
  source: string;
  chunk_count: number;
  strategies: string[];
}

export interface IngestResponse {
  documents: DocumentInfo[];
  chunks_indexed: number;
  duplicates_skipped: number;
}
