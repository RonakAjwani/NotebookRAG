// Typed client for the hybrid RAG backend (/v1).

import type {
  AnswerResponse,
  DocumentInfo,
  IngestResponse,
  RetrievalMode,
} from "./types";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // non-JSON error body; keep statusText
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  async ask(
    question: string,
    mode: RetrievalMode = "hybrid",
    verifyCitations = true,
  ): Promise<AnswerResponse> {
    const res = await fetch(`${BASE_URL}/v1/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode, verify_citations: verifyCitations }),
    });
    return handle<AnswerResponse>(res);
  },

  async ingest(files: File[], reset = false): Promise<IngestResponse> {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    const url = new URL(`${BASE_URL}/v1/ingest`);
    if (reset) url.searchParams.set("reset", "true");
    const res = await fetch(url.toString(), { method: "POST", body: form });
    return handle<IngestResponse>(res);
  },

  async listDocuments(): Promise<DocumentInfo[]> {
    const res = await fetch(`${BASE_URL}/v1/documents`);
    return handle<DocumentInfo[]>(res);
  },

  async deleteDocument(docId: string): Promise<void> {
    const res = await fetch(`${BASE_URL}/v1/documents/${docId}`, { method: "DELETE" });
    return handle<void>(res);
  },
};
