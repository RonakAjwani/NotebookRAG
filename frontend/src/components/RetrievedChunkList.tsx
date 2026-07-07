import { Badge } from "@/components/ui/badge";
import { FileText } from "lucide-react";
import type { RetrievedChunk } from "@/lib/types";

interface Props {
  chunks: RetrievedChunk[];
  highlightMarker?: number | null;
}

function scoreLabel(rc: RetrievedChunk): string {
  if (rc.rerank_score != null) return `rerank ${rc.rerank_score.toFixed(2)}`;
  if (rc.fused_score != null) return `fused ${rc.fused_score.toFixed(3)}`;
  return "";
}

export function RetrievedChunkList({ chunks, highlightMarker }: Props) {
  if (chunks.length === 0) {
    return <p className="text-sm text-muted-foreground">No chunks retrieved.</p>;
  }
  return (
    <div className="space-y-3">
      {chunks.map((rc, idx) => {
        const marker = idx + 1;
        const highlighted = highlightMarker === marker;
        return (
          <div
            key={rc.chunk.id || idx}
            id={`chunk-${marker}`}
            className={`rounded-lg border p-3 transition-colors ${
              highlighted ? "border-primary bg-primary/5" : "border-border"
            }`}
          >
            <div className="mb-2 flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 min-w-0">
                <span className="font-mono text-xs bg-primary/15 px-1.5 py-0.5 rounded">
                  [{marker}]
                </span>
                <FileText className="w-3.5 h-3.5 opacity-50 shrink-0" />
                <span className="text-sm font-medium truncate">{rc.chunk.source}</span>
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                {rc.chunk.page != null && (
                  <Badge variant="outline" className="text-xs">p.{rc.chunk.page}</Badge>
                )}
                <Badge variant="secondary" className="text-xs font-mono">{scoreLabel(rc)}</Badge>
              </div>
            </div>
            {rc.chunk.section && (
              <p className="text-xs text-muted-foreground mb-1 truncate">§ {rc.chunk.section}</p>
            )}
            <p className="text-sm leading-relaxed text-muted-foreground line-clamp-4">
              {rc.chunk.text}
            </p>
          </div>
        );
      })}
    </div>
  );
}
