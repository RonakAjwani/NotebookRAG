import { Fragment } from "react";
import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { AnswerResponse } from "@/lib/types";

interface Props {
  answer: AnswerResponse;
  onCitationClick?: (marker: number) => void;
}

// Split answer text into pieces, turning [n] markers into clickable buttons.
function renderWithCitations(text: string, onClick?: (m: number) => void) {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      const marker = Number(match[1]);
      return (
        <button
          key={i}
          onClick={() => onClick?.(marker)}
          className="mx-0.5 inline-flex items-center rounded bg-primary/15 px-1 text-xs font-mono text-primary hover:bg-primary/30 transition-colors align-baseline"
        >
          {marker}
        </button>
      );
    }
    return <Fragment key={i}>{part}</Fragment>;
  });
}

export function AnswerView({ answer, onCitationClick }: Props) {
  if (!answer.answered) {
    return (
      <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-4">
        <div className="flex items-center gap-2 mb-2 text-amber-600 dark:text-amber-400">
          <AlertTriangle className="w-4 h-4" />
          <span className="text-sm font-medium">Not enough evidence to answer</span>
        </div>
        <p className="text-sm leading-relaxed">{answer.answer}</p>
        {answer.suggested_sources.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {answer.suggested_sources.map((s) => (
              <Badge key={s} variant="outline" className="text-xs">{s}</Badge>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-base leading-relaxed whitespace-pre-wrap">
        {renderWithCitations(answer.answer, onCitationClick)}
      </p>

      {answer.citations.length > 0 && (
        <div className="space-y-1.5">
          <h4 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Citations
          </h4>
          {answer.citations.map((c) => (
            <button
              key={c.marker}
              onClick={() => onCitationClick?.(c.marker)}
              className="flex w-full items-center gap-2 rounded-md border px-2.5 py-1.5 text-left text-sm hover:bg-muted/50 transition-colors"
            >
              <span className="font-mono text-xs bg-primary/15 px-1.5 py-0.5 rounded">[{c.marker}]</span>
              <span className="flex-1 truncate">{c.source}{c.section ? ` - ${c.section}` : ""}</span>
              {c.verified === true && <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />}
              {c.verified === false && <XCircle className="w-4 h-4 text-red-500 shrink-0" />}
            </button>
          ))}
          <p className="text-xs text-muted-foreground pt-1">
            <CheckCircle2 className="inline w-3 h-3 text-emerald-500" /> verified &nbsp;
            <XCircle className="inline w-3 h-3 text-red-500" /> unsupported by cited source
          </p>
        </div>
      )}
    </div>
  );
}
