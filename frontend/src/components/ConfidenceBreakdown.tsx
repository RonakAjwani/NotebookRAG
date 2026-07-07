import { Progress } from "@/components/ui/progress";
import type { ConfidenceBreakdown as Conf } from "@/lib/types";

const DIMENSIONS: { key: keyof Conf; label: string; hint: string }[] = [
  { key: "retrieval", label: "Retrieval", hint: "Relevance of the evidence retrieved" },
  { key: "citation_coverage", label: "Citation coverage", hint: "Claims backed by a verified citation" },
  { key: "completeness", label: "Completeness", hint: "How fully the question was answered" },
];

function pct(v: number) {
  return `${Math.round(v * 100)}%`;
}

function toneClass(v: number) {
  if (v >= 0.66) return "text-emerald-600 dark:text-emerald-400";
  if (v >= 0.33) return "text-amber-600 dark:text-amber-400";
  return "text-red-600 dark:text-red-400";
}

export function ConfidenceBreakdown({ confidence }: { confidence: Conf }) {
  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <h3 className="text-sm font-medium text-muted-foreground">Confidence</h3>
        <span className={`text-2xl font-semibold ${toneClass(confidence.composite)}`}>
          {pct(confidence.composite)}
        </span>
      </div>
      <div className="space-y-3">
        {DIMENSIONS.map(({ key, label, hint }) => {
          const value = confidence[key] as number;
          return (
            <div key={key} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span title={hint}>{label}</span>
                <span className={`font-mono ${toneClass(value)}`}>{pct(value)}</span>
              </div>
              <Progress value={value * 100} className="h-1.5" />
            </div>
          );
        })}
      </div>
    </div>
  );
}
