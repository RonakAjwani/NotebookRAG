import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Loader2, Search } from "lucide-react";
import { AnswerView } from "@/components/AnswerView";
import { ConfidenceBreakdown } from "@/components/ConfidenceBreakdown";
import { RetrievedChunkList } from "@/components/RetrievedChunkList";
import { IngestPanel } from "@/components/IngestPanel";
import { api } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import type { AnswerResponse, RetrievalMode } from "@/lib/types";

const MODES: RetrievalMode[] = ["hybrid", "dense", "sparse"];

function ResultCard({
  title,
  answer,
}: {
  title: string;
  answer: AnswerResponse;
}) {
  const [highlight, setHighlight] = useState<number | null>(null);
  return (
    <Card className="p-5 space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">{title}</h2>
      </div>
      <AnswerView answer={answer} onCitationClick={setHighlight} />
      {answer.confidence && <ConfidenceBreakdown confidence={answer.confidence} />}
      <div>
        <h4 className="text-xs font-medium uppercase tracking-wide text-muted-foreground mb-2">
          Retrieved chunks
        </h4>
        <RetrievedChunkList chunks={answer.retrieved} highlightMarker={highlight} />
      </div>
    </Card>
  );
}

export default function Dashboard() {
  const { toast } = useToast();
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [compare, setCompare] = useState(false);
  const [loading, setLoading] = useState(false);
  const [single, setSingle] = useState<AnswerResponse | null>(null);
  const [pair, setPair] = useState<{ hybrid: AnswerResponse; dense: AnswerResponse } | null>(null);

  const ask = async () => {
    if (!question.trim() || loading) return;
    setLoading(true);
    setSingle(null);
    setPair(null);
    try {
      if (compare) {
        const [hybrid, dense] = await Promise.all([
          api.ask(question, "hybrid"),
          api.ask(question, "dense"),
        ]);
        setPair({ hybrid, dense });
      } else {
        setSingle(await api.ask(question, mode));
      }
    } catch (e: unknown) {
      toast({
        title: "Query failed",
        description: e instanceof Error ? e.message : "Is the backend running?",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto max-w-7xl px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-6">
          {/* Query + results */}
          <div className="space-y-6">
            <Card className="p-5 space-y-4">
              <Textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question about your internal docs..."
                className="min-h-[80px] text-base resize-none"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
                }}
              />
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-2">
                  {MODES.map((m) => (
                    <Button
                      key={m}
                      size="sm"
                      variant={mode === m && !compare ? "default" : "outline"}
                      disabled={compare}
                      onClick={() => setMode(m)}
                      className="capitalize"
                    >
                      {m}
                    </Button>
                  ))}
                </div>
                <div className="flex items-center gap-2">
                  <Switch id="compare" checked={compare} onCheckedChange={setCompare} />
                  <Label htmlFor="compare" className="text-sm">Compare hybrid vs dense</Label>
                </div>
              </div>
              <Button onClick={ask} disabled={loading || !question.trim()} className="w-full" size="lg">
                {loading ? <Loader2 className="w-5 h-5 mr-2 animate-spin" /> : <Search className="w-5 h-5 mr-2" />}
                {loading ? "Retrieving..." : "Ask"}
              </Button>
            </Card>

            {single && <ResultCard title={`${single.mode} retrieval`} answer={single} />}

            {pair && (
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <ResultCard title="Hybrid" answer={pair.hybrid} />
                <ResultCard title="Dense only" answer={pair.dense} />
              </div>
            )}

            {!single && !pair && !loading && (
              <Card className="p-10 text-center text-muted-foreground">
                <Search className="w-10 h-10 mx-auto mb-3 opacity-30" />
                <p>Ask a question to see grounded, cited answers.</p>
              </Card>
            )}
          </div>

          {/* Sidebar: ingest / documents */}
          <aside>
            <Card className="p-5 sticky top-24">
              <IngestPanel />
            </Card>
          </aside>
        </div>
      </main>
    </div>
  );
}
