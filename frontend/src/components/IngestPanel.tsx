import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Upload, Trash2, Loader2, FileText } from "lucide-react";
import { api } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import type { DocumentInfo } from "@/lib/types";

export function IngestPanel() {
  const { toast } = useToast();
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = async () => {
    try {
      setDocs(await api.listDocuments());
    } catch (e) {
      // backend may be down; leave list empty
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setUploading(true);
    try {
      const res = await api.ingest(Array.from(files));
      toast({
        title: "Ingested",
        description: `${res.chunks_indexed} chunks indexed, ${res.duplicates_skipped} duplicates skipped.`,
      });
      await refresh();
    } catch (e: unknown) {
      toast({
        title: "Ingest failed",
        description: e instanceof Error ? e.message : "Unknown error",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      await api.deleteDocument(docId);
      await refresh();
    } catch (e: unknown) {
      toast({ title: "Delete failed", variant: "destructive" });
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Indexed documents</h3>
        <Badge variant="secondary">{docs.length}</Badge>
      </div>

      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".pdf,.txt,.md,.markdown,.html,.htm"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
      <Button
        variant="outline"
        className="w-full"
        disabled={uploading}
        onClick={() => inputRef.current?.click()}
      >
        {uploading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Upload className="w-4 h-4 mr-2" />}
        {uploading ? "Ingesting..." : "Upload documents"}
      </Button>

      <div className="space-y-1.5 max-h-[320px] overflow-y-auto">
        {docs.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-6">
            No documents indexed yet.
          </p>
        ) : (
          docs.map((d) => (
            <div key={d.doc_id} className="flex items-center gap-2 rounded-md border px-2.5 py-1.5">
              <FileText className="w-3.5 h-3.5 opacity-50 shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm truncate">{d.source}</p>
                <p className="text-xs text-muted-foreground">
                  {d.chunk_count} chunks | {d.strategies.join(", ") || "-"}
                </p>
              </div>
              <Button variant="ghost" size="icon" className="h-7 w-7 shrink-0" onClick={() => handleDelete(d.doc_id)}>
                <Trash2 className="w-3.5 h-3.5" />
              </Button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
