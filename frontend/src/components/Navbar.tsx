import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Layers, Moon, Sun } from "lucide-react";
import { useTheme } from "@/components/ThemeProvider";

export const Navbar = () => {
  const { theme, setTheme } = useTheme();

  return (
    <nav className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-6 h-16 flex items-center justify-between max-w-7xl">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center transition-transform group-hover:scale-105">
            <Layers className="w-5 h-5 text-primary-foreground" />
          </div>
          <div className="flex flex-col leading-tight">
            <span className="font-semibold text-lg">Hybrid RAG</span>
            <span className="text-xs text-muted-foreground">Internal docs, grounded answers</span>
          </div>
        </Link>

        <Button
          variant="ghost"
          size="icon"
          className="h-10 w-10"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          aria-label="Toggle theme"
        >
          {theme === "dark" ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </Button>
      </div>
    </nav>
  );
};
