import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import Compare from "./pages/Compare";
import EvaluationDetail from "./pages/EvaluationDetail";
import Playground from "./pages/Playground";
import Help from "./pages/Help";
import ContradictionDetector from "./pages/ContradictionDetector";
import DelimiterAnalyzer from "./pages/DelimiterAnalyzer";
import MetapromptGenerator from "./pages/MetapromptGenerator";
import ABTesting from "./pages/ABTesting";
import { Toaster } from "./components/ui/toaster";
import { ThemeProvider } from "./components/theme-provider";
import "./App.css";

// API base URL - adjust this if your backend runs on a different port
export const API = process.env.REACT_APP_API_URL || "http://localhost:8000/api";

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="promptcritic-theme">
      <Router>
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/history" element={<History />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/playground" element={<Playground />} />
            <Route path="/contradiction-detector" element={<ContradictionDetector />} />
            <Route path="/delimiter-analyzer" element={<DelimiterAnalyzer />} />
            <Route path="/metaprompt-generator" element={<MetapromptGenerator />} />
            <Route path="/ab-testing" element={<ABTesting />} />
            <Route path="/help" element={<Help />} />
            <Route path="/evaluation/:id" element={<EvaluationDetail />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
          <Toaster />
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;

