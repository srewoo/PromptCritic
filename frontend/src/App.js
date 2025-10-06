import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import Compare from "./pages/Compare";
import EvaluationDetail from "./pages/EvaluationDetail";
import { Toaster } from "./components/ui/toaster";
import "./App.css";

// API base URL - adjust this if your backend runs on a different port
export const API = process.env.REACT_APP_API_URL || "http://localhost:8000/api";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/history" element={<History />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/evaluation/:id" element={<EvaluationDetail />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        <Toaster />
      </div>
    </Router>
  );
}

export default App;

