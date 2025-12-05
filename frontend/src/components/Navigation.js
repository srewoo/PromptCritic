import React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "./ui/button";
import { Home, History, Settings, Sparkles } from "lucide-react";

const Navigation = () => {
  const navigate = useNavigate();

  return (
    <div className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">
              PromptCritic
            </h1>
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate("/")}
                className="text-slate-300 hover:text-white"
              >
                <Home className="w-4 h-4 mr-2" />
                Dashboard
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate("/optimizer")}
                className="text-slate-300 hover:text-white"
              >
                <Sparkles className="w-4 h-4 mr-2" />
                Optimizer
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate("/history")}
                className="text-slate-300 hover:text-white"
              >
                <History className="w-4 h-4 mr-2" />
                History
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Navigation;
