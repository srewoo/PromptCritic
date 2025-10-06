import { useState, useEffect } from "react";
import axios from "axios";
import { API } from "../App";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Trash2, Eye, Download, Loader2 } from "lucide-react";
import { useToast } from "../hooks/use-toast";
import { Badge } from "../components/ui/badge";

export default function History() {
  const [evaluations, setEvaluations] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const { toast } = useToast();

  useEffect(() => {
    loadEvaluations();
  }, []);

  const loadEvaluations = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/evaluations?limit=100`);
      setEvaluations(response.data);
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to load evaluation history"
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm("Are you sure you want to delete this evaluation?")) {
      return;
    }

    try {
      await axios.delete(`${API}/evaluations/${id}`);
      setEvaluations(evaluations.filter(e => e.id !== id));
      toast({
        title: "Deleted",
        description: "Evaluation deleted successfully"
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to delete evaluation"
      });
    }
  };

  const getScoreColor = (score) => {
    if (score >= 140) return "bg-emerald-500";
    if (score >= 100) return "bg-blue-500";
    if (score >= 70) return "bg-amber-500";
    return "bg-red-500";
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString();
  };

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <Button
              variant="ghost"
              onClick={() => navigate("/")}
              className="mb-4 text-slate-400 hover:text-white"
              data-testid="back-button"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Dashboard
            </Button>
            <h1 className="text-4xl font-bold text-white mb-2">Evaluation History</h1>
            <p className="text-slate-400 text-lg">
              Browse and manage your past evaluations
            </p>
          </div>
        </div>

        {loading && (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="h-12 w-12 animate-spin text-blue-500" />
          </div>
        )}

        {!loading && evaluations.length === 0 && (
          <Card className="bg-slate-800/50 border-slate-700">
            <CardContent className="pt-12 pb-12 text-center">
              <p className="text-slate-400 text-lg">No evaluations yet</p>
              <p className="text-slate-500 text-sm mt-2">Start by evaluating your first prompt</p>
              <Button
                onClick={() => navigate("/")}
                className="mt-4 bg-blue-600 hover:bg-blue-700"
              >
                Create Evaluation
              </Button>
            </CardContent>
          </Card>
        )}

        {!loading && evaluations.length > 0 && (
          <div className="space-y-4" data-testid="evaluations-list">
            {evaluations.map((evaluation) => (
              <Card key={evaluation.id} className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-colors">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-3 mb-2">
                        <div className={`px-3 py-1 rounded-full text-white font-semibold text-sm ${getScoreColor(evaluation.total_score)}`}>
                          {evaluation.total_score}/175
                        </div>
                        <Badge variant="outline" className="bg-slate-900 border-slate-600 text-slate-300">
                          {evaluation.llm_provider}
                        </Badge>
                        <span className="text-slate-500 text-sm">
                          {formatDate(evaluation.created_at)}
                        </span>
                      </div>
                      <p className="text-slate-300 text-sm line-clamp-2 font-mono">
                        {evaluation.prompt_text}
                      </p>
                      <div className="mt-3 flex flex-wrap gap-2">
                        {evaluation.refinement_suggestions.slice(0, 2).map((suggestion, idx) => (
                          <span key={idx} className="text-xs bg-blue-900/30 text-blue-300 px-2 py-1 rounded">
                            💡 {suggestion.substring(0, 50)}...
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => navigate(`/evaluation/${evaluation.id}`)}
                        className="bg-slate-900 border-slate-600 hover:bg-slate-700 text-white"
                        data-testid={`view-eval-${evaluation.id}`}
                      >
                        <Eye className="h-4 w-4" />
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => window.open(`${API}/export/pdf/${evaluation.id}`, '_blank')}
                        className="bg-slate-900 border-slate-600 hover:bg-slate-700 text-white"
                        data-testid={`download-eval-${evaluation.id}`}
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleDelete(evaluation.id)}
                        className="bg-slate-900 border-slate-600 hover:bg-red-900/50 text-red-400"
                        data-testid={`delete-eval-${evaluation.id}`}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}