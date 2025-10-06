import { useState, useEffect } from "react";
import axios from "axios";
import { API } from "../App";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Loader2, GitCompare } from "lucide-react";
import { useToast } from "../hooks/use-toast";
import { Checkbox } from "../components/ui/checkbox";
import { Badge } from "../components/ui/badge";

export default function Compare() {
  const [evaluations, setEvaluations] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [comparing, setComparing] = useState(false);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const { toast } = useToast();

  useEffect(() => {
    loadEvaluations();
  }, []);

  const loadEvaluations = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/evaluations?limit=50`);
      setEvaluations(response.data);
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to load evaluations"
      });
    } finally {
      setLoading(false);
    }
  };

  const handleToggle = (id) => {
    if (selectedIds.includes(id)) {
      setSelectedIds(selectedIds.filter(i => i !== id));
    } else {
      if (selectedIds.length >= 5) {
        toast({
          variant: "destructive",
          title: "Limit Reached",
          description: "You can compare up to 5 evaluations at once"
        });
        return;
      }
      setSelectedIds([...selectedIds, id]);
    }
  };

  const handleCompare = async () => {
    if (selectedIds.length < 2) {
      toast({
        variant: "destructive",
        title: "Selection Required",
        description: "Please select at least 2 evaluations to compare"
      });
      return;
    }

    setComparing(true);
    try {
      const response = await axios.post(`${API}/compare`, {
        evaluation_ids: selectedIds
      });
      setComparison(response.data);
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to compare evaluations"
      });
    } finally {
      setComparing(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 140) return "text-emerald-400";
    if (score >= 100) return "text-blue-400";
    if (score >= 70) return "text-amber-400";
    return "text-red-400";
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString();
  };

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        <Button
          variant="ghost"
          onClick={() => navigate("/")}
          className="mb-4 text-slate-400 hover:text-white"
          data-testid="back-button"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Dashboard
        </Button>

        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Compare Evaluations</h1>
          <p className="text-slate-400 text-lg">
            Select 2-5 evaluations to compare side-by-side
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Selection Panel */}
          <Card className="bg-slate-800/50 border-slate-700 h-fit">
            <CardHeader>
              <CardTitle className="text-white">Select Evaluations</CardTitle>
              <CardDescription className="text-slate-400">
                Choose {selectedIds.length}/5 evaluations
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading && (
                <div className="flex justify-center py-8">
                  <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
                </div>
              )}

              {!loading && evaluations.length === 0 && (
                <p className="text-slate-400 text-center py-8">No evaluations available</p>
              )}

              {!loading && evaluations.length > 0 && (
                <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
                  {evaluations.map((evaluation) => (
                    <div
                      key={evaluation.id}
                      className={`p-4 rounded-lg border cursor-pointer transition-all ${
                        selectedIds.includes(evaluation.id)
                          ? 'bg-blue-900/30 border-blue-600'
                          : 'bg-slate-900 border-slate-700 hover:border-slate-600'
                      }`}
                      onClick={() => handleToggle(evaluation.id)}
                      data-testid={`eval-checkbox-${evaluation.id}`}
                    >
                      <div className="flex items-start gap-3">
                        <Checkbox
                          checked={selectedIds.includes(evaluation.id)}
                          onCheckedChange={() => handleToggle(evaluation.id)}
                          className="mt-1"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className={`font-bold ${getScoreColor(evaluation.total_score)}`}>
                              {evaluation.total_score}/175
                            </span>
                            <Badge variant="outline" className="bg-slate-800 border-slate-600 text-xs">
                              {evaluation.llm_provider}
                            </Badge>
                          </div>
                          <p className="text-slate-400 text-xs mb-1">{formatDate(evaluation.created_at)}</p>
                          <p className="text-slate-300 text-sm line-clamp-2 font-mono">
                            {evaluation.prompt_text}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <Button
                onClick={handleCompare}
                disabled={selectedIds.length < 2 || comparing}
                className="w-full mt-4 bg-blue-600 hover:bg-blue-700 text-white"
                data-testid="compare-button"
              >
                {comparing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Comparing...
                  </>
                ) : (
                  <>
                    <GitCompare className="mr-2 h-4 w-4" />
                    Compare Selected
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Comparison Results */}
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Comparison Results</CardTitle>
              <CardDescription className="text-slate-400">
                Side-by-side analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!comparison && (
                <div className="flex flex-col items-center justify-center h-[400px] text-slate-500">
                  <GitCompare className="h-16 w-16 mb-4 opacity-30" />
                  <p className="text-lg">No comparison yet</p>
                  <p className="text-sm">Select evaluations and click Compare</p>
                </div>
              )}

              {comparison && (
                <div className="space-y-6" data-testid="comparison-results">
                  {/* Summary Stats */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-slate-900 p-4 rounded-lg border border-slate-700 text-center">
                      <p className="text-slate-400 text-xs mb-1">Average</p>
                      <p className={`text-2xl font-bold ${getScoreColor(comparison.summary.avg_score)}`}>
                        {Math.round(comparison.summary.avg_score)}
                      </p>
                    </div>
                    <div className="bg-slate-900 p-4 rounded-lg border border-slate-700 text-center">
                      <p className="text-slate-400 text-xs mb-1">Best</p>
                      <p className="text-2xl font-bold text-emerald-400">
                        {comparison.summary.max_score}
                      </p>
                    </div>
                    <div className="bg-slate-900 p-4 rounded-lg border border-slate-700 text-center">
                      <p className="text-slate-400 text-xs mb-1">Lowest</p>
                      <p className="text-2xl font-bold text-red-400">
                        {comparison.summary.min_score}
                      </p>
                    </div>
                  </div>

                  {/* Individual Evaluations */}
                  <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
                    {comparison.evaluations.map((evaluation, idx) => (
                      <Card key={evaluation.id} className="bg-slate-900 border-slate-700">
                        <CardContent className="p-4">
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-2">
                              <span className="text-slate-500 font-mono text-sm">#{idx + 1}</span>
                              <Badge variant="outline" className="bg-slate-800 border-slate-600">
                                {evaluation.llm_provider}
                              </Badge>
                            </div>
                            <span className={`text-2xl font-bold ${getScoreColor(evaluation.total_score)}`}>
                              {evaluation.total_score}
                            </span>
                          </div>
                          <p className="text-slate-400 text-xs mb-2">{formatDate(evaluation.created_at)}</p>
                          <p className="text-slate-300 text-sm font-mono line-clamp-2">
                            {evaluation.prompt_text}
                          </p>
                          <div className="mt-3 space-y-1">
                            <p className="text-slate-400 text-xs font-semibold">Top Suggestions:</p>
                            {evaluation.refinement_suggestions.slice(0, 2).map((suggestion, sidx) => (
                              <p key={sidx} className="text-slate-500 text-xs pl-2">
                                • {suggestion.substring(0, 60)}...
                              </p>
                            ))}
                          </div>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => navigate(`/evaluation/${evaluation.id}`)}
                            className="w-full mt-3 bg-slate-800 border-slate-600 hover:bg-slate-700 text-white text-xs"
                          >
                            View Full Details
                          </Button>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}