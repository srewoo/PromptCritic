import { useState, useEffect } from "react";
import axios from "axios";
import { API } from "../App";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Download, Loader2, TrendingUp, TrendingDown } from "lucide-react";
import { useToast } from "../hooks/use-toast";
import { Badge } from "../components/ui/badge";
import { Progress } from "../components/ui/progress";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "../components/ui/accordion";

export default function EvaluationDetail() {
  const { id } = useParams();
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const { toast } = useToast();

  useEffect(() => {
    loadEvaluation();
  }, [id]);

  const loadEvaluation = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/evaluations/${id}`);
      setEvaluation(response.data);
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to load evaluation"
      });
      navigate("/history");
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 4) return "text-emerald-400";
    if (score >= 3) return "text-blue-400";
    if (score >= 2) return "text-amber-400";
    return "text-red-400";
  };

  const getScoreBg = (score) => {
    if (score >= 4) return "bg-emerald-500";
    if (score >= 3) return "bg-blue-500";
    if (score >= 2) return "bg-amber-500";
    return "bg-red-500";
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString();
  };

  if (loading) {
    return (
      <div className="min-h-screen p-6 flex items-center justify-center">
        <Loader2 className="h-12 w-12 animate-spin text-blue-500" />
      </div>
    );
  }

  if (!evaluation) {
    return null;
  }

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-6xl mx-auto">
        <Button
          variant="ghost"
          onClick={() => navigate("/history")}
          className="mb-4 text-slate-400 hover:text-white"
          data-testid="back-button"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to History
        </Button>

        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-4xl font-bold text-white">Evaluation Details</h1>
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={() => window.open(`${API}/export/json/${evaluation.id}`, '_blank')}
                className="bg-slate-800 border-slate-600 hover:bg-slate-700 text-white"
                data-testid="export-json-button"
              >
                <Download className="mr-2 h-4 w-4" />
                JSON
              </Button>
              <Button
                variant="outline"
                onClick={() => window.open(`${API}/export/pdf/${evaluation.id}`, '_blank')}
                className="bg-slate-800 border-slate-600 hover:bg-slate-700 text-white"
                data-testid="export-pdf-button"
              >
                <Download className="mr-2 h-4 w-4" />
                PDF
              </Button>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="bg-slate-800 border-slate-600 text-slate-300">
              {evaluation.llm_provider}
            </Badge>
            <span className="text-slate-500 text-sm">{formatDate(evaluation.created_at)}</span>
          </div>
        </div>

        {/* Score Overview */}
        <Card className="bg-slate-800/50 border-slate-700 mb-6" data-testid="score-overview">
          <CardHeader>
            <CardTitle className="text-white">Overall Score</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-6xl font-bold text-blue-400">
                  {evaluation.total_score}
                  <span className="text-3xl text-slate-500">/175</span>
                </p>
                <p className="text-slate-400 mt-2">
                  {Math.round((evaluation.total_score / 175) * 100)}% Quality Score
                </p>
              </div>
              <div className="text-right">
                <p className="text-slate-400 text-sm mb-2">Performance</p>
                <Progress
                  value={(evaluation.total_score / 175) * 100}
                  className="h-4 w-48 bg-slate-700"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Original Prompt */}
        <Card className="bg-slate-800/50 border-slate-700 mb-6">
          <CardHeader>
            <CardTitle className="text-white">Original Prompt</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-slate-900 p-4 rounded-lg border border-slate-700">
              <pre className="text-slate-300 text-sm font-mono whitespace-pre-wrap">
                {evaluation.prompt_text}
              </pre>
            </div>
          </CardContent>
        </Card>

        {/* Criteria Scores */}
        <Card className="bg-slate-800/50 border-slate-700 mb-6">
          <CardHeader>
            <CardTitle className="text-white">Detailed Criteria Analysis</CardTitle>
            <CardDescription className="text-slate-400">
              35 expert criteria evaluated
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Accordion type="single" collapsible className="space-y-2" data-testid="criteria-list">
              {evaluation.criteria_scores.map((criterion, idx) => (
                <AccordionItem
                  key={idx}
                  value={`item-${idx}`}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-4"
                >
                  <AccordionTrigger className="hover:no-underline py-4">
                    <div className="flex items-center justify-between w-full pr-4">
                      <div className="flex items-center gap-3">
                        <span className="text-slate-500 font-mono text-sm">#{idx + 1}</span>
                        <span className="text-white font-medium">{criterion.criterion}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className={`w-12 h-12 rounded-lg ${getScoreBg(criterion.score)} flex items-center justify-center`}>
                          <span className="text-white font-bold text-lg">{criterion.score}</span>
                        </div>
                      </div>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="pb-4 pt-2 space-y-3">
                    <div className="bg-slate-800 p-3 rounded-lg">
                      <div className="flex items-start gap-2">
                        <TrendingUp className="h-4 w-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="text-emerald-400 text-xs font-semibold mb-1">Strength</p>
                          <p className="text-slate-300 text-sm">{criterion.strength}</p>
                        </div>
                      </div>
                    </div>
                    <div className="bg-slate-800 p-3 rounded-lg">
                      <div className="flex items-start gap-2">
                        <TrendingDown className="h-4 w-4 text-amber-400 mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="text-amber-400 text-xs font-semibold mb-1">Improvement</p>
                          <p className="text-slate-300 text-sm">{criterion.improvement}</p>
                        </div>
                      </div>
                    </div>
                    <div className="bg-slate-800 p-3 rounded-lg">
                      <p className="text-slate-400 text-xs font-semibold mb-1">Rationale</p>
                      <p className="text-slate-300 text-sm">{criterion.rationale}</p>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </CardContent>
        </Card>

        {/* Refinement Suggestions */}
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">Refinement Suggestions</CardTitle>
            <CardDescription className="text-slate-400">
              Actionable improvements for your prompt
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3" data-testid="suggestions-list">
              {evaluation.refinement_suggestions.map((suggestion, idx) => (
                <div key={idx} className="bg-slate-900 p-4 rounded-lg border border-slate-700">
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">
                      {idx + 1}
                    </div>
                    <p className="text-slate-300 text-sm flex-1">{suggestion}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}