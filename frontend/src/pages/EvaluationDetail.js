import { useState, useEffect } from "react";
import axios from "axios";
import { API } from "../App";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Download, Loader2, TrendingUp, TrendingDown, Sparkles, Copy, DollarSign } from "lucide-react";
import { useToast } from "../hooks/use-toast";
import { Badge } from "../components/ui/badge";
import { Progress } from "../components/ui/progress";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "../components/ui/accordion";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "../components/ui/dialog";
import { Textarea } from "../components/ui/textarea";

export default function EvaluationDetail() {
  const { id } = useParams();
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [rewriting, setRewriting] = useState(false);
  const [rewriteResult, setRewriteResult] = useState(null);
  const [showRewriteDialog, setShowRewriteDialog] = useState(false);
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

  const handleRewrite = async () => {
    if (!evaluation) return;
    
    setRewriting(true);
    try {
      const response = await axios.post(`${API}/rewrite`, {
        prompt_text: evaluation.prompt_text,
        evaluation_id: evaluation.id
      });
      setRewriteResult(response.data);
      setShowRewriteDialog(true);
      toast({
        title: "Prompt Rewritten!",
        description: "AI has improved your prompt based on the evaluation"
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Rewrite Failed",
        description: error.response?.data?.detail || "An error occurred"
      });
    } finally {
      setRewriting(false);
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied!",
      description: "Text copied to clipboard"
    });
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
                onClick={handleRewrite}
                disabled={rewriting}
                className="bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white"
              >
                {rewriting ? (
                  <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Rewriting...</>
                ) : (
                  <><Sparkles className="mr-2 h-4 w-4" /> AI Rewrite</>
                )}
              </Button>
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

        {/* Cost Information */}
        {evaluation.cost && (
          <Card className="bg-slate-800/50 border-slate-700 mb-6">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <DollarSign className="h-5 w-5" />
                Cost Breakdown
              </CardTitle>
              <CardDescription className="text-slate-400">
                Estimated API costs for this evaluation
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-slate-500 text-sm">Input Tokens</p>
                  <p className="text-white font-semibold text-lg">{evaluation.cost.input_tokens.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-slate-500 text-sm">Output Tokens</p>
                  <p className="text-white font-semibold text-lg">{evaluation.cost.output_tokens.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-slate-500 text-sm">Input Cost</p>
                  <p className="text-white font-semibold text-lg">${evaluation.cost.input_cost.toFixed(6)}</p>
                </div>
                <div>
                  <p className="text-slate-500 text-sm">Total Cost</p>
                  <p className="text-blue-400 font-bold text-xl">${evaluation.cost.total_cost.toFixed(6)}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

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

      {/* Rewrite Dialog */}
      <Dialog open={showRewriteDialog} onOpenChange={setShowRewriteDialog}>
        <DialogContent className="bg-slate-800 border-slate-700 text-white max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-purple-500" />
              AI-Rewritten Prompt
            </DialogTitle>
            <DialogDescription className="text-slate-400">
              Improved version based on evaluation feedback
            </DialogDescription>
          </DialogHeader>
          
          {rewriteResult && (
            <div className="space-y-4">
              {/* Rewritten Prompt */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-white">Improved Prompt</h3>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => copyToClipboard(rewriteResult.rewritten_prompt)}
                    className="bg-slate-700 border-slate-600"
                  >
                    <Copy className="h-3 w-3 mr-2" />
                    Copy
                  </Button>
                </div>
                <Textarea
                  value={rewriteResult.rewritten_prompt}
                  readOnly
                  className="min-h-[200px] bg-slate-900 border-slate-600 text-slate-200 font-mono text-sm"
                />
              </div>

              {/* Changes Made */}
              <div>
                <h3 className="font-semibold text-white mb-2">Key Improvements</h3>
                <ul className="space-y-2">
                  {rewriteResult.changes_made?.map((change, i) => (
                    <li key={i} className="flex gap-2 text-slate-300 text-sm">
                      <span className="text-green-500">✓</span>
                      <span>{change}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Rationale */}
              {rewriteResult.rationale && (
                <div>
                  <h3 className="font-semibold text-white mb-2">Rationale</h3>
                  <p className="text-slate-300 text-sm">{rewriteResult.rationale}</p>
                </div>
              )}

              {/* Cost */}
              {rewriteResult.cost && (
                <Card className="bg-slate-900 border-slate-700">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm text-white">Rewrite Cost</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm">
                    <div className="flex justify-between text-slate-300">
                      <span>Total Cost:</span>
                      <span className="font-bold text-blue-400">${rewriteResult.cost.total_cost.toFixed(6)}</span>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}