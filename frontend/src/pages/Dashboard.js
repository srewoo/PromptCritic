import { useState, useEffect } from "react";
import axios from "axios";
import { API } from "../App";
import { Button } from "../components/ui/button";
import { Textarea } from "../components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Label } from "../components/ui/label";
import { Input } from "../components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "../components/ui/dialog";
import { useNavigate } from "react-router-dom";
import { Loader2, Settings, History, GitCompare, Download, CheckCircle, AlertCircle, Play, Sparkles, DollarSign, Copy, HelpCircle } from "lucide-react";
import { useToast } from "../hooks/use-toast";
import { Progress } from "../components/ui/progress";

export default function Dashboard() {
  const [promptText, setPromptText] = useState("");
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [settings, setSettings] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [llmProvider, setLlmProvider] = useState("openai");
  const [apiKey, setApiKey] = useState("");
  const [modelName, setModelName] = useState("");
  const [latestEvaluation, setLatestEvaluation] = useState(null);
  const [rewriting, setRewriting] = useState(false);
  const [rewriteResult, setRewriteResult] = useState(null);
  const [showRewriteDialog, setShowRewriteDialog] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const response = await axios.get(`${API}/settings`);
      if (response.data) {
        setSettings(response.data);
        setLlmProvider(response.data.llm_provider);
        setApiKey(response.data.api_key);
        setModelName(response.data.model_name || "");
      }
    } catch (error) {
      console.error("Failed to load settings:", error);
    }
  };

  const saveSettings = async () => {
    if (!apiKey.trim()) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please provide an API key"
      });
      return;
    }

    try {
      const response = await axios.post(`${API}/settings`, {
        llm_provider: llmProvider,
        api_key: apiKey,
        model_name: modelName || null
      });
      setSettings(response.data);
      setShowSettings(false);
      toast({
        title: "Settings Saved",
        description: "Your LLM configuration has been updated successfully."
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to save settings"
      });
    }
  };

  const handleEvaluate = async () => {
    if (!promptText.trim()) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please enter a prompt to evaluate"
      });
      return;
    }

    if (!settings) {
      toast({
        variant: "destructive",
        title: "Configuration Required",
        description: "Please configure your LLM settings first"
      });
      setShowSettings(true);
      return;
    }

    setIsEvaluating(true);
    try {
      const response = await axios.post(`${API}/evaluate`, {
        prompt_text: promptText
      }, {
        timeout: 150000  // 2.5 minute timeout (150 seconds)
      });
      setLatestEvaluation(response.data);
      toast({
        title: "Evaluation Complete!",
        description: `Total Score: ${response.data.total_score}/175`
      });
    } catch (error) {
      const errorMessage = error.code === 'ECONNABORTED' 
        ? "Request timed out. The evaluation took too long. Try a shorter prompt or try again."
        : error.response?.data?.detail || "An error occurred during evaluation";
      
      toast({
        variant: "destructive",
        title: "Evaluation Failed",
        description: errorMessage
      });
    } finally {
      setIsEvaluating(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 140) return "text-emerald-400";
    if (score >= 100) return "text-blue-400";
    if (score >= 70) return "text-amber-400";
    return "text-red-400";
  };

  const getScoreLabel = (score) => {
    if (score >= 140) return "Excellent";
    if (score >= 100) return "Good";
    if (score >= 70) return "Fair";
    return "Needs Improvement";
  };

  const handleRewrite = async () => {
    if (!latestEvaluation) return;
    
    setRewriting(true);
    try {
      const response = await axios.post(`${API}/rewrite`, {
        prompt_text: promptText,
        evaluation_id: latestEvaluation.id
      }, {
        timeout: 150000  // 2.5 minute timeout (150 seconds)
      });
      setRewriteResult(response.data);
      setShowRewriteDialog(true);
      toast({
        title: "Prompt Rewritten!",
        description: "AI has improved your prompt based on the evaluation"
      });
    } catch (error) {
      const errorMessage = error.code === 'ECONNABORTED'
        ? "Request timed out. The rewrite took too long. Try again."
        : error.response?.data?.detail || "An error occurred";
      
      toast({
        variant: "destructive",
        title: "Rewrite Failed",
        description: errorMessage
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

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2" data-testid="dashboard-title">
              Prompt Critic
            </h1>
            <p className="text-slate-400 text-lg">
              Advanced AI-powered prompt evaluation using 35 expert criteria
            </p>
          </div>
          <div className="flex gap-3">
            <Button
              variant="outline"
              onClick={() => navigate("/playground")}
              className="bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 border-0 text-white"
              data-testid="playground-button"
            >
              <Play className="mr-2 h-4 w-4" />
              Playground
            </Button>
            <Button
              variant="outline"
              onClick={() => navigate("/history")}
              className="bg-slate-800 border-slate-600 hover:bg-slate-700 text-white"
              data-testid="history-button"
            >
              <History className="mr-2 h-4 w-4" />
              History
            </Button>
            <Button
              variant="outline"
              onClick={() => navigate("/compare")}
              className="bg-slate-800 border-slate-600 hover:bg-slate-700 text-white"
              data-testid="compare-button"
            >
              <GitCompare className="mr-2 h-4 w-4" />
              Compare
            </Button>
            <Button
              variant="outline"
              onClick={() => navigate("/help")}
              className="bg-slate-800 border-slate-600 hover:bg-slate-700 text-white"
              data-testid="help-button"
            >
              <HelpCircle className="mr-2 h-4 w-4" />
              Help
            </Button>
            <Dialog open={showSettings} onOpenChange={setShowSettings}>
              <DialogTrigger asChild>
                <Button
                  variant="outline"
                  className="bg-slate-800 border-slate-600 hover:bg-slate-700 text-white"
                  data-testid="settings-button"
                >
                  <Settings className="mr-2 h-4 w-4" />
                  Settings
                </Button>
              </DialogTrigger>
              <DialogContent className="bg-slate-800 border-slate-700 text-white max-w-md">
                <DialogHeader>
                  <DialogTitle className="text-2xl">LLM Configuration</DialogTitle>
                  <DialogDescription className="text-slate-400">
                    Configure your LLM provider and API key
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 mt-4">
                  <div>
                    <Label className="text-slate-200 mb-2 block">LLM Provider</Label>
                    <Select value={llmProvider} onValueChange={setLlmProvider}>
                      <SelectTrigger className="bg-slate-900 border-slate-600 text-white" data-testid="provider-select">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-slate-900 border-slate-700 text-white">
                        <SelectItem value="openai">OpenAI GPT-5</SelectItem>
                        <SelectItem value="claude">Claude Sonnet 4</SelectItem>
                        <SelectItem value="gemini">Gemini 2.5 Pro</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label className="text-slate-200 mb-2 block">API Key</Label>
                    <Input
                      type="password"
                      placeholder="Enter your API key"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      className="bg-slate-900 border-slate-600 text-white placeholder:text-slate-500"
                      data-testid="api-key-input"
                    />
                  </div>
                  <div>
                    <Label className="text-slate-200 mb-2 block">Model Name (Optional)</Label>
                    <Input
                      type="text"
                      placeholder="e.g., gpt-4o, claude-3-7-sonnet-20250219"
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      className="bg-slate-900 border-slate-600 text-white placeholder:text-slate-500"
                      data-testid="model-name-input"
                    />
                    <p className="text-xs text-slate-500 mt-1">Leave empty for default model</p>
                  </div>
                  <Button
                    onClick={saveSettings}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                    data-testid="save-settings-button"
                  >
                    Save Configuration
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </div>

        {/* Settings Alert */}
        {!settings && (
          <Card className="mb-6 bg-amber-900/20 border-amber-700">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <AlertCircle className="h-5 w-5 text-amber-500" />
                <div>
                  <p className="text-amber-200 font-medium">Configuration Required</p>
                  <p className="text-amber-300/80 text-sm">
                    Please configure your LLM settings before evaluating prompts
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Section */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white text-xl">Prompt Input</CardTitle>
              <CardDescription className="text-slate-400">
                Enter your prompt for comprehensive evaluation
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                placeholder="Enter your prompt here..."
                value={promptText}
                onChange={(e) => setPromptText(e.target.value)}
                className="min-h-[400px] bg-slate-900 border-slate-600 text-white placeholder:text-slate-500 font-mono text-sm resize-none"
                data-testid="prompt-input"
              />
              <Button
                onClick={handleEvaluate}
                disabled={isEvaluating || !promptText.trim()}
                className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white py-6 text-lg font-semibold"
                data-testid="evaluate-button"
              >
                {isEvaluating ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Evaluating...
                  </>
                ) : (
                  <>
                    <CheckCircle className="mr-2 h-5 w-5" />
                    Evaluate Prompt
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Results Section */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white text-xl">Evaluation Results</CardTitle>
              <CardDescription className="text-slate-400">
                AI-powered analysis with actionable insights
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isEvaluating && (
                <div className="flex flex-col items-center justify-center h-[400px] space-y-4">
                  <Loader2 className="h-12 w-12 animate-spin text-blue-500" />
                  <p className="text-slate-400">Analyzing your prompt...</p>
                </div>
              )}

              {!isEvaluating && !latestEvaluation && (
                <div className="flex flex-col items-center justify-center h-[400px] text-slate-500">
                  <CheckCircle className="h-16 w-16 mb-4 opacity-30" />
                  <p className="text-lg">No evaluation yet</p>
                  <p className="text-sm">Enter a prompt and click Evaluate</p>
                </div>
              )}

              {!isEvaluating && latestEvaluation && (
                <div className="space-y-6" data-testid="evaluation-results">
                  {/* Score Overview */}
                  <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <p className="text-slate-400 text-sm">Total Score</p>
                        <p className={`text-5xl font-bold ${getScoreColor(latestEvaluation.total_score)}`}>
                          {latestEvaluation.total_score}
                          <span className="text-2xl text-slate-500">/175</span>
                        </p>
                      </div>
                      <div className="text-right">
                        <p className={`text-xl font-semibold ${getScoreColor(latestEvaluation.total_score)}`}>
                          {getScoreLabel(latestEvaluation.total_score)}
                        </p>
                        <p className="text-slate-500 text-sm">
                          {Math.round((latestEvaluation.total_score / 175) * 100)}% Quality
                        </p>
                      </div>
                    </div>
                    <Progress
                      value={(latestEvaluation.total_score / 175) * 100}
                      className="h-2 bg-slate-800"
                    />
                  </div>

                  {/* Top Suggestions */}
                  <div>
                    <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                      <span className="text-blue-400">💡</span>
                      Top Refinement Suggestions
                    </h3>
                    <div className="space-y-2">
                      {latestEvaluation.refinement_suggestions.slice(0, 5).map((suggestion, idx) => (
                        <div key={idx} className="bg-slate-900 p-3 rounded-lg border border-slate-700">
                          <p className="text-slate-300 text-sm">{suggestion}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-2">
                    <Button
                      onClick={handleRewrite}
                      disabled={rewriting}
                      className="flex-1 bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white"
                    >
                      {rewriting ? (
                        <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Rewriting...</>
                      ) : (
                        <><Sparkles className="mr-2 h-4 w-4" /> AI Rewrite</>
                      )}
                    </Button>
                    <Button
                      onClick={() => navigate(`/evaluation/${latestEvaluation.id}`)}
                      className="flex-1 bg-slate-700 hover:bg-slate-600 text-white"
                      data-testid="view-details-button"
                    >
                      View Full Details
                    </Button>
                    <Button
                      onClick={() => window.open(`${API}/export/pdf/${latestEvaluation.id}`, '_blank')}
                      variant="outline"
                      className="bg-slate-900 border-slate-600 hover:bg-slate-700 text-white"
                      data-testid="export-pdf-button"
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <Card className="bg-slate-800/30 border-slate-700">
            <CardContent className="pt-6">
              <div className="text-center">
                <p className="text-3xl font-bold text-blue-400">35</p>
                <p className="text-slate-400 text-sm mt-1">Evaluation Criteria</p>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-slate-800/30 border-slate-700">
            <CardContent className="pt-6">
              <div className="text-center">
                <p className="text-3xl font-bold text-emerald-400">AI</p>
                <p className="text-slate-400 text-sm mt-1">Powered Analysis</p>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-slate-800/30 border-slate-700">
            <CardContent className="pt-6">
              <div className="text-center">
                <p className="text-3xl font-bold text-purple-400">∞</p>
                <p className="text-slate-400 text-sm mt-1">Unlimited Evaluations</p>
              </div>
            </CardContent>
          </Card>
        </div>
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
                    <CardTitle className="text-sm text-white flex items-center gap-2">
                      <DollarSign className="h-4 w-4" />
                      Rewrite Cost
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm">
                    <div className="flex justify-between text-slate-300">
                      <span>Total Cost:</span>
                      <span className="font-bold text-blue-400">${rewriteResult.cost.total_cost.toFixed(6)}</span>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Use Rewritten Prompt Button */}
              <Button
                onClick={() => {
                  setPromptText(rewriteResult.rewritten_prompt);
                  setShowRewriteDialog(false);
                  setLatestEvaluation(null);
                  toast({
                    title: "Prompt Updated!",
                    description: "You can now evaluate the improved prompt"
                  });
                }}
                className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white"
              >
                Use This Prompt
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}