import { useState } from "react";
import axios from "axios";
import { API } from "../App";
import { Button } from "../components/ui/button";
import { Textarea } from "../components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Play, DollarSign, Loader2 } from "lucide-react";
import { useToast } from "../hooks/use-toast";

export default function Playground() {
  const [promptText, setPromptText] = useState("Write a {input}");
  const [testInput, setTestInput] = useState("");
  const [response, setResponse] = useState(null);
  const [testing, setTesting] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleTest = async () => {
    if (!promptText.trim() || !testInput.trim()) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please provide both prompt and test input"
      });
      return;
    }

    setTesting(true);
    try {
      const result = await axios.post(`${API}/playground`, {
        prompt_text: promptText,
        test_input: testInput
      }, {
        timeout: 90000  // 1.5 minute timeout (90 seconds)
      });
      setResponse(result.data);
      toast({
        title: "Test Complete!",
        description: "Your prompt has been executed successfully"
      });
    } catch (error) {
      const errorMessage = error.code === 'ECONNABORTED'
        ? "Request timed out. The test took too long. Try a simpler prompt or try again."
        : error.response?.data?.detail || "An error occurred";
      
      toast({
        variant: "destructive",
        title: "Test Failed",
        description: errorMessage
      });
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Button
              variant="outline"
              onClick={() => navigate("/")}
              className="bg-slate-800 border-slate-600 hover:bg-slate-700 text-white"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
            <div>
              <h1 className="text-4xl font-bold text-white">Prompt Playground</h1>
              <p className="text-slate-400 text-lg">Test your prompts with live input</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Section */}
          <div className="space-y-6">
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">Prompt Template</CardTitle>
                <CardDescription className="text-slate-400">
                  Use {"{input}"} as a placeholder for test data
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Textarea
                  placeholder="Enter your prompt template..."
                  value={promptText}
                  onChange={(e) => setPromptText(e.target.value)}
                  className="min-h-[200px] bg-slate-900 border-slate-600 text-white font-mono text-sm"
                />
              </CardContent>
            </Card>

            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">Test Input</CardTitle>
                <CardDescription className="text-slate-400">
                  This will replace {"{input}"} in your prompt
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label className="text-slate-200">Test Data</Label>
                  <Input
                    placeholder="e.g., short story about a robot"
                    value={testInput}
                    onChange={(e) => setTestInput(e.target.value)}
                    className="bg-slate-900 border-slate-600 text-white mt-2"
                  />
                </div>
                <Button
                  onClick={handleTest}
                  disabled={testing || !promptText.trim() || !testInput.trim()}
                  className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white py-6"
                >
                  {testing ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Testing...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-5 w-5" />
                      Test Prompt
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Response Section */}
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Response</CardTitle>
              <CardDescription className="text-slate-400">
                AI-generated output from your prompt
              </CardDescription>
            </CardHeader>
            <CardContent>
              {testing && (
                <div className="flex flex-col items-center justify-center h-[400px]">
                  <Loader2 className="h-12 w-12 animate-spin text-blue-500 mb-4" />
                  <p className="text-slate-400">Generating response...</p>
                </div>
              )}

              {!testing && !response && (
                <div className="flex flex-col items-center justify-center h-[400px] text-slate-500">
                  <Play className="h-16 w-16 mb-4 opacity-30" />
                  <p className="text-lg">No response yet</p>
                  <p className="text-sm">Test your prompt to see results</p>
                </div>
              )}

              {!testing && response && (
                <div className="space-y-4">
                  <div className="bg-slate-900 p-4 rounded-lg border border-slate-700">
                    <p className="text-slate-300 whitespace-pre-wrap">{response.response}</p>
                  </div>

                  {/* Cost Information */}
                  {response.cost && (
                    <Card className="bg-slate-900 border-slate-700">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm flex items-center gap-2 text-white">
                          <DollarSign className="h-4 w-4" />
                          Cost Breakdown
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        <div className="flex justify-between text-slate-300">
                          <span>Input Tokens:</span>
                          <span>{response.cost.input_tokens.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between text-slate-300">
                          <span>Output Tokens:</span>
                          <span>{response.cost.output_tokens.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between text-slate-300">
                          <span>Input Cost:</span>
                          <span>${response.cost.input_cost.toFixed(6)}</span>
                        </div>
                        <div className="flex justify-between text-slate-300">
                          <span>Output Cost:</span>
                          <span>${response.cost.output_cost.toFixed(6)}</span>
                        </div>
                        <div className="flex justify-between font-bold text-blue-400 pt-2 border-t border-slate-700">
                          <span>Total Cost:</span>
                          <span>${response.cost.total_cost.toFixed(6)}</span>
                        </div>
                        <p className="text-xs text-slate-500 pt-2">
                          Provider: {response.provider} | Model: {response.model}
                        </p>
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

