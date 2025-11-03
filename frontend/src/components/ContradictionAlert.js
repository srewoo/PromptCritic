import React from 'react';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Badge } from './ui/badge';
import { AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

const ContradictionAlert = ({ contradictionAnalysis }) => {
  if (!contradictionAnalysis) {
    return null;
  }

  const { has_contradictions, contradictions, severity, recommendations } = contradictionAnalysis;

  const getSeverityColor = (sev) => {
    switch (sev) {
      case 'high':
        return 'destructive';
      case 'medium':
        return 'default';
      case 'low':
        return 'secondary';
      default:
        return 'outline';
    }
  };

  const getSeverityIcon = (sev) => {
    switch (sev) {
      case 'high':
        return <XCircle className="w-5 h-5 text-red-600" />;
      case 'medium':
        return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
      case 'low':
        return <AlertTriangle className="w-5 h-5 text-blue-600" />;
      default:
        return <CheckCircle className="w-5 h-5 text-green-600" />;
    }
  };

  if (!has_contradictions) {
    return (
      <Alert className="border-green-200 bg-green-50 dark:bg-green-950 dark:border-green-800">
        <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
        <AlertTitle className="text-green-800 dark:text-green-200">No Contradictions Detected</AlertTitle>
        <AlertDescription className="text-green-700 dark:text-green-300">
          Your prompt has consistent instructions with no conflicting requirements.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <Card className="w-full border-red-200 dark:border-red-800">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {getSeverityIcon(severity)}
          <span>Contradictions Detected</span>
          <Badge variant={getSeverityColor(severity)} className="ml-auto">
            {severity.toUpperCase()} SEVERITY
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {contradictions && contradictions.map((contradiction, index) => (
          <div key={index} className="p-4 rounded-lg border bg-card space-y-3">
            <div className="flex items-start gap-2">
              <Badge variant="outline" className="mt-0.5">
                {contradiction.type?.replace(/_/g, ' ').toUpperCase()}
              </Badge>
            </div>
            
            <div className="space-y-2">
              <div className="p-2 rounded bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800">
                <p className="text-sm font-medium text-red-900 dark:text-red-100">
                  ❌ {contradiction.instruction_1}
                </p>
              </div>
              <div className="p-2 rounded bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800">
                <p className="text-sm font-medium text-red-900 dark:text-red-100">
                  ❌ {contradiction.instruction_2}
                </p>
              </div>
            </div>

            {contradiction.explanation && (
              <div className="text-sm text-muted-foreground">
                <strong>Why this is problematic:</strong> {contradiction.explanation}
              </div>
            )}

            {contradiction.suggestion && (
              <div className="p-3 rounded-lg bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800">
                <p className="text-sm font-medium text-blue-900 dark:text-blue-100">
                  💡 <strong>Suggestion:</strong> {contradiction.suggestion}
                </p>
              </div>
            )}
          </div>
        ))}

        {recommendations && recommendations.length > 0 && (
          <div className="mt-4 p-4 rounded-lg bg-muted">
            <h4 className="font-semibold mb-2 flex items-center gap-2">
              <span>🔧</span>
              How to Fix:
            </h4>
            <ul className="space-y-1">
              {recommendations.map((rec, index) => (
                <li key={index} className="text-sm text-muted-foreground pl-5 relative">
                  <span className="absolute left-0">•</span>
                  {rec}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ContradictionAlert;
