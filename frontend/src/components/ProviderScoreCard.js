import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Lightbulb } from 'lucide-react';

const ProviderScoreCard = ({ providerScores }) => {
  if (!providerScores || providerScores.length === 0) {
    return null;
  }

  const getProviderIcon = (provider) => {
    switch (provider.toLowerCase()) {
      case 'openai':
        return '🤖';
      case 'claude':
        return '🧠';
      case 'gemini':
        return '✨';
      default:
        return '🔮';
    }
  };

  const getScoreColor = (percentage) => {
    if (percentage >= 80) return 'text-green-600 dark:text-green-400';
    if (percentage >= 60) return 'text-blue-600 dark:text-blue-400';
    if (percentage >= 40) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getProgressColor = (percentage) => {
    if (percentage >= 80) return 'bg-green-500';
    if (percentage >= 60) return 'bg-blue-500';
    if (percentage >= 40) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="text-2xl">🎯</span>
          Provider Optimization Scores
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          How well your prompt is optimized for each LLM provider
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {providerScores.map((provider, index) => (
          <div key={index} className="space-y-3 p-4 rounded-lg border bg-card">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-2xl">{getProviderIcon(provider.provider)}</span>
                <h3 className="font-semibold">{provider.provider}</h3>
              </div>
              <span className={`font-bold text-lg ${getScoreColor(provider.percentage)}`}>
                {provider.percentage.toFixed(0)}%
              </span>
            </div>
            
            <Progress 
              value={provider.percentage} 
              className="h-2"
              indicatorClassName={getProgressColor(provider.percentage)}
            />
            
            {provider.recommendations && provider.recommendations.length > 0 && (
              <div className="space-y-2 mt-3">
                <div className="flex items-center gap-1 text-sm font-medium text-muted-foreground">
                  <Lightbulb className="w-4 h-4" />
                  <span>Recommendations:</span>
                </div>
                <ul className="space-y-1">
                  {provider.recommendations.map((rec, recIndex) => (
                    <li key={recIndex} className="text-sm text-muted-foreground pl-5 relative">
                      <span className="absolute left-0">•</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  );
};

export default ProviderScoreCard;
