import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Badge } from './ui/badge';
import { Zap, Target, Brain, FileText, Layers } from 'lucide-react';

const EvaluationModeSelector = ({ selectedMode, onModeChange }) => {
  const modes = [
    {
      id: 'quick',
      name: 'Quick Scan',
      icon: Zap,
      color: 'from-yellow-500 to-orange-500',
      borderColor: 'border-yellow-500',
      criteria: '10 Critical Criteria',
      time: '~30s',
      description: 'Fast evaluation of essential prompt quality factors',
      maxScore: 50,
      bestFor: ['Quick checks', 'Rapid iteration', 'Initial assessment']
    },
    {
      id: 'standard',
      name: 'Standard',
      icon: Target,
      color: 'from-blue-500 to-cyan-500',
      borderColor: 'border-blue-500',
      criteria: '50 Comprehensive Criteria',
      time: '~60s',
      description: 'Complete evaluation across all 7 categories',
      maxScore: 250,
      bestFor: ['General prompts', 'Balanced analysis', 'Most use cases'],
      default: true
    },
    {
      id: 'deep',
      name: 'Deep Analysis',
      icon: Brain,
      color: 'from-purple-500 to-pink-500',
      borderColor: 'border-purple-500',
      criteria: '50 Criteria + Deep Insights',
      time: '~90s',
      description: 'Comprehensive evaluation with additional semantic and performance analysis',
      maxScore: 250,
      bestFor: ['Production prompts', 'Critical applications', 'Detailed optimization']
    },
    {
      id: 'agentic',
      name: 'Agentic Workflow',
      icon: Layers,
      color: 'from-green-500 to-emerald-500',
      borderColor: 'border-green-500',
      criteria: '50 Criteria (Agentic Focus)',
      time: '~60s',
      description: 'Specialized evaluation for multi-step workflows and tool use',
      maxScore: 250,
      bestFor: ['Agent systems', 'Tool calling', 'Multi-step tasks']
    },
    {
      id: 'long_context',
      name: 'Long Context',
      icon: FileText,
      color: 'from-indigo-500 to-violet-500',
      borderColor: 'border-indigo-500',
      criteria: '50 Criteria (Context Focus)',
      time: '~60s',
      description: 'Optimized for prompts handling 50K+ tokens with document processing',
      maxScore: 250,
      bestFor: ['RAG systems', 'Document analysis', 'Large context windows']
    }
  ];

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-white mb-2">Evaluation Mode</h3>
        <p className="text-sm text-slate-400">
          Choose the evaluation depth and focus area for your prompt
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {modes.map((mode) => {
          const Icon = mode.icon;
          const isSelected = selectedMode === mode.id;
          
          return (
            <Card
              key={mode.id}
              className={`cursor-pointer transition-all duration-200 ${
                isSelected
                  ? `${mode.borderColor} border-2 bg-slate-800/80`
                  : 'border-slate-700 hover:border-slate-600 bg-slate-800/50'
              }`}
              onClick={() => onModeChange(mode.id)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between mb-2">
                  <div className={`p-2 rounded-lg bg-gradient-to-br ${mode.color}`}>
                    <Icon className="w-5 h-5 text-white" />
                  </div>
                  {mode.default && !isSelected && (
                    <Badge variant="outline" className="text-xs">
                      Default
                    </Badge>
                  )}
                  {isSelected && (
                    <Badge className="bg-blue-600 text-white text-xs">
                      Selected
                    </Badge>
                  )}
                </div>
                <CardTitle className="text-white text-base">{mode.name}</CardTitle>
                <CardDescription className="text-xs text-slate-400">
                  {mode.description}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400">Criteria:</span>
                  <span className="text-white font-medium">{mode.criteria}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400">Max Score:</span>
                  <span className="text-white font-medium">{mode.maxScore}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400">Est. Time:</span>
                  <span className="text-white font-medium">{mode.time}</span>
                </div>
                <div className="pt-2 border-t border-slate-700">
                  <p className="text-xs text-slate-400 mb-1">Best for:</p>
                  <div className="flex flex-wrap gap-1">
                    {mode.bestFor.map((use, idx) => (
                      <Badge key={idx} variant="secondary" className="text-xs">
                        {use}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
};

export default EvaluationModeSelector;
