import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';

const CategoryScoreCard = ({ categoryScores }) => {
  if (!categoryScores || categoryScores.length === 0) {
    return null;
  }

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

  const getRating = (percentage) => {
    if (percentage >= 80) return 'Excellent';
    if (percentage >= 60) return 'Good';
    if (percentage >= 40) return 'Fair';
    return 'Needs Work';
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="text-2xl">📊</span>
          Category Breakdown
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {categoryScores.map((category, index) => (
          <div key={index} className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-sm">{category.category}</h3>
                <Badge variant="outline" className={getScoreColor(category.percentage)}>
                  {getRating(category.percentage)}
                </Badge>
              </div>
              <span className={`font-bold ${getScoreColor(category.percentage)}`}>
                {category.score}/{category.max_score}
              </span>
            </div>
            <div className="space-y-1">
              <Progress 
                value={category.percentage} 
                className="h-2"
                indicatorClassName={getProgressColor(category.percentage)}
              />
              <p className="text-xs text-muted-foreground text-right">
                {category.percentage.toFixed(1)}%
              </p>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
};

export default CategoryScoreCard;
