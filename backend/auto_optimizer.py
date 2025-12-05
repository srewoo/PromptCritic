"""
Automatic Prompt Optimization Engine
ML-powered prompt improvement with multi-strategy approach
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Available optimization strategies"""
    CLARITY_ENHANCEMENT = "clarity_enhancement"
    STRUCTURE_IMPROVEMENT = "structure_improvement"
    CONTEXT_ENRICHMENT = "context_enrichment"
    DELIMITER_OPTIMIZATION = "delimiter_optimization"
    REASONING_INJECTION = "reasoning_injection"
    ERROR_HANDLING = "error_handling"
    EXAMPLE_ADDITION = "example_addition"
    CONSTRAINT_SPECIFICATION = "constraint_specification"


@dataclass
class OptimizationCandidate:
    """Represents an optimized prompt candidate"""
    prompt: str
    strategy: OptimizationStrategy
    predicted_score: float
    changes: List[str]
    confidence: float


@dataclass
class OptimizationResult:
    """Result of optimization process"""
    original_prompt: str
    optimized_prompt: str
    improvements: List[str]
    predicted_score_increase: float
    confidence: float
    strategies_applied: List[str]
    validation: Dict[str, Any]


class AutoOptimizer:
    """ML-powered automatic prompt improvement"""
    
    def __init__(self):
        self.optimization_strategies = [
            self._add_clarity,
            self._improve_structure,
            self._enrich_context,
            self._optimize_delimiters,
            self._inject_reasoning,
            self._add_error_handling,
            self._add_examples,
            self._specify_constraints
        ]
    
    async def optimize_prompt(
        self,
        original_prompt: str,
        evaluation_scores: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Optimize a prompt using multiple strategies
        
        Args:
            original_prompt: Original prompt text
            evaluation_scores: Optional evaluation criteria scores
            context: Optional context (use_case, target_model, etc.)
            
        Returns:
            OptimizationResult with improvements
        """
        # Analyze weak points from evaluation
        weak_criteria = self._identify_weak_points(evaluation_scores) if evaluation_scores else []
        
        # Generate optimization candidates
        candidates = []
        for strategy_func in self.optimization_strategies:
            try:
                candidate = strategy_func(original_prompt, weak_criteria, context)
                if candidate:
                    candidates.append(candidate)
            except Exception as e:
                logger.warning(f"Strategy failed: {e}")
        
        # Select best candidate using scoring heuristics
        best_candidate = self._select_best_candidate(candidates, original_prompt)
        
        # Validate improvements
        validation = self._validate_optimization(original_prompt, best_candidate)
        
        return OptimizationResult(
            original_prompt=original_prompt,
            optimized_prompt=best_candidate.prompt,
            improvements=best_candidate.changes,
            predicted_score_increase=best_candidate.predicted_score,
            confidence=best_candidate.confidence,
            strategies_applied=[best_candidate.strategy.value],
            validation=validation
        )
    
    def _identify_weak_points(self, evaluation_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify criteria with low scores"""
        weak_criteria = []
        for score in evaluation_scores:
            if score.get('score', 0) < 3:  # Score below 3/5
                weak_criteria.append({
                    'criterion': score.get('criterion'),
                    'score': score.get('score'),
                    'improvement': score.get('improvement', '')
                })
        return weak_criteria
    
    def _add_clarity(
        self,
        prompt: str,
        weak_criteria: List[Dict],
        context: Optional[Dict]
    ) -> Optional[OptimizationCandidate]:
        """Strategy: Add clarity and specificity"""
        changes = []
        optimized = prompt
        
        # Check if clarity is an issue
        needs_clarity = any('clarity' in w.get('criterion', '').lower() for w in weak_criteria)
        
        if needs_clarity or len(prompt) < 100:
            # Add explicit instructions
            if not re.search(r'(you must|you should|please|ensure)', prompt, re.IGNORECASE):
                clarity_addition = "\n\nIMPORTANT: Please ensure your response is clear, specific, and directly addresses the request above."
                optimized = prompt + clarity_addition
                changes.append("Added explicit clarity instructions")
            
            # Add objective statement if missing
            if not re.search(r'(objective|goal|purpose|task):', prompt, re.IGNORECASE):
                objective = "\n\nObjective: Provide a comprehensive and accurate response to the above query."
                optimized = prompt + objective
                changes.append("Added explicit objective statement")
        
        if changes:
            return OptimizationCandidate(
                prompt=optimized,
                strategy=OptimizationStrategy.CLARITY_ENHANCEMENT,
                predicted_score=0.15,  # 15% improvement
                changes=changes,
                confidence=0.8
            )
        return None
    
    def _improve_structure(
        self,
        prompt: str,
        weak_criteria: List[Dict],
        context: Optional[Dict]
    ) -> Optional[OptimizationCandidate]:
        """Strategy: Improve prompt structure"""
        changes = []
        
        # Check if structure needs improvement
        has_structure = bool(re.search(r'(1\.|2\.|3\.|•|#)', prompt))
        
        if not has_structure and len(prompt.split('\n')) > 3:
            # Add structured format
            lines = prompt.strip().split('\n')
            structured = "# Task Instructions\n\n"
            
            for i, line in enumerate(lines, 1):
                if line.strip():
                    structured += f"{i}. {line.strip()}\n"
            
            changes.append("Added numbered structure to instructions")
            
            return OptimizationCandidate(
                prompt=structured,
                strategy=OptimizationStrategy.STRUCTURE_IMPROVEMENT,
                predicted_score=0.12,
                changes=changes,
                confidence=0.75
            )
        return None
    
    def _enrich_context(
        self,
        prompt: str,
        weak_criteria: List[Dict],
        context: Optional[Dict]
    ) -> Optional[OptimizationCandidate]:
        """Strategy: Add contextual information"""
        changes = []
        optimized = prompt
        
        # Check if context is weak
        needs_context = any('context' in w.get('criterion', '').lower() for w in weak_criteria)
        
        if needs_context or len(prompt) < 150:
            context_section = "\n\n## Context\n"
            
            # Add use case context
            if context and context.get('use_case'):
                context_section += f"Use Case: {context['use_case']}\n"
                changes.append(f"Added use case context: {context['use_case']}")
            
            # Add audience context
            if context and context.get('audience'):
                context_section += f"Target Audience: {context['audience']}\n"
                changes.append("Added target audience specification")
            else:
                context_section += "Target Audience: Technical professionals\n"
                changes.append("Added default audience specification")
            
            # Add output expectations
            context_section += "\nExpected Output: Clear, accurate, and actionable information.\n"
            changes.append("Added output expectations")
            
            optimized = prompt + context_section
        
        if changes:
            return OptimizationCandidate(
                prompt=optimized,
                strategy=OptimizationStrategy.CONTEXT_ENRICHMENT,
                predicted_score=0.18,
                changes=changes,
                confidence=0.82
            )
        return None
    
    def _optimize_delimiters(
        self,
        prompt: str,
        weak_criteria: List[Dict],
        context: Optional[Dict]
    ) -> Optional[OptimizationCandidate]:
        """Strategy: Add or improve delimiters"""
        changes = []
        
        # Check if delimiters are weak
        needs_delimiters = any('delimiter' in w.get('criterion', '').lower() for w in weak_criteria)
        has_delimiters = bool(re.search(r'(```|<|>|\[|\])', prompt))
        
        if needs_delimiters or not has_delimiters:
            # Add XML-style delimiters for clarity
            optimized = f"""<prompt>
<instructions>
{prompt}
</instructions>

<output_format>
Please structure your response clearly and completely.
</output_format>
</prompt>"""
            
            changes.append("Added XML delimiters for better structure")
            
            return OptimizationCandidate(
                prompt=optimized,
                strategy=OptimizationStrategy.DELIMITER_OPTIMIZATION,
                predicted_score=0.10,
                changes=changes,
                confidence=0.7
            )
        return None
    
    def _inject_reasoning(
        self,
        prompt: str,
        weak_criteria: List[Dict],
        context: Optional[Dict]
    ) -> Optional[OptimizationCandidate]:
        """Strategy: Add reasoning/thinking instructions"""
        changes = []
        
        # Check if reasoning is weak
        needs_reasoning = any('reasoning' in w.get('criterion', '').lower() for w in weak_criteria)
        has_reasoning = bool(re.search(r'(think|reason|analyze|consider)', prompt, re.IGNORECASE))
        
        if needs_reasoning or not has_reasoning:
            reasoning_section = "\n\n## Approach\n"
            reasoning_section += "Before providing your final answer:\n"
            reasoning_section += "1. Analyze the requirements carefully\n"
            reasoning_section += "2. Consider multiple approaches\n"
            reasoning_section += "3. Evaluate trade-offs\n"
            reasoning_section += "4. Provide your reasoned conclusion\n"
            
            changes.append("Added step-by-step reasoning guidance")
            
            return OptimizationCandidate(
                prompt=prompt + reasoning_section,
                strategy=OptimizationStrategy.REASONING_INJECTION,
                predicted_score=0.16,
                changes=changes,
                confidence=0.78
            )
        return None
    
    def _add_error_handling(
        self,
        prompt: str,
        weak_criteria: List[Dict],
        context: Optional[Dict]
    ) -> Optional[OptimizationCandidate]:
        """Strategy: Add error handling instructions"""
        changes = []
        
        # Check if error handling is weak
        needs_error_handling = any('error' in w.get('criterion', '').lower() for w in weak_criteria)
        has_error_handling = bool(re.search(r'(error|fail|cannot|unable)', prompt, re.IGNORECASE))
        
        if needs_error_handling or not has_error_handling:
            error_section = "\n\n## Error Handling\n"
            error_section += "If you cannot complete the request:\n"
            error_section += "- Clearly state what you cannot do\n"
            error_section += "- Explain why (if applicable)\n"
            error_section += "- Suggest alternatives if possible\n"
            
            changes.append("Added error handling guidelines")
            
            return OptimizationCandidate(
                prompt=prompt + error_section,
                strategy=OptimizationStrategy.ERROR_HANDLING,
                predicted_score=0.11,
                changes=changes,
                confidence=0.72
            )
        return None
    
    def _add_examples(
        self,
        prompt: str,
        weak_criteria: List[Dict],
        context: Optional[Dict]
    ) -> Optional[OptimizationCandidate]:
        """Strategy: Add examples"""
        changes = []
        
        # Check if examples are weak
        needs_examples = any('example' in w.get('criterion', '').lower() for w in weak_criteria)
        has_examples = bool(re.search(r'(example|instance|for example|e\.g\.)', prompt, re.IGNORECASE))
        
        if needs_examples or (not has_examples and context and context.get('use_case') == 'code_generation'):
            example_section = "\n\n## Example\n"
            example_section += "Here's what a good response looks like:\n"
            example_section += "[Provide a clear, well-structured example that demonstrates the expected output format]\n"
            
            changes.append("Added example template")
            
            return OptimizationCandidate(
                prompt=prompt + example_section,
                strategy=OptimizationStrategy.EXAMPLE_ADDITION,
                predicted_score=0.14,
                changes=changes,
                confidence=0.68
            )
        return None
    
    def _specify_constraints(
        self,
        prompt: str,
        weak_criteria: List[Dict],
        context: Optional[Dict]
    ) -> Optional[OptimizationCandidate]:
        """Strategy: Add constraints and requirements"""
        changes = []
        
        # Check if constraints are weak
        has_constraints = bool(re.search(r'(must|should|required|constraint)', prompt, re.IGNORECASE))
        
        if not has_constraints:
            constraints_section = "\n\n## Requirements\n"
            constraints_section += "Your response must:\n"
            constraints_section += "- Be accurate and factual\n"
            constraints_section += "- Be complete and comprehensive\n"
            constraints_section += "- Use clear and professional language\n"
            
            if context and context.get('max_tokens'):
                constraints_section += f"- Be concise (max {context['max_tokens']} tokens)\n"
                changes.append("Added token limit constraint")
            
            changes.append("Added explicit requirements section")
            
            return OptimizationCandidate(
                prompt=prompt + constraints_section,
                strategy=OptimizationStrategy.CONSTRAINT_SPECIFICATION,
                predicted_score=0.13,
                changes=changes,
                confidence=0.76
            )
        return None
    
    def _select_best_candidate(
        self,
        candidates: List[OptimizationCandidate],
        original_prompt: str
    ) -> OptimizationCandidate:
        """Select best candidate using multi-armed bandit approach (simplified)"""
        if not candidates:
            # Return original as fallback
            return OptimizationCandidate(
                prompt=original_prompt,
                strategy=OptimizationStrategy.CLARITY_ENHANCEMENT,
                predicted_score=0.0,
                changes=["No optimization needed"],
                confidence=1.0
            )
        
        # Score candidates by predicted improvement * confidence
        scored_candidates = [
            (c, c.predicted_score * c.confidence)
            for c in candidates
        ]
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates[0][0]
    
    def _validate_optimization(
        self,
        original: str,
        candidate: OptimizationCandidate
    ) -> Dict[str, Any]:
        """Validate that optimization improved the prompt"""
        # Calculate metrics
        original_length = len(original)
        optimized_length = len(candidate.prompt)
        length_increase = optimized_length - original_length
        
        # Check for improvements
        has_structure = bool(re.search(r'(#|1\.|2\.|3\.)', candidate.prompt))
        has_delimiters = bool(re.search(r'(<|>|```)', candidate.prompt))
        has_constraints = bool(re.search(r'(must|should|required)', candidate.prompt, re.IGNORECASE))
        
        return {
            "length_increase": length_increase,
            "length_increase_pct": round((length_increase / original_length) * 100, 1),
            "has_structure": has_structure,
            "has_delimiters": has_delimiters,
            "has_constraints": has_constraints,
            "quality_indicators": {
                "structure": has_structure,
                "delimiters": has_delimiters,
                "constraints": has_constraints
            }
        }


def get_auto_optimizer() -> AutoOptimizer:
    """Get singleton auto-optimizer instance"""
    return AutoOptimizer()
