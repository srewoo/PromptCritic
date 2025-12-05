"""
Cost Optimization Engine
Optimize prompts for token efficiency while maintaining quality
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of cost optimization"""
    original_prompt: str
    optimized_prompt: str
    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    percentage_saved: float
    quality_score: float
    monthly_cost_reduction: float
    strategies_applied: List[str]


class CostOptimizer:
    """Optimize prompts for token efficiency"""
    
    def __init__(self):
        self.token_costs = self._load_token_costs()
    
    def optimize_for_cost(
        self,
        prompt: str,
        quality_threshold: float = 0.9,
        target_model: str = "gpt-4o"
    ) -> OptimizationResult:
        """
        Optimize prompt for cost while maintaining quality
        
        Args:
            prompt: Original prompt
            quality_threshold: Minimum quality to maintain (0-1)
            target_model: Target LLM model for cost calculation
            
        Returns:
            OptimizationResult with savings analysis
        """
        original_tokens = self.count_tokens(prompt)
        original_quality = self._evaluate_quality(prompt)
        
        # Progressive compression strategies
        strategies = [
            ("remove_redundancy", self._remove_redundancy),
            ("compress_instructions", self._compress_instructions),
            ("optimize_examples", self._optimize_examples),
            ("restructure_format", self._restructure_format),
            ("use_references", self._use_references)
        ]
        
        optimized = prompt
        strategies_applied = []
        
        for strategy_name, strategy_func in strategies:
            candidate = strategy_func(optimized)
            candidate_quality = self._evaluate_quality(candidate)
            
            # Only apply if quality maintained
            if candidate_quality >= original_quality * quality_threshold:
                optimized = candidate
                strategies_applied.append(strategy_name)
                logger.info(f"Applied strategy: {strategy_name}")
            else:
                logger.info(f"Skipped strategy {strategy_name}: quality too low ({candidate_quality:.2f})")
                break
        
        optimized_tokens = self.count_tokens(optimized)
        tokens_saved = original_tokens - optimized_tokens
        percentage_saved = (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0
        
        # Calculate cost reduction
        monthly_reduction = self._calculate_cost_savings(
            original_tokens,
            optimized_tokens,
            target_model
        )
        
        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            tokens_saved=tokens_saved,
            percentage_saved=round(percentage_saved, 1),
            quality_score=self._evaluate_quality(optimized),
            monthly_cost_reduction=monthly_reduction,
            strategies_applied=strategies_applied
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count
        Rule of thumb: 1 token ≈ 4 characters for English text
        """
        # Simple estimation (could be replaced with tiktoken for accuracy)
        char_count = len(text)
        word_count = len(text.split())
        
        # Average between character-based and word-based estimates
        char_estimate = char_count // 4
        word_estimate = int(word_count * 1.3)  # ~1.3 tokens per word
        
        return (char_estimate + word_estimate) // 2
    
    def _remove_redundancy(self, prompt: str) -> str:
        """Remove redundant phrases and repetitions"""
        optimized = prompt
        
        # Remove repeated phrases
        lines = optimized.split('\n')
        seen_lines = set()
        unique_lines = []
        
        for line in lines:
            line_stripped = line.strip().lower()
            if line_stripped and line_stripped not in seen_lines:
                seen_lines.add(line_stripped)
                unique_lines.append(line)
            elif not line.strip():  # Keep empty lines for formatting
                unique_lines.append(line)
        
        optimized = '\n'.join(unique_lines)
        
        # Remove redundant words
        redundant_phrases = [
            (r'\s+please\s+note\s+that\s+', ' '),
            (r'\s+it\s+is\s+important\s+to\s+note\s+that\s+', ' '),
            (r'\s+you\s+should\s+also\s+', ' '),
            (r'\s+make\s+sure\s+(to\s+)?', ' '),
            (r'\s+be\s+sure\s+(to\s+)?', ' ')
        ]
        
        for pattern, replacement in redundant_phrases:
            optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
        
        return optimized
    
    def _compress_instructions(self, prompt: str) -> str:
        """Compress verbose instructions"""
        optimized = prompt
        
        # Compress common verbose patterns
        compressions = [
            (r'you\s+are\s+required\s+to\s+', 'you must '),
            (r'it\s+is\s+necessary\s+that\s+you\s+', 'you must '),
            (r'please\s+provide\s+me\s+with\s+', 'provide '),
            (r'I\s+would\s+like\s+you\s+to\s+', 'please '),
            (r'can\s+you\s+please\s+', 'please '),
            (r'in\s+order\s+to\s+', 'to '),
            (r'for\s+the\s+purpose\s+of\s+', 'to '),
            (r'with\s+regard\s+to\s+', 'regarding '),
            (r'in\s+the\s+event\s+that\s+', 'if '),
            (r'due\s+to\s+the\s+fact\s+that\s+', 'because '),
            (r'at\s+this\s+point\s+in\s+time\s+', 'now '),
            (r'in\s+a\s+timely\s+manner\s+', 'promptly ')
        ]
        
        for pattern, replacement in compressions:
            optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
        
        # Remove filler words
        filler_words = [
            r'\s+actually\s+',
            r'\s+basically\s+',
            r'\s+literally\s+',
            r'\s+obviously\s+',
            r'\s+simply\s+',
            r'\s+just\s+',
            r'\s+really\s+',
            r'\s+very\s+much\s+'
        ]
        
        for filler in filler_words:
            optimized = re.sub(filler, ' ', optimized, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        optimized = re.sub(r'\s+', ' ', optimized)
        optimized = re.sub(r'\n\s*\n\s*\n+', '\n\n', optimized)
        
        return optimized.strip()
    
    def _optimize_examples(self, prompt: str) -> str:
        """Optimize or consolidate examples"""
        optimized = prompt
        
        # Find example sections
        example_pattern = r'(example|instance|e\.g\.|for example)[:\s]+(.*?)(?=\n\n|\n#|\n\d+\.|\Z)'
        examples = list(re.finditer(example_pattern, prompt, re.IGNORECASE | re.DOTALL))
        
        if len(examples) > 3:
            # Keep only most important examples
            logger.info(f"Found {len(examples)} examples, keeping first 3")
            
            # Keep first 3 examples, remove others
            for match in examples[3:]:
                optimized = optimized.replace(match.group(0), '')
        
        # Compress example formatting
        optimized = re.sub(r'Example\s+\d+:', 'Ex:', optimized)
        optimized = re.sub(r'For\s+example:', 'E.g.:', optimized, flags=re.IGNORECASE)
        
        return optimized
    
    def _restructure_format(self, prompt: str) -> str:
        """Restructure for more efficient token usage"""
        optimized = prompt
        
        # Convert verbose headings to concise format
        optimized = re.sub(r'^#+\s+(.+?)\s*$', r'**\1**', optimized, flags=re.MULTILINE)
        
        # Convert bullet points to more compact format
        optimized = re.sub(r'^\s*[-•]\s+', '- ', optimized, flags=re.MULTILINE)
        
        # Consolidate spacing
        optimized = re.sub(r'\n\n\n+', '\n\n', optimized)
        
        # Remove trailing whitespace
        lines = [line.rstrip() for line in optimized.split('\n')]
        optimized = '\n'.join(lines)
        
        return optimized
    
    def _use_references(self, prompt: str) -> str:
        """Use references instead of repetition"""
        optimized = prompt
        
        # Find repeated long phrases (>20 chars)
        words = optimized.split()
        phrase_counts = {}
        
        for i in range(len(words) - 4):  # Check 5-word phrases
            phrase = ' '.join(words[i:i+5])
            if len(phrase) > 20:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Replace repeated phrases with references
        for phrase, count in phrase_counts.items():
            if count > 1:
                # Replace subsequent occurrences with shorter reference
                parts = optimized.split(phrase)
                if len(parts) > 2:
                    # Keep first occurrence, replace others
                    optimized = phrase.join([parts[0]] + ['(see above)' for _ in parts[1:]])
                    logger.info(f"Replaced {count-1} occurrences of repeated phrase")
        
        return optimized
    
    def _evaluate_quality(self, prompt: str) -> float:
        """
        Evaluate prompt quality (0-1)
        Simple heuristic based on structure and completeness
        """
        score = 0.5  # Base score
        
        # Length check (not too short, not too long)
        length = len(prompt)
        if 100 <= length <= 2000:
            score += 0.15
        elif length > 50:
            score += 0.05
        
        # Structure indicators
        has_structure = bool(re.search(r'(#|\d+\.|-)', prompt))
        if has_structure:
            score += 0.1
        
        # Clear instructions
        has_instructions = bool(re.search(r'(you\s+(must|should|will)|please|provide)', prompt, re.IGNORECASE))
        if has_instructions:
            score += 0.1
        
        # Context provided
        has_context = bool(re.search(r'(context|background|objective|goal)', prompt, re.IGNORECASE))
        if has_context:
            score += 0.1
        
        # Output specification
        has_output_spec = bool(re.search(r'(output|format|response|result)', prompt, re.IGNORECASE))
        if has_output_spec:
            score += 0.05
        
        return min(1.0, score)
    
    def _calculate_cost_savings(
        self,
        original_tokens: int,
        optimized_tokens: int,
        model: str,
        monthly_calls: int = 1000
    ) -> float:
        """Calculate monthly cost reduction"""
        tokens_saved = original_tokens - optimized_tokens
        
        # Get pricing for model
        pricing = self.token_costs.get(model, {"input": 2.50, "output": 10.00})
        
        # Calculate per-call savings (input tokens)
        cost_per_million_tokens = pricing["input"]
        savings_per_call = (tokens_saved / 1_000_000) * cost_per_million_tokens
        
        # Monthly savings
        monthly_savings = savings_per_call * monthly_calls
        
        return round(monthly_savings, 2)
    
    def _load_token_costs(self) -> Dict[str, Dict[str, float]]:
        """Load token pricing per model (per 1M tokens)"""
        return {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "claude-3-7-sonnet": {"input": 3.00, "output": 15.00},
            "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
            "claude-3-opus": {"input": 15.00, "output": 75.00},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
            "gemini-1.5-pro": {"input": 1.25, "output": 5.00}
        }
    
    def analyze_cost_breakdown(
        self,
        prompt: str,
        model: str = "gpt-4o",
        monthly_calls: int = 1000
    ) -> Dict[str, Any]:
        """Provide detailed cost analysis"""
        tokens = self.count_tokens(prompt)
        pricing = self.token_costs.get(model, {"input": 2.50, "output": 10.00})
        
        # Calculate costs
        cost_per_call = (tokens / 1_000_000) * pricing["input"]
        monthly_cost = cost_per_call * monthly_calls
        annual_cost = monthly_cost * 12
        
        # Estimate output tokens (typically 2-3x input for detailed responses)
        estimated_output_tokens = tokens * 2.5
        output_cost_per_call = (estimated_output_tokens / 1_000_000) * pricing["output"]
        monthly_output_cost = output_cost_per_call * monthly_calls
        
        total_monthly_cost = monthly_cost + monthly_output_cost
        
        return {
            "model": model,
            "prompt_tokens": tokens,
            "estimated_output_tokens": int(estimated_output_tokens),
            "pricing": pricing,
            "costs": {
                "per_call": {
                    "input": round(cost_per_call, 6),
                    "output": round(output_cost_per_call, 6),
                    "total": round(cost_per_call + output_cost_per_call, 6)
                },
                "monthly": {
                    "calls": monthly_calls,
                    "input": round(monthly_cost, 2),
                    "output": round(monthly_output_cost, 2),
                    "total": round(total_monthly_cost, 2)
                },
                "annual": {
                    "total": round(total_monthly_cost * 12, 2)
                }
            },
            "optimization_potential": {
                "current_efficiency": "baseline",
                "tokens_per_dollar": round(1_000_000 / pricing["input"], 0)
            }
        }


def get_cost_optimizer() -> CostOptimizer:
    """Get singleton cost optimizer instance"""
    return CostOptimizer()
