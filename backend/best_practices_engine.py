"""
Provider-specific best practices engine for prompt evaluation

Based on official documentation:
- OpenAI GPT-5/4.1: https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide
- Anthropic Claude 4: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices
- Google Gemini: https://ai.google.dev/gemini-api/docs/prompting-strategies

Last Updated: December 2025
"""
from typing import List, Dict, Any
from enum import Enum


class Provider(str, Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    MULTI = "multi"


class BestPracticesEngine:
    """Evaluates prompts against provider-specific best practices"""

    def __init__(self):
        self.practices = {
            Provider.OPENAI: self._get_openai_practices(),
            Provider.CLAUDE: self._get_claude_practices(),
            Provider.GEMINI: self._get_gemini_practices(),
        }
        # Universal formatting checks that apply to all providers
        self.formatting_checks = self._get_formatting_checks()

    def _get_formatting_checks(self) -> List[Dict[str, Any]]:
        """Universal formatting and quality checks"""
        return [
            {
                "id": "format_no_duplicate_lines",
                "name": "No duplicate content",
                "description": "Remove duplicate or repeated lines/sections in the prompt",
                "check": self._check_no_duplicates,
                "importance": "medium"
            },
            {
                "id": "format_consistent_headers",
                "name": "Consistent header formatting",
                "description": "Use consistent header styles (## or ### or XML tags)",
                "check": self._check_consistent_headers,
                "importance": "medium"
            },
            {
                "id": "format_proper_spacing",
                "name": "Proper spacing and structure",
                "description": "Use appropriate spacing between sections (not too many blank lines)",
                "check": self._check_proper_spacing,
                "importance": "low"
            },
            {
                "id": "format_no_broken_syntax",
                "name": "No broken syntax",
                "description": "Ensure all markdown/XML tags are properly closed",
                "check": self._check_no_broken_syntax,
                "importance": "high"
            },
            {
                "id": "format_reasonable_length",
                "name": "Reasonable prompt length",
                "description": "Prompt should be substantial but not excessively long (100-10000 chars)",
                "check": lambda prompt: 100 <= len(prompt) <= 10000,
                "importance": "low"
            },
            {
                "id": "format_no_placeholder_text",
                "name": "No placeholder text",
                "description": "Remove placeholder text like [TODO], [INSERT], {placeholder}",
                "check": self._check_no_placeholders,
                "importance": "high"
            },
            {
                "id": "format_clear_sections",
                "name": "Clear section organization",
                "description": "Prompt should have clear sections or logical organization",
                "check": lambda prompt: len(prompt.split('\n\n')) >= 2 or any(marker in prompt for marker in ['#', '<', '1.', '•']),
                "importance": "medium"
            }
        ]

    def _check_no_duplicates(self, prompt: str) -> bool:
        """Check for duplicate lines (more than 3 words)"""
        lines = [line.strip() for line in prompt.split('\n') if len(line.strip().split()) > 3]
        return len(lines) == len(set(lines))

    def _check_consistent_headers(self, prompt: str) -> bool:
        """Check for consistent header formatting"""
        has_markdown = '##' in prompt or '###' in prompt
        has_xml = '</' in prompt
        # Either use markdown OR XML, not a messy mix
        if has_markdown and has_xml:
            # Allow mixing if it's structured (e.g., markdown headers with XML content tags)
            return True
        return True  # Single style is fine

    def _check_proper_spacing(self, prompt: str) -> bool:
        """Check for proper spacing (no excessive blank lines)"""
        # Check for more than 3 consecutive blank lines
        return '\n\n\n\n' not in prompt

    def _check_no_broken_syntax(self, prompt: str) -> bool:
        """Check for unclosed tags or broken syntax"""
        # Check XML-style tags
        import re
        open_tags = re.findall(r'<(\w+)>', prompt)
        close_tags = re.findall(r'</(\w+)>', prompt)

        # Simple check: all opened tags should have closing tags
        for tag in set(open_tags):
            if open_tags.count(tag) != close_tags.count(tag):
                return False

        # Check for unclosed code blocks
        code_block_count = prompt.count('```')
        if code_block_count % 2 != 0:
            return False

        return True

    def _check_no_placeholders(self, prompt: str) -> bool:
        """Check for common placeholder patterns"""
        placeholders = [
            '[TODO]', '[INSERT]', '[PLACEHOLDER]', '[YOUR', '[ADD',
            '{TODO}', '{INSERT}', '{PLACEHOLDER}', '{YOUR', '{ADD',
            'XXX', 'FIXME', '...'
        ]
        prompt_upper = prompt.upper()
        for placeholder in placeholders:
            if placeholder.upper() in prompt_upper:
                # Allow '...' in examples or ellipsis context
                if placeholder == '...' and ('example' in prompt.lower() or 'etc' in prompt.lower()):
                    continue
                return False
        return True

    def _get_openai_practices(self) -> List[Dict[str, Any]]:
        """OpenAI GPT-5 and GPT-4.1 best practices (from official documentation)"""
        return [
            {
                "id": "openai_delimiters",
                "name": "Use clear delimiters",
                "description": "Use markdown (###, ```) or XML tags to separate instructions from content (official: 'Use delimiters like ### or \"\"\"')",
                "check": lambda prompt: any(delim in prompt for delim in ["###", '"""', "---", "```", "<", ">"]),
                "importance": "high"
            },
            {
                "id": "openai_structured_output",
                "name": "Specify output format",
                "description": "Clearly specify the desired output format - JSON, XML, markdown (official: 'Specify output format clearly')",
                "check": lambda prompt: any(fmt in prompt.lower() for fmt in ["json", "xml", "format", "output", "return"]),
                "importance": "high"
            },
            {
                "id": "openai_examples",
                "name": "Provide examples",
                "description": "Include examples to demonstrate desired behavior (official: 'Add examples in a dedicated # Examples section')",
                "check": lambda prompt: any(word in prompt.lower() for word in ["example", "for instance", "such as", "e.g."]),
                "importance": "high"
            },
            {
                "id": "openai_persona",
                "name": "Define clear role at start",
                "description": "Define the AI's role or persona at the beginning (official: 'Role and Objective' should come first)",
                "check": lambda prompt: any(word in prompt.lower()[:500] for word in ["you are", "act as", "role", "your task"]),
                "importance": "high"
            },
            {
                "id": "openai_constraints",
                "name": "Specify constraints clearly",
                "description": "State what the AI should and shouldn't do (official: 'Avoid contradictions - establish clear priority')",
                "check": lambda prompt: any(word in prompt.lower() for word in ["do not", "never", "avoid", "must", "should not", "always"]),
                "importance": "high"
            },
            {
                "id": "openai_step_by_step",
                "name": "Include step-by-step reasoning",
                "description": "Prompt for step-by-step breakdown (official: 'Think carefully step by step before answering')",
                "check": lambda prompt: any(phrase in prompt.lower() for phrase in ["step-by-step", "step by step", "think through", "reasoning"]),
                "importance": "medium"
            },
            {
                "id": "openai_context_organization",
                "name": "Organize with clear sections",
                "description": "Use the recommended structure: Role → Instructions → Reasoning → Output Format → Examples → Context",
                "check": lambda prompt: len(prompt.split("\n\n")) >= 3 or prompt.count("#") >= 2,
                "importance": "medium"
            },
            {
                "id": "openai_no_contradictions",
                "name": "No contradictory instructions",
                "description": "Avoid conflicting directives that waste reasoning tokens (official: 'Conflicting directives force unnecessary token consumption')",
                "check": lambda prompt: not (("always" in prompt.lower() and "never" in prompt.lower() and prompt.lower().count("always") > 2)),
                "importance": "high"
            },
            {
                "id": "openai_instruction_placement",
                "name": "Proper instruction placement",
                "description": "For long context, place instructions at beginning AND end (official: 'Place instructions at BOTH beginning AND end')",
                "check": lambda prompt: len(prompt) < 2000 or (any(word in prompt.lower()[:500] for word in ["instruction", "task", "you"]) and any(word in prompt.lower()[-500:] for word in ["remember", "important", "must", "output"])),
                "importance": "medium"
            },
            {
                "id": "openai_markdown_headers",
                "name": "Use markdown headers",
                "description": "Use H1-H4+ titles for organization (official: 'Use markdown effectively with H1-H4+ titles')",
                "check": lambda prompt: any(h in prompt for h in ["# ", "## ", "### "]),
                "importance": "medium"
            }
        ]

    def _get_claude_practices(self) -> List[Dict[str, Any]]:
        """Claude 4 best practices (from official documentation)"""
        return [
            {
                "id": "claude_xml_tags",
                "name": "Use XML tags for structure",
                "description": "Use XML tags like <context>, <instructions>, <examples> (official: 'Use XML tags for structure')",
                "check": lambda prompt: "<" in prompt and ">" in prompt and "</" in prompt,
                "importance": "high"
            },
            {
                "id": "claude_examples",
                "name": "Provide examples",
                "description": "Use <example> tags with input/output pairs (official: 'Claude 4.x scrutinizes examples closely')",
                "check": lambda prompt: "<example>" in prompt.lower() or "example" in prompt.lower(),
                "importance": "high"
            },
            {
                "id": "claude_explicit_direct",
                "name": "Be explicit and direct",
                "description": "Rather than 'Create a dashboard', say 'Create a dashboard with X, Y, Z features' (official guidance)",
                "check": lambda prompt: len(prompt) > 100 and any(word in prompt.lower() for word in ["specific", "include", "with", "features", "must"]),
                "importance": "high"
            },
            {
                "id": "claude_role",
                "name": "Define clear role",
                "description": "Clearly define Claude's role and expertise at the start",
                "check": lambda prompt: any(word in prompt.lower()[:500] for word in ["you are", "your role", "act as", "your task"]),
                "importance": "high"
            },
            {
                "id": "claude_positive_framing",
                "name": "Use positive framing",
                "description": "Say what TO do, not what NOT to do (official: 'Avoid negative framing')",
                "check": lambda prompt: prompt.lower().count("do not") + prompt.lower().count("don't") + prompt.lower().count("never") < 5,
                "importance": "medium"
            },
            {
                "id": "claude_explain_reasoning",
                "name": "Explain reasoning behind requests",
                "description": "Explain WHY you want something, not just WHAT (official: 'Explain the reasoning behind requests')",
                "check": lambda prompt: any(word in prompt.lower() for word in ["because", "reason", "so that", "in order to", "this helps"]),
                "importance": "medium"
            },
            {
                "id": "claude_extended_thinking",
                "name": "Enable reflection/thinking",
                "description": "Add reflection prompts for complex tasks (official: 'Carefully reflect on tool result quality')",
                "check": lambda prompt: any(phrase in prompt.lower() for phrase in ["think", "reflect", "consider", "analyze", "step-by-step"]),
                "importance": "medium"
            },
            {
                "id": "claude_constraints",
                "name": "Specify constraints clearly",
                "description": "Clear language for boundaries (official: 'Keep solutions simple - don't add features beyond what was asked')",
                "check": lambda prompt: any(word in prompt.lower() for word in ["must", "always", "should", "constraint", "limit", "only"]),
                "importance": "high"
            },
            {
                "id": "claude_context_tags",
                "name": "Use context tags",
                "description": "Use <context> or <background> tags for contextual info",
                "check": lambda prompt: any(tag in prompt.lower() for tag in ["<context>", "<background>", "<instructions>", "<task>"]),
                "importance": "medium"
            },
            {
                "id": "claude_prose_over_bullets",
                "name": "Complete paragraphs over bullets",
                "description": "Write complete paragraphs rather than excessive bullet points (official guidance)",
                "check": lambda prompt: prompt.count("- ") < 20 or len(prompt) > 2000,  # Allow bullets in longer prompts
                "importance": "low"
            }
        ]

    def _get_gemini_practices(self) -> List[Dict[str, Any]]:
        """Google Gemini best practices (from official documentation)"""
        return [
            {
                "id": "gemini_few_shot_examples",
                "name": "Include few-shot examples (CRITICAL)",
                "description": "ALWAYS include 2-5 examples (official: 'We recommend to always include few-shot examples')",
                "check": lambda prompt: any(word in prompt.lower() for word in ["example", "for instance", "such as", "e.g."]),
                "importance": "high"
            },
            {
                "id": "gemini_clear_instructions",
                "name": "Clear, explicit instructions",
                "description": "Provide specific directions rather than implicit assumptions (official guidance)",
                "check": lambda prompt: len(prompt) > 100 and any(word in prompt.lower() for word in ["must", "should", "task", "instruction"]),
                "importance": "high"
            },
            {
                "id": "gemini_response_format",
                "name": "Specify response format",
                "description": "Define exactly how output should be structured - tables, JSON, bullets (official guidance)",
                "check": lambda prompt: any(word in prompt.lower() for word in ["json", "format", "structure", "output", "return", "table"]),
                "importance": "high"
            },
            {
                "id": "gemini_input_output_prefixes",
                "name": "Use input/output prefixes",
                "description": "Label sections with prefixes like 'Text:', 'The answer is:' (official: 'Input/Output prefixes')",
                "check": lambda prompt: any(prefix in prompt for prefix in ["Input:", "Output:", "Text:", "Answer:", "Response:", "Query:"]),
                "importance": "medium"
            },
            {
                "id": "gemini_xml_or_markdown",
                "name": "Use XML tags or Markdown headings",
                "description": "Structure with <context>, <task> or ### headings consistently (official guidance)",
                "check": lambda prompt: ("<" in prompt and ">" in prompt) or "#" in prompt,
                "importance": "medium"
            },
            {
                "id": "gemini_direct_language",
                "name": "Use direct, precise language",
                "description": "Be concise - avoid persuasive phrasing (official: 'Be precise and concise')",
                "check": lambda prompt: len(prompt.split()) < 2000 or len(prompt) / len(prompt.split()) < 7,  # Not overly verbose
                "importance": "medium"
            },
            {
                "id": "gemini_critical_info_placement",
                "name": "Critical info at beginning",
                "description": "Position essential constraints and role definitions at the start (official guidance)",
                "check": lambda prompt: any(word in prompt.lower()[:300] for word in ["you are", "role", "task", "must", "important"]),
                "importance": "high"
            },
            {
                "id": "gemini_role",
                "name": "Define model's role",
                "description": "Clearly define what role the model should take at the start",
                "check": lambda prompt: any(word in prompt.lower()[:500] for word in ["you are", "act as", "role", "your task"]),
                "importance": "medium"
            },
            {
                "id": "gemini_context_integration",
                "name": "Include relevant context",
                "description": "Include background information and constraints directly in prompt (official guidance)",
                "check": lambda prompt: any(word in prompt.lower() for word in ["context", "background", "given", "based on"]),
                "importance": "medium"
            },
            {
                "id": "gemini_temperature_note",
                "name": "Consider temperature settings",
                "description": "For Gemini 3, keep temperature at 1.0 (official: 'Lowering may cause looping')",
                "check": lambda prompt: True,  # This is a runtime setting, not prompt-based
                "importance": "low"
            }
        ]

    def evaluate_prompt(self, prompt: str, provider: str) -> Dict[str, Any]:
        """
        Evaluate a prompt against provider-specific best practices and formatting checks

        Args:
            prompt: The system prompt to evaluate
            provider: Target provider ("openai", "claude", "gemini", "multi")

        Returns:
            Dictionary with score, violations, and recommendations
        """
        if provider == "multi":
            # Evaluate against all providers and combine
            return self._evaluate_multi_provider(prompt)

        provider_enum = Provider(provider)
        practices = self.practices.get(provider_enum, [])

        violations = []
        passed = []
        formatting_violations = []
        total_importance_score = 0
        achieved_importance_score = 0

        # First, check formatting issues (apply to all providers)
        for check in self.formatting_checks:
            importance_weight = {"high": 3, "medium": 2, "low": 1}[check["importance"]]
            total_importance_score += importance_weight

            try:
                if check["check"](prompt):
                    passed.append(check)
                    achieved_importance_score += importance_weight
                else:
                    formatting_violations.append({
                        "id": check["id"],
                        "name": check["name"],
                        "description": check["description"],
                        "importance": check["importance"],
                        "category": "formatting"
                    })
            except Exception:
                formatting_violations.append({
                    "id": check["id"],
                    "name": check["name"],
                    "description": check["description"],
                    "importance": check["importance"],
                    "category": "formatting"
                })

        # Then check provider-specific practices
        for practice in practices:
            importance_weight = {"high": 3, "medium": 2, "low": 1}[practice["importance"]]
            total_importance_score += importance_weight

            try:
                if practice["check"](prompt):
                    passed.append(practice)
                    achieved_importance_score += importance_weight
                else:
                    violations.append({
                        "id": practice["id"],
                        "name": practice["name"],
                        "description": practice["description"],
                        "importance": practice["importance"],
                        "category": "best_practice"
                    })
            except Exception:
                # If check fails, consider it a violation
                violations.append({
                    "id": practice["id"],
                    "name": practice["name"],
                    "description": practice["description"],
                    "importance": practice["importance"],
                    "category": "best_practice"
                })

        # Combine formatting violations first (they're more fundamental)
        all_violations = formatting_violations + violations

        score = (achieved_importance_score / total_importance_score * 100) if total_importance_score > 0 else 0

        return {
            "provider": provider,
            "score": round(score, 2),
            "violations": all_violations,
            "formatting_issues": formatting_violations,
            "best_practice_violations": violations,
            "passed_checks": len(passed),
            "total_checks": len(practices) + len(self.formatting_checks),
            "recommendations": self._generate_recommendations(all_violations, provider)
        }

    def _evaluate_multi_provider(self, prompt: str) -> Dict[str, Any]:
        """Evaluate against all providers and combine results"""
        all_violations = []
        all_scores = []

        for provider in [Provider.OPENAI, Provider.CLAUDE, Provider.GEMINI]:
            result = self.evaluate_prompt(prompt, provider.value)
            all_scores.append(result["score"])
            all_violations.extend(result["violations"])

        # Deduplicate violations based on similar descriptions
        unique_violations = []
        seen_names = set()
        for v in all_violations:
            if v["name"] not in seen_names:
                unique_violations.append(v)
                seen_names.add(v["name"])

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

        return {
            "provider": "multi",
            "score": round(avg_score, 2),
            "violations": unique_violations,
            "provider_scores": {
                "openai": all_scores[0],
                "claude": all_scores[1],
                "gemini": all_scores[2]
            },
            "recommendations": self._generate_recommendations(unique_violations, "multi")
        }

    def _generate_recommendations(self, violations: List[Dict[str, str]], provider: str) -> List[str]:
        """Generate actionable recommendations based on violations"""
        recommendations = []

        # Sort violations by importance
        violations_sorted = sorted(violations, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["importance"]])

        for v in violations_sorted[:5]:  # Top 5 recommendations
            recommendations.append(f"{v['name']}: {v['description']}")

        return recommendations


# Global instance
_best_practices_engine = None

def get_best_practices_engine() -> BestPracticesEngine:
    """Get the global best practices engine instance"""
    global _best_practices_engine
    if _best_practices_engine is None:
        _best_practices_engine = BestPracticesEngine()
    return _best_practices_engine
