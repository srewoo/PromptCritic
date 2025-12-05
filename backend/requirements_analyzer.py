"""
Requirements analyzer - evaluates prompt alignment with user requirements using LLM-based semantic analysis
"""
from typing import List, Dict, Any, Optional
import re
import json
import openai
import anthropic
import google.generativeai as genai


class RequirementsAnalyzer:
    """Analyzes how well a prompt aligns with user requirements using semantic analysis"""

    async def analyze_alignment(
        self,
        prompt: str,
        use_case: str,
        key_requirements: List[str],
        expected_behavior: str = None,
        constraints: Dict[str, Any] = None,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze how well a prompt aligns with requirements using LLM semantic analysis

        Args:
            prompt: The system prompt to evaluate
            use_case: The intended use case
            key_requirements: List of key requirements
            expected_behavior: Optional expected behavior description
            constraints: Optional constraints
            provider: LLM provider for semantic analysis
            api_key: API key for the provider
            model_name: Optional specific model to use

        Returns:
            Dictionary with alignment score, gaps, and suggestions
        """
        # Use LLM-based semantic analysis if API key is provided
        if api_key:
            return await self._analyze_with_llm(
                prompt=prompt,
                use_case=use_case,
                key_requirements=key_requirements,
                expected_behavior=expected_behavior,
                constraints=constraints,
                provider=provider,
                api_key=api_key,
                model_name=model_name
            )
        else:
            # Fallback to keyword-based analysis
            return self._analyze_keyword_based(
                prompt=prompt,
                use_case=use_case,
                key_requirements=key_requirements,
                expected_behavior=expected_behavior,
                constraints=constraints
            )

    async def _analyze_with_llm(
        self,
        prompt: str,
        use_case: str,
        key_requirements: List[str],
        expected_behavior: Optional[str],
        constraints: Optional[Dict[str, Any]],
        provider: str,
        api_key: str,
        model_name: Optional[str]
    ) -> Dict[str, Any]:
        """Perform semantic analysis using LLM"""

        # Build the analysis prompt
        analysis_prompt = self._build_semantic_analysis_prompt(
            prompt=prompt,
            use_case=use_case,
            key_requirements=key_requirements,
            expected_behavior=expected_behavior,
            constraints=constraints
        )

        # Call appropriate LLM provider
        try:
            if provider == "openai":
                result = await self._analyze_with_openai(analysis_prompt, api_key, model_name)
            elif provider == "claude":
                result = await self._analyze_with_claude(analysis_prompt, api_key, model_name)
            elif provider == "gemini":
                result = await self._analyze_with_gemini(analysis_prompt, api_key, model_name)
            else:
                # Fallback to keyword analysis
                return self._analyze_keyword_based(
                    prompt, use_case, key_requirements, expected_behavior, constraints
                )

            # Parse and validate the result
            return self._parse_llm_analysis(result, key_requirements)

        except Exception as e:
            print(f"LLM analysis failed: {e}. Falling back to keyword analysis.")
            return self._analyze_keyword_based(
                prompt, use_case, key_requirements, expected_behavior, constraints
            )

    def _build_semantic_analysis_prompt(
        self,
        prompt: str,
        use_case: str,
        key_requirements: List[str],
        expected_behavior: Optional[str],
        constraints: Optional[Dict[str, Any]]
    ) -> str:
        """Build the prompt for LLM semantic analysis"""

        requirements_list = "\n".join([f"{i+1}. {req}" for i, req in enumerate(key_requirements)])

        constraints_text = ""
        if constraints:
            constraints_text = f"""
## CONSTRAINTS TO CHECK
{json.dumps(constraints, indent=2)}
"""

        behavior_text = ""
        if expected_behavior:
            behavior_text = f"""
## EXPECTED BEHAVIOR
{expected_behavior}
"""

        analysis_prompt = f"""You are an expert prompt analyst. Your task is to perform a deep semantic analysis of how well a system prompt aligns with its intended requirements.

## SYSTEM PROMPT TO ANALYZE
<system_prompt>
{prompt}
</system_prompt>

## INTENDED USE CASE
{use_case}

## KEY REQUIREMENTS
{requirements_list}
{behavior_text}
{constraints_text}

## ANALYSIS INSTRUCTIONS

Perform a thorough semantic analysis considering:

1. **Use Case Alignment**: Does the prompt's overall purpose and approach match the intended use case? Consider:
   - Is the role/persona appropriate?
   - Are the instructions aligned with the use case goals?
   - Does the tone match the expected context?

2. **Requirements Coverage**: For EACH requirement, determine:
   - Is it explicitly addressed? (directly mentioned)
   - Is it implicitly covered? (addressed through related instructions)
   - Is it partially covered? (some aspects addressed but gaps exist)
   - Is it missing entirely?

3. **Semantic Understanding**: Look beyond keywords to understand:
   - Does the prompt semantically capture the requirement's intent?
   - Are there paraphrased instructions that fulfill the requirement?
   - Could the instructions lead to the desired behavior even without exact wording?

4. **Gap Analysis**: Identify what's missing:
   - Which requirements have no coverage (explicit or implicit)?
   - What aspects of covered requirements are incomplete?
   - What could be added to improve alignment?

5. **Suggestions**: Provide specific, actionable improvements:
   - How to address each gap
   - How to strengthen weak coverage
   - Priority of each suggestion

## OUTPUT FORMAT

Return your analysis as a valid JSON object with this EXACT structure:

{{
  "alignment_score": <number 0-100>,
  "use_case_alignment": {{
    "score": <number 0-100>,
    "analysis": "<detailed explanation of use case alignment>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"]
  }},
  "requirements_coverage": [
    {{
      "requirement": "<requirement text>",
      "covered": <true/false>,
      "coverage_type": "<explicit/implicit/partial/missing>",
      "confidence": "<high/medium/low>",
      "confidence_score": <number 0-100>,
      "evidence": "<quote or description of how it's covered>",
      "gaps": "<what's missing if partial/not covered>"
    }}
  ],
  "coverage_percentage": <number 0-100>,
  "gaps": ["<gap 1>", "<gap 2>"],
  "behavior_alignment": {{
    "score": <number 0-100>,
    "analysis": "<how well prompt supports expected behavior>"
  }},
  "constraint_violations": ["<violation 1>", "<violation 2>"],
  "suggestions": [
    {{
      "type": "<gap/improvement/constraint>",
      "priority": "<high/medium/low>",
      "requirement_index": <number or null>,
      "suggestion": "<specific actionable suggestion>",
      "example": "<example of how to implement>"
    }}
  ],
  "overall_analysis": "<2-3 sentence summary of the analysis>",
  "analysis_method": "semantic"
}}

IMPORTANT:
- Be thorough but fair in your assessment
- Look for semantic equivalence, not just keyword matches
- Consider implicit coverage through related instructions
- Provide specific evidence for your assessments
- Make suggestions actionable with examples

Return ONLY the JSON object, no other text."""

        return analysis_prompt

    async def _analyze_with_openai(
        self,
        analysis_prompt: str,
        api_key: str,
        model_name: Optional[str]
    ) -> str:
        """Perform semantic analysis using OpenAI"""
        client = openai.AsyncOpenAI(api_key=api_key)

        response = await client.chat.completions.create(
            model=model_name or "gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert prompt analyst specializing in semantic analysis of system prompts."
                },
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=4000,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content

    async def _analyze_with_claude(
        self,
        analysis_prompt: str,
        api_key: str,
        model_name: Optional[str]
    ) -> str:
        """Perform semantic analysis using Claude"""
        client = anthropic.AsyncAnthropic(api_key=api_key)

        response = await client.messages.create(
            model=model_name or "claude-3-5-sonnet-20241022",
            max_tokens=4000,
            messages=[
                {"role": "user", "content": analysis_prompt}
            ]
        )

        return response.content[0].text

    async def _analyze_with_gemini(
        self,
        analysis_prompt: str,
        api_key: str,
        model_name: Optional[str]
    ) -> str:
        """Perform semantic analysis using Gemini"""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or 'gemini-2.5-pro')

        response = await model.generate_content_async(analysis_prompt)

        return response.text

    def _parse_llm_analysis(
        self,
        llm_response: str,
        key_requirements: List[str]
    ) -> Dict[str, Any]:
        """Parse and validate the LLM analysis response"""
        try:
            # Clean up response if needed
            response_text = llm_response.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            # Parse JSON
            result = json.loads(response_text.strip())

            # Validate and ensure all required fields exist
            validated_result = {
                "alignment_score": result.get("alignment_score", 0),
                "use_case_alignment": result.get("use_case_alignment", {}).get("score", 0)
                    if isinstance(result.get("use_case_alignment"), dict)
                    else result.get("use_case_alignment", 0),
                "use_case_analysis": result.get("use_case_alignment", {}),
                "requirements_coverage": result.get("requirements_coverage", []),
                "coverage_percentage": result.get("coverage_percentage", 0),
                "gaps": result.get("gaps", []),
                "behavior_alignment": result.get("behavior_alignment", {}).get("score", 100)
                    if isinstance(result.get("behavior_alignment"), dict)
                    else result.get("behavior_alignment", 100),
                "behavior_analysis": result.get("behavior_alignment", {}),
                "constraint_violations": result.get("constraint_violations", []),
                "suggestions": result.get("suggestions", []),
                "overall_analysis": result.get("overall_analysis", ""),
                "analysis_method": "semantic"
            }

            # Ensure requirements_coverage has entries for all requirements
            if len(validated_result["requirements_coverage"]) < len(key_requirements):
                # Fill in missing requirements
                covered_reqs = {rc.get("requirement", ""): rc for rc in validated_result["requirements_coverage"]}
                for req in key_requirements:
                    if req not in covered_reqs:
                        validated_result["requirements_coverage"].append({
                            "requirement": req,
                            "covered": False,
                            "coverage_type": "missing",
                            "confidence": "low",
                            "confidence_score": 0,
                            "evidence": "",
                            "gaps": req
                        })

            return validated_result

        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {e}")
            # Return a default structure
            return {
                "alignment_score": 0,
                "use_case_alignment": 0,
                "requirements_coverage": [
                    {"requirement": req, "covered": False, "confidence": "low"}
                    for req in key_requirements
                ],
                "coverage_percentage": 0,
                "gaps": key_requirements,
                "behavior_alignment": 0,
                "constraint_violations": [],
                "suggestions": [{
                    "type": "error",
                    "priority": "high",
                    "suggestion": "Analysis failed - please retry"
                }],
                "analysis_method": "failed"
            }

    def _analyze_keyword_based(
        self,
        prompt: str,
        use_case: str,
        key_requirements: List[str],
        expected_behavior: Optional[str],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fallback keyword-based analysis (original implementation)"""
        prompt_lower = prompt.lower()

        # Analyze use case alignment
        use_case_score = self._check_use_case_alignment(prompt, use_case)

        # Analyze each requirement
        requirements_coverage = []
        covered_count = 0

        for req in key_requirements:
            is_covered = self._check_requirement_coverage(prompt_lower, req)
            requirements_coverage.append({
                "requirement": req,
                "covered": is_covered,
                "coverage_type": "keyword_match" if is_covered else "missing",
                "confidence": "medium" if is_covered else "low",
                "confidence_score": 70 if is_covered else 20,
                "evidence": "Keyword match found" if is_covered else "",
                "gaps": "" if is_covered else req
            })
            if is_covered:
                covered_count += 1

        # Calculate coverage percentage
        coverage_percentage = (covered_count / len(key_requirements) * 100) if key_requirements else 100

        # Identify gaps
        gaps = [rc["requirement"] for rc in requirements_coverage if not rc["covered"]]

        # Check expected behavior
        behavior_score = 100
        if expected_behavior:
            behavior_score = self._check_behavior_alignment(prompt, expected_behavior)

        # Check constraints
        constraint_violations = []
        if constraints:
            constraint_violations = self._check_constraints(prompt, constraints)

        # Calculate overall alignment score
        alignment_score = self._calculate_alignment_score(
            use_case_score,
            coverage_percentage,
            behavior_score,
            len(constraint_violations)
        )

        return {
            "alignment_score": round(alignment_score, 2),
            "use_case_alignment": round(use_case_score, 2),
            "use_case_analysis": {
                "score": round(use_case_score, 2),
                "analysis": "Keyword-based analysis",
                "strengths": [],
                "weaknesses": []
            },
            "requirements_coverage": requirements_coverage,
            "coverage_percentage": round(coverage_percentage, 2),
            "gaps": gaps,
            "behavior_alignment": round(behavior_score, 2),
            "behavior_analysis": {
                "score": round(behavior_score, 2),
                "analysis": "Keyword-based analysis"
            },
            "constraint_violations": constraint_violations,
            "suggestions": self._generate_suggestions(gaps, constraint_violations, use_case_score),
            "overall_analysis": f"Keyword-based analysis found {covered_count}/{len(key_requirements)} requirements covered.",
            "analysis_method": "keyword"
        }

    def _check_use_case_alignment(self, prompt: str, use_case: str) -> float:
        """Check if the prompt aligns with the use case (keyword-based)"""
        use_case_terms = self._extract_key_terms(use_case)
        prompt_lower = prompt.lower()

        matches = sum(1 for term in use_case_terms if term in prompt_lower)
        score = (matches / len(use_case_terms) * 100) if use_case_terms else 100

        return min(score, 100)

    def _check_requirement_coverage(self, prompt_lower: str, requirement: str) -> bool:
        """Check if a specific requirement is covered (keyword-based)"""
        req_terms = self._extract_key_terms(requirement)

        # Check if at least 50% of key terms are present
        matches = sum(1 for term in req_terms if term in prompt_lower)
        coverage = (matches / len(req_terms)) if req_terms else 0

        return coverage >= 0.5

    def _check_behavior_alignment(self, prompt: str, expected_behavior: str) -> float:
        """Check if the prompt aligns with expected behavior (keyword-based)"""
        behavior_terms = self._extract_key_terms(expected_behavior)
        prompt_lower = prompt.lower()

        matches = sum(1 for term in behavior_terms if term in prompt_lower)
        score = (matches / len(behavior_terms) * 100) if behavior_terms else 100

        return min(score, 100)

    def _check_constraints(self, prompt: str, constraints: Dict[str, Any]) -> List[str]:
        """Check if the prompt violates any constraints"""
        violations = []
        prompt_lower = prompt.lower()

        # Check token limit constraint
        if "max_tokens" in constraints:
            word_count = len(prompt.split())
            estimated_tokens = int(word_count / 0.75)
            if estimated_tokens > constraints["max_tokens"]:
                violations.append(
                    f"Prompt may exceed token limit ({estimated_tokens} estimated vs {constraints['max_tokens']} max)"
                )

        # Check forbidden terms
        if "forbidden_terms" in constraints:
            for term in constraints["forbidden_terms"]:
                if term.lower() in prompt_lower:
                    violations.append(f"Contains forbidden term: '{term}'")

        # Check required terms
        if "required_terms" in constraints:
            for term in constraints["required_terms"]:
                if term.lower() not in prompt_lower:
                    violations.append(f"Missing required term: '{term}'")

        return violations

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        words = re.findall(r'\b\w+\b', text.lower())
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]

        return key_terms

    def _calculate_alignment_score(
        self,
        use_case_score: float,
        coverage_percentage: float,
        behavior_score: float,
        constraint_violations_count: int
    ) -> float:
        """Calculate overall alignment score"""
        base_score = (
            use_case_score * 0.3 +
            coverage_percentage * 0.5 +
            behavior_score * 0.2
        )

        penalty = constraint_violations_count * 10
        final_score = max(0, base_score - penalty)

        return final_score

    def _generate_suggestions(
        self,
        gaps: List[str],
        constraint_violations: List[str],
        use_case_score: float
    ) -> List[Dict[str, str]]:
        """Generate actionable suggestions"""
        suggestions = []

        for i, gap in enumerate(gaps[:5]):
            suggestions.append({
                "type": "gap",
                "priority": "high",
                "requirement_index": i,
                "suggestion": f"Add explicit instructions addressing: {gap}",
                "example": f"Consider adding a section that specifically handles '{gap}'"
            })

        for violation in constraint_violations:
            suggestions.append({
                "type": "constraint",
                "priority": "high",
                "requirement_index": None,
                "suggestion": f"Fix constraint violation: {violation}",
                "example": ""
            })

        if use_case_score < 70:
            suggestions.append({
                "type": "improvement",
                "priority": "medium",
                "requirement_index": None,
                "suggestion": "Strengthen alignment with the stated use case by incorporating more relevant terminology and context",
                "example": "Add a clear role definition and use case-specific instructions at the beginning"
            })

        return suggestions


# Global instance
_requirements_analyzer = None

def get_requirements_analyzer() -> RequirementsAnalyzer:
    """Get the global requirements analyzer instance"""
    global _requirements_analyzer
    if _requirements_analyzer is None:
        _requirements_analyzer = RequirementsAnalyzer()
    return _requirements_analyzer
