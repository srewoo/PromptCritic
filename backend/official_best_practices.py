"""
Official Best Practices from Provider Documentation

Sources:
- OpenAI GPT-5: https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide
- OpenAI GPT-4.1: https://cookbook.openai.com/examples/gpt4-1_prompting_guide
- Google Gemini: https://ai.google.dev/gemini-api/docs/prompting-strategies
- Anthropic Claude 4: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices

Last Updated: December 2025
"""

# =============================================================================
# OPENAI GPT-5 & GPT-4.1 BEST PRACTICES
# =============================================================================

OPENAI_BEST_PRACTICES = """
## OpenAI GPT-5 & GPT-4.1 Official Best Practices

### Structure & Formatting
1. **Use Markdown effectively**: Use H1-H4+ titles, inline backticks for code/function names, numbered/bulleted lists
2. **XML tags work well**: Strong performance for structure; enables metadata and nesting
3. **Avoid JSON for document collections**: Performs poorly in testing; use XML or ID-based formats instead
4. **Use clear delimiters**: Separate instructions from content with ###, ```, ---, or XML tags
5. **Refresh markdown instructions**: In long conversations, refresh formatting instructions every 3-5 messages

### Prompt Structure Template (Recommended Order)
```
# Role and Objective
# Instructions
## Sub-categories for detailed instructions
# Reasoning Steps
# Output Format
# Examples
# Context
# Final instructions and step-by-step prompt
```

### Instruction Following
1. **Avoid contradictions**: Conflicting directives waste reasoning tokens and cause confusion
2. **Establish clear hierarchy**: When rules might conflict, specify which takes precedence
3. **Be explicit**: GPT-4.1 follows instructions literally - state exactly what you want
4. **Instructions placement**: For long context, place instructions at BOTH beginning AND end
5. **Single-sentence corrections**: The model is highly steerable - brief corrections work well

### Chain of Thought & Reasoning
1. **Prompt step-by-step breakdown**: "Think carefully step by step before answering"
2. **Structure reasoning phases**: Query Analysis → Context Analysis → Synthesis
3. **Use reasoning effort levels**: low/medium/high for different task complexity
4. **Separate distinct tasks**: Across multiple turns for optimal performance

### Agentic Workflows
1. **Include three key reminders**:
   - Persistence: "Keep going until the user's query is completely resolved"
   - Tool-calling: "Use tools fully rather than guessing"
   - Planning: "Reflect step-by-step before tool calls"
2. **Tool preambles**: Begin by rephrasing goals, outline plans, narrate execution
3. **Use API tools field**: Not manual prompt injection (2% performance gain)
4. **Set clear stop conditions**: Define when to stop and what's unsafe

### Output Control
1. **Specify output format clearly**: JSON, XML, markdown, etc.
2. **Use verbosity parameter**: Control final answer length distinct from reasoning
3. **Apply semantic markdown only**: Backticks for code, parentheses for math
4. **Break long outputs**: Model may resist very long repetitive outputs

### Diff/Code Generation (V4A Format)
1. **No line numbers**: Context identifies code location
2. **Use @@ operators**: For class/function context when needed
3. **3 lines context**: Above and below changes by default
4. **Format**: `*** [ACTION] File: [path]` with `-` for old, `+` for new

### Common Pitfalls to Avoid
- Don't use ALL-CAPS, bribes, or tips unnecessarily
- Mandatory tool-calling can cause hallucinated inputs - add "if lacking info, ask"
- Sample phrases may be repeated verbatim - instruct variation
- Over-reliance on JSON for complex document structures
"""

# =============================================================================
# ANTHROPIC CLAUDE 4 BEST PRACTICES
# =============================================================================

CLAUDE_BEST_PRACTICES = """
## Anthropic Claude 4 Official Best Practices

### Core Principles
1. **Be explicit and direct**: Rather than "Create a dashboard", say "Create a dashboard. Include as many relevant features as possible."
2. **Explain reasoning behind requests**: Instead of "NEVER use ellipses", say "Your response will be read aloud by TTS, so never use ellipses since TTS won't know how to pronounce them."
3. **Detail sensitivity**: Claude 4.x scrutinizes examples closely - ensure they align with desired behaviors

### Formatting & Structure
1. **Use XML tags for structure**: `<context>`, `<instructions>`, `<examples>`, `<output_format>`
2. **Avoid negative framing**: Don't say "Do not use markdown" - say "Compose in flowing prose paragraphs"
3. **XML indicators for sections**: Use tags like `<smoothly_flowing_prose>` to guide structure
4. **Reserve markdown for**: Inline code, code blocks, and simple headings only
5. **Write complete paragraphs**: Avoid excessive bullet points unless presenting discrete items

### Tool Usage Patterns
1. **Proactive implementation**: Claude 4.5 defaults to suggesting - encourage action with "Implement changes rather than only suggesting them"
2. **Conservative approach** (if preferred): "Do not jump into implementation unless clearly instructed"
3. **Parallel execution**: "Make all independent tool calls in parallel... Maximize parallel tool calls"
4. **Exploration first**: "ALWAYS read and understand relevant files before proposing code edits"

### Extended Thinking
1. **Enable reflection after tool use**: "Carefully reflect on tool result quality and determine optimal next steps"
2. **Use thinking for planning**: "Use your thinking to plan and iterate before proceeding"

### Code & Development
1. **Reduce overengineering**: "Only make changes directly requested or clearly necessary. Keep solutions simple."
2. **General solutions**: "Write a high-quality, general-purpose solution that works for all valid inputs, not just test cases"
3. **Minimize temporary files**: "If you create temporary files, clean them up at the end"
4. **No speculation**: "Do not speculate about code you have not inspected"

### State Management (Long Tasks)
1. **Use structured formats**: JSON for discrete data like test results
2. **Store progress notes**: In unstructured text files
3. **Leverage git**: For state tracking across sessions
4. **Create setup scripts**: To avoid repeated initialization

### Communication Style
1. **Concise and direct**: Claude 4.5 uses more concise language than predecessors
2. **Request summaries**: "After completing a task with tool use, provide a quick summary"
3. **Context compaction awareness**: Inform Claude when context will auto-compact

### Frontend Design (Avoid "AI Slop")
1. **Distinctive typography**: Avoid Arial, Inter - try unique fonts
2. **Cohesive color themes**: Use CSS variables
3. **Purposeful animations**: Staggered reveals at load for high impact
4. **Atmospheric backgrounds**: Rather than flat colors
5. **Context-specific choices**: Make creative decisions based on the specific use case

### Research & Complex Tasks
1. **Structured research**: Develop competing hypotheses
2. **Track confidence levels**: In progress notes
3. **Self-critique regularly**: Update hypothesis trees for transparency

### Migration Tips (from earlier Claude versions)
1. **Frame with quality modifiers**: "Create a high-quality, professional..."
2. **Request interactive elements explicitly**: Animations, interactions
3. **Describe exactly what you want**: Be specific about output appearance
4. **Natural prompting**: Avoid aggressive language - Claude 4.5 is more responsive to system prompts
"""

# =============================================================================
# GOOGLE GEMINI BEST PRACTICES
# =============================================================================

GEMINI_BEST_PRACTICES = """
## Google Gemini Official Best Practices

### Core Principles
1. **Clear, explicit instructions**: Provide specific directions rather than relying on implicit assumptions
2. **Context integration**: Include relevant background information and constraints directly in prompts
3. **Response format specification**: Define exactly how you want output structured (tables, JSON, bullets, templates)

### Few-Shot Examples (Strongly Recommended)
1. **Always include examples**: "We recommend to always include few-shot examples in your prompts"
2. **Use 2-5 varied examples**: Consistently formatted, showing different scenarios
3. **Examples beat instructions alone**: Showing patterns works better than describing them

### Formatting Techniques
1. **Input/Output prefixes**: Label sections with prefixes like "Text:" for inputs and "The answer is:" for outputs
2. **XML tags or Markdown headings**: Use `<context>`, `<task>` or ### consistently
3. **Direct language**: Be precise and concise - avoid persuasive phrasing
4. **Critical info placement**: Position essential constraints and role definitions at the beginning

### Prompt Chaining
1. **Break complex tasks**: Into sequential prompts where each output feeds the next input
2. **Improves accuracy**: For multi-stage reasoning tasks
3. **Completion strategy**: Begin structuring expected output to guide the model

### Gemini 3-Specific
1. **Keep temperature at 1.0**: "When using Gemini 3, we strongly recommend keeping temperature at default 1.0"
2. **Lower temperature issues**: May cause looping or degraded performance on complex tasks

### Parameter Experimentation
1. **Max Output Tokens**: Control response length (100 tokens ≈ 60-80 words)
2. **TopK/TopP**: Adjust token selection probability for creativity vs consistency
3. **Stop Sequences**: Define where generation should terminate

### Iteration Strategies (When Results Disappoint)
1. **Rephrase**: Use different wording
2. **Reframe**: Try analogous tasks achieving similar goals
3. **Reorder**: Rearrange prompt components (examples, context, input)

### Agentic Workflows
1. **Logical decomposition**: Require explicit planning
2. **Risk assessment**: Distinguish exploratory vs state-changing actions
3. **Ambiguity handling**: Define when to assume vs request clarification
4. **Persistence levels**: Configure error recovery behavior

### Common Pitfalls
1. **Don't rely on models for pure factual info**: Without verification
2. **Complex math**: Always validate independently
3. **Critical outputs**: Always validate independently
"""

# =============================================================================
# COMBINED PROVIDER-SPECIFIC GUIDANCE FOR PROMPT REWRITING
# =============================================================================

def get_provider_guidance(provider: str) -> str:
    """Get provider-specific guidance for prompt rewriting"""

    if provider == "openai":
        return """
### OpenAI (GPT-4o/GPT-4o-mini) Optimization

Based on official OpenAI documentation:

**Structure:**
- Use markdown headers (##, ###) to organize sections
- Apply clear delimiters (```, ---, ###) between instructions and content
- Follow this order: Role → Instructions → Reasoning Steps → Output Format → Examples → Context

**Formatting:**
- Use inline backticks for code/function names
- Use numbered lists for sequential steps
- Use bullet points for non-sequential items
- Refresh formatting instructions every 3-5 messages in long conversations

**Instructions:**
- Be explicit - GPT-4.1 follows instructions literally
- Avoid contradictions - establish clear priority when rules might conflict
- Place instructions at BOTH beginning AND end for long context
- Include step-by-step breakdown: "Think carefully step by step before answering"

**For Agentic Tasks:**
- Include persistence reminder: "Keep going until the user's query is completely resolved"
- Include tool-calling reminder: "Use tools fully rather than guessing"
- Include planning reminder: "Reflect step-by-step before tool calls"

**Output:**
- Specify output format clearly (JSON, XML, markdown)
- Use V4A format for code diffs (no line numbers, @@ operators, 3 lines context)
"""

    elif provider == "claude":
        return """
### Claude (Sonnet 3.7/3.5) Optimization

Based on official Anthropic documentation:

**Structure:**
- Use XML tags for clear structure: <context>, <instructions>, <examples>, <output_format>
- Use <example> tags with input/output pairs for few-shot learning
- Enable extended thinking with reflection prompts for complex tasks

**Formatting:**
- Avoid negative framing - say what TO do, not what NOT to do
- Reserve markdown for: inline code, code blocks, simple headings only
- Write complete paragraphs rather than excessive bullet points
- Use XML indicators like <smoothly_flowing_prose> to guide sections

**Instructions:**
- Be explicit and direct: "Create X with Y features" not just "Create X"
- Explain reasoning behind requests for better compliance
- Claude 4.x scrutinizes examples closely - ensure alignment with desired behavior

**For Agentic Tasks:**
- Encourage action: "Implement changes rather than only suggesting them"
- Require exploration first: "ALWAYS read and understand relevant files before proposing edits"
- Enable parallel execution: "Make all independent tool calls in parallel"
- Add reflection: "Carefully reflect on tool result quality before proceeding"

**Code Generation:**
- "Write a high-quality, general-purpose solution for all valid inputs"
- "Keep solutions simple - don't add features beyond what was asked"
- "Do not speculate about code you have not inspected"
"""

    elif provider == "gemini":
        return """
### Gemini (2.5 Pro/Flash) Optimization

Based on official Google documentation:

**Structure:**
- Use XML tags (<context>, <task>) or Markdown headings consistently
- Position critical info (constraints, role definitions) at the beginning
- Use Input/Output prefixes to label sections: "Text:", "The answer is:"

**Few-Shot Examples (CRITICAL):**
- "We recommend to always include few-shot examples in your prompts"
- Use 2-5 varied, consistently-formatted examples
- Showing patterns works better than describing them

**Formatting:**
- Be precise and concise - avoid persuasive phrasing
- Use direct language to state goals clearly
- Define exactly how you want output structured (tables, JSON, bullets)

**Temperature:**
- Keep at default 1.0 for Gemini 3 - lower may cause looping
- Adjust TopK/TopP for creativity vs consistency tradeoff

**Complex Tasks:**
- Use prompt chaining: break into sequential prompts
- Begin structuring expected output to guide completion
- Provide clear stop sequences

**For Agentic Tasks:**
- Require explicit logical decomposition and planning
- Define risk assessment for exploratory vs state-changing actions
- Configure ambiguity handling: when to assume vs request clarification
"""

    else:  # multi-provider
        return """
### Multi-Provider Optimization

For prompts that must work across OpenAI, Claude, and Gemini:

**Universal Structure:**
- Use clear section headers (## or ### work across all)
- Separate sections with blank lines
- Put most important instructions at the beginning AND end

**Universal Formatting:**
- Use numbered lists for sequential steps
- Use bullet points for non-sequential items
- Keep formatting simple - avoid provider-specific syntax
- Use plain text delimiters (---, blank lines) over complex markup

**Universal Instructions:**
- Be explicit and direct in all instructions
- Avoid contradictions and establish clear priorities
- Include examples - all providers benefit from few-shot learning
- Specify output format clearly

**Avoid:**
- Heavy XML (works great on Claude, less so on others)
- Complex nested markdown (inconsistent rendering)
- Provider-specific features (extended thinking, tool preambles)

**Test Critical Behaviors:**
- Verify the prompt works correctly on all target providers
- Focus on core functionality that must be consistent
"""


def get_full_best_practices_reference() -> str:
    """Get the complete best practices reference for LLM context"""
    return f"""
# Official Provider Best Practices Reference

This reference contains best practices extracted from official documentation.

{OPENAI_BEST_PRACTICES}

{CLAUDE_BEST_PRACTICES}

{GEMINI_BEST_PRACTICES}
"""
