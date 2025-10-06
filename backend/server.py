from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import openai
import anthropic
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
from fastapi.responses import StreamingResponse
import json


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# ============= Models =============

class Settings(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    llm_provider: str  # "openai", "claude", "gemini"
    api_key: str
    model_name: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SettingsCreate(BaseModel):
    llm_provider: str
    api_key: str
    model_name: Optional[str] = None


class CriterionScore(BaseModel):
    criterion: str
    score: int
    strength: str
    improvement: str
    rationale: str


class Evaluation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt_text: str
    llm_provider: str
    model_name: Optional[str] = None
    criteria_scores: List[CriterionScore]
    total_score: int
    refinement_suggestions: List[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvaluateRequest(BaseModel):
    prompt_text: str


class CompareRequest(BaseModel):
    evaluation_ids: List[str]


class RewriteRequest(BaseModel):
    prompt_text: str
    evaluation_id: Optional[str] = None
    focus_areas: Optional[List[str]] = None


class PlaygroundRequest(BaseModel):
    prompt_text: str
    test_input: str
    llm_provider: Optional[str] = None
    model_name: Optional[str] = None


# ============= Evaluation Prompt Template =============

EVALUATION_SYSTEM_PROMPT = """You are a **senior prompt engineer** participating in the **Prompt Evaluation Chain**, a quality system built to enhance prompt design through systematic reviews and iterative feedback. Your task is to **analyze and score a given prompt** following a detailed rubric.

You will evaluate the prompt using the 35-criteria rubric below. For each criterion, assign a score from 1 (Poor) to 5 (Excellent).

## 📊 Evaluation Criteria:

1. Clarity & Specificity
2. Context / Background Provided
3. Explicit Task Definition
4. Feasibility within Model Constraints
5. Avoiding Ambiguity or Contradictions
6. Model Fit / Scenario Appropriateness
7. Desired Output Format / Style
8. Use of Role or Persona
9. Step-by-Step Reasoning Encouraged
10. Structured / Numbered Instructions
11. Brevity vs. Detail Balance
12. Iteration / Refinement Potential
13. Examples or Demonstrations
14. Handling Uncertainty / Gaps
15. Hallucination Minimization
16. Knowledge Boundary Awareness
17. Audience Specification
18. Style Emulation or Imitation
19. Memory Anchoring (Multi-Turn Systems)
20. Meta-Cognition Triggers
21. Divergent vs. Convergent Thinking Management
22. Hypothetical Frame Switching
23. Safe Failure Mode
24. Progressive Complexity
25. Alignment with Evaluation Metrics
26. Calibration Requests
27. Output Validation Hooks
28. Time/Effort Estimation Request
29. Ethical Alignment or Bias Mitigation
30. Limitations Disclosure
31. Compression / Summarization Ability
32. Cross-Disciplinary Bridging
33. Emotional Resonance Calibration
34. Output Risk Categorization
35. Self-Repair Loops

**IMPORTANT**: You must return a valid JSON object with this EXACT structure:
{
  "criteria_scores": [
    {
      "criterion": "Clarity & Specificity",
      "score": 4,
      "strength": "Clear statement of purpose",
      "improvement": "Add more specific constraints",
      "rationale": "The prompt clearly defines what is needed but could benefit from additional details."
    },
    ... (repeat for all 35 criteria)
  ],
  "total_score": 140,
  "refinement_suggestions": [
    "Add specific output format requirements",
    "Include example inputs and outputs",
    ... (7-10 suggestions total)
  ]
}

Return ONLY the JSON object, no other text."""


REWRITE_SYSTEM_PROMPT = """You are an expert prompt engineer. Your task is to rewrite and improve a prompt based on evaluation feedback and focus areas.

Guidelines:
1. Maintain the original intent and core purpose
2. Address specific weaknesses identified in the evaluation
3. Incorporate best practices for prompt engineering
4. Make the prompt more clear, specific, and effective
5. Add appropriate context, examples, or structure as needed

Return a JSON object with this structure:
{
  "rewritten_prompt": "The improved version of the prompt...",
  "changes_made": ["List of specific improvements", "Another improvement"],
  "rationale": "Brief explanation of why these changes improve the prompt"
}

Return ONLY the JSON object, no other text."""


# ============= Cost Calculation Constants =============

# Pricing per 1M tokens (as of 2025)
TOKEN_COSTS = {
    "openai": {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    },
    "claude": {
        "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    },
    "gemini": {
        "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    }
}


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters)"""
    return len(text) // 4


def calculate_cost(prompt_text: str, response_text: str, provider: str, model: str) -> Dict[str, float]:
    """Calculate estimated API cost"""
    input_tokens = estimate_tokens(prompt_text) + 1000  # +1000 for system prompt
    output_tokens = estimate_tokens(response_text)
    
    # Get pricing, default to gpt-4o if not found
    pricing = TOKEN_COSTS.get(provider, {}).get(model, TOKEN_COSTS["openai"]["gpt-4o"])
    
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
        "currency": "USD"
    }


# ============= Helper Functions =============

def prepare_for_mongo(data: dict) -> dict:
    """Convert datetime objects to ISO strings for MongoDB"""
    if isinstance(data.get('created_at'), datetime):
        data['created_at'] = data['created_at'].isoformat()
    return data


def parse_from_mongo(item: dict) -> dict:
    """Convert ISO strings back to datetime objects"""
    if isinstance(item.get('created_at'), str):
        item['created_at'] = datetime.fromisoformat(item['created_at'])
    return item


async def get_llm_evaluation(prompt_text: str, provider: str, api_key: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    """Call LLM to evaluate the prompt"""
    
    # Map provider to default models
    default_models = {
        "openai": "gpt-4o",
        "claude": "claude-3-7-sonnet-20250219",
        "gemini": "gemini-2.0-flash-exp"
    }
    
    model = model_name or default_models.get(provider, "gpt-4o")
    
    user_prompt = f"Please evaluate the following prompt:\n\n```\n{prompt_text}\n```"
    
    try:
        if provider == "openai":
            client = openai.AsyncOpenAI(api_key=api_key, timeout=120.0)  # 2 minute timeout
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            response_text = response.choices[0].message.content
            
        elif provider == "claude":
            client = anthropic.AsyncAnthropic(api_key=api_key, timeout=120.0)  # 2 minute timeout
            response = await client.messages.create(
                model=model,
                max_tokens=4096,
                system=EVALUATION_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                timeout=120.0  # 2 minute timeout
            )
            response_text = response.content[0].text
            
        elif provider == "gemini":
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel(model)
            response = await model_obj.generate_content_async(
                f"{EVALUATION_SYSTEM_PROMPT}\n\n{user_prompt}"
            )
            response_text = response.text
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
        
        # Parse the JSON response
        # Clean the response if it has markdown code blocks
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        evaluation_data = json.loads(response_text)
        
        # Add cost calculation
        cost_info = calculate_cost(user_prompt, response_text, provider, model)
        evaluation_data["cost"] = cost_info
        
        return evaluation_data
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse LLM response: {response_text}")
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM evaluation response: {str(e)}")
    except Exception as e:
        logging.error(f"LLM API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")


def generate_pdf(evaluation: Evaluation) -> BytesIO:
    """Generate a PDF report for an evaluation"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=8
    )
    
    story = []
    
    # Title
    story.append(Paragraph("Prompt Evaluation Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Metadata
    story.append(Paragraph(f"<b>Evaluation ID:</b> {evaluation.id}", styles['Normal']))
    story.append(Paragraph(f"<b>Date:</b> {evaluation.created_at.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"<b>LLM Provider:</b> {evaluation.llm_provider.title()}", styles['Normal']))
    story.append(Paragraph(f"<b>Total Score:</b> {evaluation.total_score}/175", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Prompt Text
    story.append(Paragraph("Original Prompt", heading_style))
    prompt_text = evaluation.prompt_text[:500] + "..." if len(evaluation.prompt_text) > 500 else evaluation.prompt_text
    story.append(Paragraph(prompt_text.replace('\n', '<br/>'), styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Criteria Scores Table
    story.append(Paragraph("Evaluation Scores", heading_style))
    
    table_data = [['Criterion', 'Score', 'Strength', 'Improvement']]
    for score in evaluation.criteria_scores[:10]:  # Show first 10 for PDF brevity
        table_data.append([
            Paragraph(score.criterion, styles['Normal']),
            str(score.score),
            Paragraph(score.strength[:50] + "..." if len(score.strength) > 50 else score.strength, styles['Normal']),
            Paragraph(score.improvement[:50] + "..." if len(score.improvement) > 50 else score.improvement, styles['Normal'])
        ])
    
    table = Table(table_data, colWidths=[2*inch, 0.6*inch, 2*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.2*inch))
    
    # Refinement Suggestions
    story.append(Paragraph("Refinement Suggestions", heading_style))
    for i, suggestion in enumerate(evaluation.refinement_suggestions, 1):
        story.append(Paragraph(f"{i}. {suggestion}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


# ============= API Endpoints =============

@api_router.post("/settings", response_model=Settings)
async def save_settings(input: SettingsCreate):
    """Save or update LLM provider settings"""
    # Delete existing settings
    await db.settings.delete_many({})
    
    settings_dict = input.dict()
    settings_obj = Settings(**settings_dict)
    settings_data = prepare_for_mongo(settings_obj.dict())
    
    await db.settings.insert_one(settings_data)
    return settings_obj


@api_router.get("/settings", response_model=Optional[Settings])
async def get_settings():
    """Get current settings"""
    settings = await db.settings.find_one()
    if not settings:
        return None
    return Settings(**parse_from_mongo(settings))


@api_router.post("/evaluate", response_model=Evaluation)
async def evaluate_prompt(input: EvaluateRequest):
    """Evaluate a prompt using the configured LLM"""
    # Get settings
    settings_doc = await db.settings.find_one()
    if not settings_doc:
        raise HTTPException(status_code=400, detail="Please configure LLM settings first")
    
    settings = Settings(**parse_from_mongo(settings_doc))
    
    # Get evaluation from LLM
    try:
        evaluation_data = await get_llm_evaluation(
            input.prompt_text,
            settings.llm_provider,
            settings.api_key,
            settings.model_name
        )
        
        # Create evaluation object
        criteria_scores = [CriterionScore(**cs) for cs in evaluation_data['criteria_scores']]
        
        evaluation = Evaluation(
            prompt_text=input.prompt_text,
            llm_provider=settings.llm_provider,
            model_name=settings.model_name,
            criteria_scores=criteria_scores,
            total_score=evaluation_data['total_score'],
            refinement_suggestions=evaluation_data['refinement_suggestions']
        )
        
        # Save to database
        eval_data = prepare_for_mongo(evaluation.dict())
        await db.evaluations.insert_one(eval_data)
        
        return evaluation
        
    except Exception as e:
        logging.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@api_router.get("/evaluations", response_model=List[Evaluation])
async def get_evaluations(limit: int = 50, skip: int = 0):
    """Get all evaluations"""
    evaluations = await db.evaluations.find().sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    return [Evaluation(**parse_from_mongo(e)) for e in evaluations]


@api_router.get("/evaluations/{evaluation_id}", response_model=Evaluation)
async def get_evaluation(evaluation_id: str):
    """Get a specific evaluation"""
    evaluation = await db.evaluations.find_one({"id": evaluation_id})
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return Evaluation(**parse_from_mongo(evaluation))


@api_router.delete("/evaluations/{evaluation_id}")
async def delete_evaluation(evaluation_id: str):
    """Delete an evaluation"""
    result = await db.evaluations.delete_one({"id": evaluation_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return {"message": "Evaluation deleted successfully"}


@api_router.post("/compare")
async def compare_evaluations(input: CompareRequest):
    """Compare multiple evaluations"""
    evaluations = []
    for eval_id in input.evaluation_ids:
        evaluation = await db.evaluations.find_one({"id": eval_id})
        if evaluation:
            evaluations.append(Evaluation(**parse_from_mongo(evaluation)))
    
    if len(evaluations) < 2:
        raise HTTPException(status_code=400, detail="At least 2 evaluations required for comparison")
    
    # Build comparison data
    comparison = {
        "evaluations": [e.dict() for e in evaluations],
        "summary": {
            "avg_score": sum(e.total_score for e in evaluations) / len(evaluations),
            "max_score": max(e.total_score for e in evaluations),
            "min_score": min(e.total_score for e in evaluations),
        }
    }
    
    return comparison


@api_router.get("/export/json/{evaluation_id}")
async def export_json(evaluation_id: str):
    """Export evaluation as JSON"""
    evaluation = await db.evaluations.find_one({"id": evaluation_id})
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    eval_obj = Evaluation(**parse_from_mongo(evaluation))
    
    return StreamingResponse(
        iter([json.dumps(eval_obj.dict(), indent=2, default=str)]),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=evaluation_{evaluation_id}.json"}
    )


@api_router.get("/export/pdf/{evaluation_id}")
async def export_pdf(evaluation_id: str):
    """Export evaluation as PDF"""
    evaluation = await db.evaluations.find_one({"id": evaluation_id})
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    eval_obj = Evaluation(**parse_from_mongo(evaluation))
    pdf_buffer = generate_pdf(eval_obj)
    
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=evaluation_{evaluation_id}.pdf"}
    )


@api_router.post("/rewrite")
async def rewrite_prompt(input: RewriteRequest):
    """AI-powered prompt rewriting based on evaluation feedback"""
    # Get settings
    settings_doc = await db.settings.find_one()
    if not settings_doc:
        raise HTTPException(status_code=400, detail="Please configure LLM settings first")
    
    settings = Settings(**parse_from_mongo(settings_doc))
    
    # Get evaluation if ID provided
    evaluation_feedback = ""
    if input.evaluation_id:
        evaluation = await db.evaluations.find_one({"id": input.evaluation_id})
        if evaluation:
            eval_obj = Evaluation(**parse_from_mongo(evaluation))
            # Create feedback summary
            low_scores = [cs for cs in eval_obj.criteria_scores if cs.score <= 2]
            evaluation_feedback = f"\n\nEvaluation Feedback:\n"
            evaluation_feedback += f"Total Score: {eval_obj.total_score}/175\n\n"
            evaluation_feedback += "Low-scoring areas:\n"
            for cs in low_scores[:5]:
                evaluation_feedback += f"- {cs.criterion} (score: {cs.score}): {cs.improvement}\n"
            evaluation_feedback += f"\n\nKey Suggestions:\n"
            for i, suggestion in enumerate(eval_obj.refinement_suggestions[:5], 1):
                evaluation_feedback += f"{i}. {suggestion}\n"
    
    # Add focus areas if provided
    if input.focus_areas:
        evaluation_feedback += f"\n\nFocus Areas: {', '.join(input.focus_areas)}"
    
    user_message = f"""Please rewrite and improve this prompt:

Original Prompt:
```
{input.prompt_text}
```
{evaluation_feedback}
"""
    
    try:
        # Use configured LLM for rewriting
        if settings.llm_provider == "openai":
            client = openai.AsyncOpenAI(api_key=settings.api_key, timeout=120.0)  # 2 minute timeout
            response = await client.chat.completions.create(
                model=settings.model_name or "gpt-4o",
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7
            )
            response_text = response.choices[0].message.content
            
        elif settings.llm_provider == "claude":
            client = anthropic.AsyncAnthropic(api_key=settings.api_key, timeout=120.0)  # 2 minute timeout
            response = await client.messages.create(
                model=settings.model_name or "claude-3-7-sonnet-20250219",
                max_tokens=4096,
                system=REWRITE_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                timeout=120.0  # 2 minute timeout
            )
            response_text = response.content[0].text
            
        elif settings.llm_provider == "gemini":
            genai.configure(api_key=settings.api_key)
            model_obj = genai.GenerativeModel(settings.model_name or "gemini-2.0-flash-exp")
            response = await model_obj.generate_content_async(
                f"{REWRITE_SYSTEM_PROMPT}\n\n{user_message}"
            )
            response_text = response.text
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {settings.llm_provider}")
        
        # Parse response
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        rewrite_data = json.loads(response_text)
        
        # Add cost calculation
        cost_info = calculate_cost(user_message, response_text, settings.llm_provider, settings.model_name or "gpt-4o")
        rewrite_data["cost"] = cost_info
        
        return rewrite_data
        
    except Exception as e:
        logging.error(f"Rewrite error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Rewrite failed: {str(e)}")


@api_router.post("/playground")
async def test_prompt_in_playground(input: PlaygroundRequest):
    """Test a prompt with sample input in a live playground"""
    # Get settings or use provided provider
    settings_doc = await db.settings.find_one()
    if not settings_doc and not input.llm_provider:
        raise HTTPException(status_code=400, detail="Please configure LLM settings or provide provider details")
    
    if settings_doc:
        settings = Settings(**parse_from_mongo(settings_doc))
        provider = input.llm_provider or settings.llm_provider
        api_key = settings.api_key
        model = input.model_name or settings.model_name or "gpt-4o"
    else:
        provider = input.llm_provider
        api_key = None  # Would need to be provided
        model = input.model_name or "gpt-4o"
        raise HTTPException(status_code=400, detail="Please configure LLM settings first")
    
    # Execute the prompt with the test input
    user_message = input.prompt_text.replace("{input}", input.test_input)
    
    try:
        if provider == "openai":
            client = openai.AsyncOpenAI(api_key=api_key, timeout=60.0)  # 1 minute timeout for playground
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7
            )
            response_text = response.choices[0].message.content
            
        elif provider == "claude":
            client = anthropic.AsyncAnthropic(api_key=api_key, timeout=60.0)  # 1 minute timeout for playground
            response = await client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                timeout=60.0  # 1 minute timeout for playground
            )
            response_text = response.content[0].text
            
        elif provider == "gemini":
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel(model)
            response = await model_obj.generate_content_async(user_message)
            response_text = response.text
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
        
        # Calculate cost
        cost_info = calculate_cost(user_message, response_text, provider, model)
        
        return {
            "prompt_used": user_message,
            "response": response_text,
            "provider": provider,
            "model": model,
            "cost": cost_info
        }
        
    except Exception as e:
        logging.error(f"Playground error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Playground test failed: {str(e)}")


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()