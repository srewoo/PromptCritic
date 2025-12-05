"""
Prompt Chain Orchestrator
Build and execute multi-step prompt workflows with dependencies
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a workflow step"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionMode(Enum):
    """Workflow execution mode"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DAG = "dag"  # Directed Acyclic Graph


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow"""
    step_id: str
    name: str
    prompt: str
    dependencies: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {"max_retries": 3, "delay": 1})
    fallback_prompt: Optional[str] = None
    timeout: int = 60
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Workflow:
    """Represents a complete workflow"""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    execution_mode: ExecutionMode
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class PromptChainOrchestrator:
    """Build and optimize multi-step prompt workflows"""
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def create_workflow(
        self,
        name: str,
        description: str,
        steps_config: List[Dict[str, Any]],
        execution_mode: ExecutionMode = ExecutionMode.DAG
    ) -> Workflow:
        """
        Create a new workflow
        
        Args:
            name: Workflow name
            description: Workflow description
            steps_config: List of step configurations
            execution_mode: How to execute steps
            
        Returns:
            Created Workflow object
        """
        workflow_id = str(uuid.uuid4())
        
        # Create workflow steps
        steps = []
        for step_config in steps_config:
            step = WorkflowStep(
                step_id=step_config.get('step_id', str(uuid.uuid4())),
                name=step_config['name'],
                prompt=step_config['prompt'],
                dependencies=step_config.get('dependencies', []),
                outputs=step_config.get('outputs', []),
                retry_policy=step_config.get('retry_policy', {"max_retries": 3, "delay": 1}),
                fallback_prompt=step_config.get('fallback_prompt'),
                timeout=step_config.get('timeout', 60)
            )
            steps.append(step)
        
        # Optimize connections (simplify dependencies)
        optimized_steps = self._optimize_data_flow(steps)
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            steps=optimized_steps,
            execution_mode=execution_mode
        )
        
        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow {workflow_id}: {name}")
        
        return workflow
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        llm_executor: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow with parallel processing where possible
        
        Args:
            workflow_id: Workflow to execute
            input_data: Initial input data
            llm_executor: Function to execute LLM calls (async)
            
        Returns:
            Workflow execution results
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Build execution graph
        execution_graph = self._build_execution_graph(workflow)
        
        # Execute by levels (topological sort)
        results = {"input": input_data}
        
        for level_num, level_steps in enumerate(execution_graph, 1):
            logger.info(f"Executing level {level_num} with {len(level_steps)} steps")
            
            # Execute all steps in this level in parallel
            if workflow.execution_mode == ExecutionMode.PARALLEL or workflow.execution_mode == ExecutionMode.DAG:
                level_results = await asyncio.gather(*[
                    self._execute_step(step, input_data, results, llm_executor)
                    for step in level_steps
                ], return_exceptions=True)
            else:
                # Sequential execution
                level_results = []
                for step in level_steps:
                    result = await self._execute_step(step, input_data, results, llm_executor)
                    level_results.append(result)
            
            # Update results
            for i, step in enumerate(level_steps):
                if isinstance(level_results[i], Exception):
                    step.status = StepStatus.FAILED
                    step.error = str(level_results[i])
                    logger.error(f"Step {step.step_id} failed: {step.error}")
                else:
                    step.status = StepStatus.COMPLETED
                    step.result = level_results[i]
                    results[step.step_id] = level_results[i]
        
        # Record execution
        execution_record = {
            "workflow_id": workflow_id,
            "execution_id": str(uuid.uuid4()),
            "input_data": input_data,
            "results": results,
            "steps_executed": len(workflow.steps),
            "steps_succeeded": len([s for s in workflow.steps if s.status == StepStatus.COMPLETED]),
            "steps_failed": len([s for s in workflow.steps if s.status == StepStatus.FAILED]),
            "executed_at": datetime.utcnow().isoformat()
        }
        self.execution_history.append(execution_record)
        
        return {
            "workflow_id": workflow_id,
            "execution_id": execution_record["execution_id"],
            "status": "completed" if all(s.status == StepStatus.COMPLETED for s in workflow.steps) else "partial",
            "results": results,
            "steps": [self._step_to_dict(s) for s in workflow.steps]
        }
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        input_data: Dict[str, Any],
        previous_results: Dict[str, Any],
        llm_executor: Optional[Callable]
    ) -> Dict[str, Any]:
        """Execute a single workflow step with retry logic"""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.utcnow()
        
        # Prepare step input from dependencies
        step_input = self._prepare_step_input(step, input_data, previous_results)
        
        # Format prompt with input data
        formatted_prompt = self._format_prompt(step.prompt, step_input)
        
        retry_count = 0
        max_retries = step.retry_policy.get("max_retries", 3)
        retry_delay = step.retry_policy.get("delay", 1)
        
        while retry_count <= max_retries:
            try:
                # Execute with timeout
                if llm_executor:
                    result = await asyncio.wait_for(
                        llm_executor(formatted_prompt),
                        timeout=step.timeout
                    )
                else:
                    # Mock execution for testing
                    await asyncio.sleep(0.1)
                    result = {
                        "step_id": step.step_id,
                        "output": f"Mock output for {step.name}",
                        "prompt": formatted_prompt
                    }
                
                step.completed_at = datetime.utcnow()
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Step {step.step_id} timed out after {step.timeout}s")
                if step.fallback_prompt and retry_count == max_retries:
                    # Try fallback
                    formatted_fallback = self._format_prompt(step.fallback_prompt, step_input)
                    if llm_executor:
                        result = await llm_executor(formatted_fallback)
                        step.completed_at = datetime.utcnow()
                        return result
                raise
                
            except Exception as e:
                logger.error(f"Step {step.step_id} failed (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                retry_count += 1
                if retry_count <= max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    step.error = str(e)
                    step.completed_at = datetime.utcnow()
                    raise
    
    def _build_execution_graph(self, workflow: Workflow) -> List[List[WorkflowStep]]:
        """
        Build execution graph using topological sort
        Returns levels of steps that can be executed in parallel
        """
        # Build dependency graph
        step_map = {step.step_id: step for step in workflow.steps}
        in_degree = {step.step_id: len(step.dependencies) for step in workflow.steps}
        
        levels = []
        remaining_steps = set(step_map.keys())
        
        while remaining_steps:
            # Find steps with no remaining dependencies
            current_level = [
                step_map[step_id]
                for step_id in remaining_steps
                if in_degree[step_id] == 0
            ]
            
            if not current_level:
                # Circular dependency detected
                raise ValueError("Circular dependency detected in workflow")
            
            levels.append(current_level)
            
            # Remove completed steps and update dependencies
            for step in current_level:
                remaining_steps.remove(step.step_id)
                
                # Decrease in-degree for dependent steps
                for other_step_id in remaining_steps:
                    other_step = step_map[other_step_id]
                    if step.step_id in other_step.dependencies:
                        in_degree[other_step_id] -= 1
        
        return levels
    
    def _optimize_data_flow(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Optimize connection points between steps"""
        # For now, just validate dependencies exist
        step_ids = {step.step_id for step in steps}
        
        for step in steps:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    logger.warning(f"Step {step.step_id} has invalid dependency: {dep_id}")
                    step.dependencies.remove(dep_id)
        
        return steps
    
    def _prepare_step_input(
        self,
        step: WorkflowStep,
        input_data: Dict[str, Any],
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare input for a step from dependencies"""
        step_input = {"initial_input": input_data}
        
        # Collect outputs from dependencies
        for dep_id in step.dependencies:
            if dep_id in previous_results:
                step_input[f"dep_{dep_id}"] = previous_results[dep_id]
        
        return step_input
    
    def _format_prompt(self, prompt_template: str, step_input: Dict[str, Any]) -> str:
        """Format prompt template with input data"""
        formatted = prompt_template
        
        # Replace placeholders like {variable_name}
        for key, value in step_input.items():
            placeholder = f"{{{key}}}"
            if placeholder in formatted:
                formatted = formatted.replace(placeholder, str(value))
        
        return formatted
    
    def _step_to_dict(self, step: WorkflowStep) -> Dict[str, Any]:
        """Convert step to dictionary"""
        return {
            "step_id": step.step_id,
            "name": step.name,
            "status": step.status.value,
            "dependencies": step.dependencies,
            "outputs": step.outputs,
            "result": step.result,
            "error": step.error,
            "started_at": step.started_at.isoformat() if step.started_at else None,
            "completed_at": step.completed_at.isoformat() if step.completed_at else None
        }
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID"""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows"""
        return [
            {
                "workflow_id": w.workflow_id,
                "name": w.name,
                "description": w.description,
                "steps_count": len(w.steps),
                "execution_mode": w.execution_mode.value,
                "created_at": w.created_at.isoformat()
            }
            for w in self.workflows.values()
        ]
    
    def get_execution_history(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get execution history"""
        if workflow_id:
            return [
                record for record in self.execution_history
                if record["workflow_id"] == workflow_id
            ]
        return self.execution_history


# Singleton instance
_orchestrator = None

def get_orchestrator() -> PromptChainOrchestrator:
    """Get singleton orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PromptChainOrchestrator()
    return _orchestrator
