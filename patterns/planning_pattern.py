"""
Planning Pattern Implementation
Agents that create and execute structured plans before acting
"""

import autogen
from typing import Dict, Any, Optional, List, Tuple
import logging
from utils.logging_utils import setup_logging
import time
import json
from enum import Enum

logger = setup_logging(__name__)


class PlanStatus(Enum):
    """Status of plan execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class PlanStep:
    """Represents a single step in a plan"""
    
    def __init__(
        self,
        step_id: str,
        description: str,
        dependencies: List[str] = None,
        estimated_time: float = 0,
        resources_needed: List[str] = None
    ):
        self.step_id = step_id
        self.description = description
        self.dependencies = dependencies or []
        self.estimated_time = estimated_time
        self.resources_needed = resources_needed or []
        self.status = PlanStatus.PENDING
        self.result = None
        self.actual_time = 0
        self.error = None
    
    def can_execute(self, completed_steps: List[str]) -> bool:
        """Check if step can be executed based on dependencies"""
        return all(dep in completed_steps for dep in self.dependencies)
    
    def to_dict(self) -> Dict:
        """Convert step to dictionary"""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "estimated_time": self.estimated_time,
            "actual_time": self.actual_time,
            "resources_needed": self.resources_needed,
            "result": self.result,
            "error": self.error
        }


class Plan:
    """Represents a complete plan"""
    
    def __init__(self, goal: str):
        self.goal = goal
        self.steps: List[PlanStep] = []
        self.created_at = time.time()
        self.completed_at = None
        self.total_estimated_time = 0
        self.total_actual_time = 0
    
    def add_step(self, step: PlanStep):
        """Add a step to the plan"""
        self.steps.append(step)
        self.total_estimated_time += step.estimated_time
    
    def get_executable_steps(self) -> List[PlanStep]:
        """Get steps that can be executed now"""
        completed = [s.step_id for s in self.steps 
                    if s.status == PlanStatus.COMPLETED]
        
        executable = []
        for step in self.steps:
            if (step.status == PlanStatus.PENDING and 
                step.can_execute(completed)):
                executable.append(step)
        
        return executable
    
    def is_complete(self) -> bool:
        """Check if plan is complete"""
        return all(s.status in [PlanStatus.COMPLETED, PlanStatus.FAILED] 
                  for s in self.steps)
    
    def get_progress(self) -> float:
        """Get plan completion progress (0-1)"""
        if not self.steps:
            return 0
        
        completed = sum(1 for s in self.steps 
                       if s.status == PlanStatus.COMPLETED)
        return completed / len(self.steps)
    
    def to_dict(self) -> Dict:
        """Convert plan to dictionary"""
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "progress": self.get_progress(),
            "total_estimated_time": self.total_estimated_time,
            "total_actual_time": self.total_actual_time,
            "is_complete": self.is_complete()
        }


class PlanningWorkflow:
    """Main workflow for planning agents"""
    
    def __init__(self, llm_config: Optional[Dict] = None):
        self.llm_config = llm_config or {
            "config_list": [
                {
                    "model": "gpt-3.5-turbo",
                    "api_key": "dummy_key_for_demonstration",
                }
            ],
            "temperature": 0.7,
        }
        
        self.plans: List[Plan] = []
        self.execution_history = []
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize planning and execution agents"""
        
        self.planner_agent = autogen.AssistantAgent(
            name="planner",
            llm_config=self.llm_config,
            system_message="""You are a strategic planner. Break down complex goals into:
            1. Clear, actionable steps
            2. Identify dependencies between steps
            3. Estimate time/resources needed
            4. Consider potential risks and contingencies
            
            Format your plan as:
            PLAN:
            Step 1: [Description] | Dependencies: [] | Time: X
            Step 2: [Description] | Dependencies: [1] | Time: Y
            ..."""
        )
        
        self.executor_agent = autogen.AssistantAgent(
            name="executor",
            llm_config=self.llm_config,
            system_message="""You are an execution agent. For each plan step:
            1. Execute the described action
            2. Monitor progress and results
            3. Handle errors gracefully
            4. Report completion status"""
        )
        
        self.monitor_agent = autogen.AssistantAgent(
            name="monitor",
            llm_config=self.llm_config,
            system_message="""You are a monitoring agent. Track plan execution:
            1. Monitor progress and dependencies
            2. Identify bottlenecks or issues
            3. Suggest plan adjustments if needed
            4. Ensure quality standards are met"""
        )
        
        self.replanner_agent = autogen.AssistantAgent(
            name="replanner",
            llm_config=self.llm_config,
            system_message="""You are a replanning agent. When plans fail or need adjustment:
            1. Analyze what went wrong
            2. Identify alternative approaches
            3. Create revised plan steps
            4. Maintain original goal achievement"""
        )
    
    def create_plan(self, goal: str) -> Plan:
        """Create a plan for achieving a goal"""
        logger.info(f"Creating plan for goal: {goal[:100]}...")
        
        message = f"Create a detailed plan to achieve this goal: {goal}"
        
        response = self.planner_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        plan = self._parse_plan(response, goal)
        self.plans.append(plan)
        
        logger.info(f"Created plan with {len(plan.steps)} steps")
        return plan
    
    def _parse_plan(self, plan_text: str, goal: str) -> Plan:
        """Parse plan text into structured Plan object"""
        plan = Plan(goal)
        
        lines = plan_text.split('\n')
        step_counter = 0
        
        for line in lines:
            if 'Step' in line or 'step' in line:
                step_counter += 1
                
                dependencies = []
                if 'Dependencies:' in line or 'dependencies:' in line:
                    dep_part = line.split('Dependencies:')[-1].split('|')[0]
                    dep_text = dep_part.strip().strip('[]')
                    if dep_text and dep_text != 'none':
                        dependencies = [f"step_{d.strip()}" for d in dep_text.split(',')]
                
                estimated_time = 1.0
                if 'Time:' in line or 'time:' in line:
                    try:
                        time_part = line.split('Time:')[-1].strip()
                        estimated_time = float(time_part.split()[0])
                    except:
                        pass
                
                description = line.split('|')[0].strip()
                if ':' in description:
                    description = description.split(':', 1)[1].strip()
                
                step = PlanStep(
                    step_id=f"step_{step_counter}",
                    description=description,
                    dependencies=dependencies,
                    estimated_time=estimated_time
                )
                
                plan.add_step(step)
        
        if not plan.steps:
            step = PlanStep(
                step_id="step_1",
                description="Execute the goal directly",
                dependencies=[],
                estimated_time=1.0
            )
            plan.add_step(step)
        
        return plan
    
    def execute_plan(self, plan: Plan) -> Dict[str, Any]:
        """Execute a plan step by step"""
        logger.info(f"Executing plan for goal: {plan.goal[:100]}...")
        start_time = time.time()
        
        execution_log = []
        
        while not plan.is_complete():
            executable_steps = plan.get_executable_steps()
            
            if not executable_steps:
                logger.warning("No executable steps available")
                
                blocked_steps = [s for s in plan.steps 
                               if s.status == PlanStatus.PENDING]
                if blocked_steps:
                    self._handle_blocked_plan(plan, blocked_steps)
                break
            
            for step in executable_steps:
                step_result = self._execute_step(step)
                execution_log.append(step_result)
                
                progress = plan.get_progress()
                logger.info(f"Plan progress: {progress:.1%}")
        
        plan.completed_at = time.time()
        plan.total_actual_time = plan.completed_at - start_time
        
        result = {
            "plan": plan.to_dict(),
            "execution_log": execution_log,
            "total_time": plan.total_actual_time,
            "success": all(s.status == PlanStatus.COMPLETED for s in plan.steps)
        }
        
        self.execution_history.append(result)
        return result
    
    def _execute_step(self, step: PlanStep) -> Dict[str, Any]:
        """Execute a single plan step"""
        logger.info(f"Executing step: {step.step_id} - {step.description}")
        
        step.status = PlanStatus.IN_PROGRESS
        start_time = time.time()
        
        message = f"Execute this step: {step.description}"
        
        try:
            response = self.executor_agent.generate_reply(
                messages=[{"role": "user", "content": message}]
            )
            
            step.result = response
            step.status = PlanStatus.COMPLETED
            status_msg = "completed"
            
        except Exception as e:
            step.error = str(e)
            step.status = PlanStatus.FAILED
            status_msg = f"failed: {e}"
            logger.error(f"Step {step.step_id} failed: {e}")
        
        step.actual_time = time.time() - start_time
        
        return {
            "step_id": step.step_id,
            "description": step.description,
            "status": step.status.value,
            "execution_time": step.actual_time,
            "result": step.result,
            "error": step.error
        }
    
    def _handle_blocked_plan(self, plan: Plan, blocked_steps: List[PlanStep]):
        """Handle a blocked plan by replanning"""
        logger.info(f"Plan blocked, attempting to replan...")
        
        blocked_info = [{"step_id": s.step_id, "description": s.description, 
                        "dependencies": s.dependencies} for s in blocked_steps]
        
        message = f"""The plan is blocked. Blocked steps:
{json.dumps(blocked_info, indent=2)}

Create alternative steps to unblock the plan or work around the blockage."""
        
        response = self.replanner_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        for step in blocked_steps:
            step.status = PlanStatus.FAILED
            step.error = "Blocked by dependencies"
    
    def adaptive_planning(self, goal: str, max_replans: int = 2) -> Dict[str, Any]:
        """Create and execute plan with adaptive replanning"""
        logger.info(f"Starting adaptive planning for: {goal[:100]}...")
        
        attempts = []
        success = False
        
        for attempt in range(max_replans + 1):
            logger.info(f"\n=== Planning Attempt {attempt + 1} ===")
            
            if attempt == 0:
                plan = self.create_plan(goal)
            else:
                plan = self._create_revised_plan(goal, attempts[-1])
            
            result = self.execute_plan(plan)
            attempts.append(result)
            
            if result["success"]:
                success = True
                break
            
            logger.info(f"Plan failed, attempt {attempt + 1} of {max_replans + 1}")
        
        return {
            "goal": goal,
            "attempts": attempts,
            "success": success,
            "total_attempts": len(attempts),
            "final_plan": attempts[-1]["plan"] if attempts else None
        }
    
    def _create_revised_plan(self, goal: str, previous_result: Dict) -> Plan:
        """Create a revised plan based on previous failure"""
        failures = [s for s in previous_result["plan"]["steps"] 
                   if s["status"] == "failed"]
        
        message = f"""Previous plan for goal '{goal}' failed.
        
Failed steps: {json.dumps(failures, indent=2)}

Create a revised plan that avoids these failures and achieves the goal."""
        
        response = self.replanner_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        return self._parse_plan(response, goal)


class HierarchicalPlanning(PlanningWorkflow):
    """Hierarchical planning with sub-goals and nested plans"""
    
    def create_hierarchical_plan(self, goal: str, max_depth: int = 2) -> Dict:
        """Create a hierarchical plan with sub-plans"""
        logger.info(f"Creating hierarchical plan for: {goal[:100]}...")
        
        return self._create_plan_level(goal, 0, max_depth)
    
    def _create_plan_level(self, goal: str, depth: int, max_depth: int) -> Dict:
        """Create a plan level in the hierarchy"""
        plan = self.create_plan(goal)
        
        hierarchical_plan = {
            "goal": goal,
            "depth": depth,
            "plan": plan.to_dict(),
            "sub_plans": []
        }
        
        if depth < max_depth:
            for step in plan.steps:
                if self._needs_subplan(step):
                    sub_plan = self._create_plan_level(
                        step.description,
                        depth + 1,
                        max_depth
                    )
                    hierarchical_plan["sub_plans"].append(sub_plan)
        
        return hierarchical_plan
    
    def _needs_subplan(self, step: PlanStep) -> bool:
        """Determine if a step needs a sub-plan"""
        complex_indicators = ["multiple", "several", "various", "complex", "detailed"]
        return any(indicator in step.description.lower() for indicator in complex_indicators)


def run_planning_examples():
    """Demonstrate Planning patterns"""
    logger.info("Starting Planning Pattern Examples")
    
    logger.info("\n=== Basic Planning Example ===")
    
    workflow = PlanningWorkflow()
    
    goal = "Create a simple web application with user authentication"
    
    plan = workflow.create_plan(goal)
    
    logger.info(f"\nGoal: {goal}")
    logger.info(f"Plan created with {len(plan.steps)} steps:")
    for step in plan.steps:
        logger.info(f"  - {step.step_id}: {step.description}")
        if step.dependencies:
            logger.info(f"    Dependencies: {step.dependencies}")
    
    execution_result = workflow.execute_plan(plan)
    
    logger.info(f"\nExecution completed:")
    logger.info(f"  Success: {execution_result['success']}")
    logger.info(f"  Total time: {execution_result['total_time']:.2f}s")
    logger.info(f"  Progress: {plan.get_progress():.1%}")
    
    logger.info("\n=== Adaptive Planning Example ===")
    
    adaptive_workflow = PlanningWorkflow()
    
    complex_goal = "Optimize database performance and implement caching"
    
    adaptive_result = adaptive_workflow.adaptive_planning(
        complex_goal,
        max_replans=2
    )
    
    logger.info(f"\nAdaptive planning for: {complex_goal}")
    logger.info(f"  Success: {adaptive_result['success']}")
    logger.info(f"  Total attempts: {adaptive_result['total_attempts']}")
    
    logger.info("\n=== Hierarchical Planning Example ===")
    
    hierarchical_workflow = HierarchicalPlanning()
    
    complex_project = "Build a machine learning pipeline for customer churn prediction"
    
    hierarchical_plan = hierarchical_workflow.create_hierarchical_plan(
        complex_project,
        max_depth=2
    )
    
    logger.info(f"\nHierarchical plan for: {complex_project}")
    logger.info(f"  Main plan steps: {len(hierarchical_plan['plan']['steps'])}")
    logger.info(f"  Sub-plans created: {len(hierarchical_plan['sub_plans'])}")
    
    def print_hierarchy(plan_dict, indent=0):
        prefix = "  " * indent
        logger.info(f"{prefix}Goal: {plan_dict['goal'][:50]}...")
        logger.info(f"{prefix}Steps: {len(plan_dict['plan']['steps'])}")
        for sub_plan in plan_dict.get('sub_plans', []):
            print_hierarchy(sub_plan, indent + 1)
    
    print_hierarchy(hierarchical_plan)
    
    logger.info("\n=== Planning with Dependencies Example ===")
    
    dependency_workflow = PlanningWorkflow()
    
    dependency_goal = "Deploy application to production with zero downtime"
    
    dep_plan = dependency_workflow.create_plan(dependency_goal)
    
    logger.info(f"\nPlan with dependencies for: {dependency_goal}")
    
    dependency_graph = {}
    for step in dep_plan.steps:
        dependency_graph[step.step_id] = step.dependencies
    
    logger.info("Dependency graph:")
    for step_id, deps in dependency_graph.items():
        if deps:
            logger.info(f"  {step_id} depends on: {deps}")
        else:
            logger.info(f"  {step_id} has no dependencies (can start immediately)")
    
    logger.info("\nPlanning Pattern Examples Complete")


if __name__ == "__main__":
    run_planning_examples()