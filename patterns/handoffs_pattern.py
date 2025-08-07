"""
Handoffs Pattern Implementation (Dynamic Task Routing)
Dynamic handoffs enable runtime specialization based on emerging requirements
Agents can transfer tasks to more specialized agents while preserving full context
"""

import autogen
from typing import Dict, Any, Optional, List, Tuple
import logging
from utils.logging_utils import setup_logging
import time
import json
from dataclasses import dataclass, asdict
from enum import Enum

logger = setup_logging(__name__)


class SpecialistType(Enum):
    """Types of specialist agents"""
    CODE = "code_specialist"
    WRITING = "writing_specialist"
    DATA = "data_specialist"
    CREATIVE = "creative_specialist"
    RESEARCH = "research_specialist"
    TECHNICAL = "technical_specialist"
    BUSINESS = "business_specialist"


@dataclass
class HandoffContext:
    """Context preserved during handoffs"""
    task_description: str
    conversation_history: List[Dict[str, str]]
    accumulated_data: Dict[str, Any]
    routing_history: List[str]
    original_requester: str
    timestamp: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def add_routing_step(self, specialist: str):
        """Add a routing step to history"""
        self.routing_history.append(specialist)
    
    def update_data(self, key: str, value: Any):
        """Update accumulated data"""
        self.accumulated_data[key] = value


class HandoffsWorkflow:
    """
    Implements dynamic task routing with context preservation
    """
    
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
        
        self.routing_agents = {}
        self.handoff_history = []
        self.context_compression_enabled = True
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize classifier and specialist agents"""
        
        self.classifier_agent = autogen.AssistantAgent(
            name="task_classifier",
            llm_config=self.llm_config,
            system_message="""You are a task classification expert. Analyze incoming tasks and route them to the most appropriate specialist.
            
Available specialists:
- code_specialist: Programming, debugging, technical implementation
- writing_specialist: Content creation, documentation, copywriting
- data_specialist: Analysis, statistics, data processing, visualization
- creative_specialist: Design, ideation, creative problem solving
- research_specialist: Information gathering, fact-checking, literature review
- technical_specialist: System design, architecture, DevOps, infrastructure
- business_specialist: Strategy, planning, financial analysis

Respond with:
ROUTE_TO: [specialist_type]
REASONING: [why this specialist]
CONFIDENCE: [0.0-1.0]
ALTERNATIVE: [backup specialist if needed]"""
        )
        
        self.routing_agents["code_specialist"] = autogen.AssistantAgent(
            name="code_specialist",
            llm_config=self.llm_config,
            system_message="""You are a programming and technical implementation specialist.
            Handle: code writing, debugging, optimization, technical problem-solving, API design, algorithms."""
        )
        
        self.routing_agents["writing_specialist"] = autogen.AssistantAgent(
            name="writing_specialist",
            llm_config=self.llm_config,
            system_message="""You are a content creation and documentation specialist.
            Handle: technical writing, documentation, articles, reports, copywriting, editing."""
        )
        
        self.routing_agents["data_specialist"] = autogen.AssistantAgent(
            name="data_specialist",
            llm_config=self.llm_config,
            system_message="""You are a data analysis and processing specialist.
            Handle: statistical analysis, data visualization, ETL, machine learning, data cleaning."""
        )
        
        self.routing_agents["creative_specialist"] = autogen.AssistantAgent(
            name="creative_specialist",
            llm_config=self.llm_config,
            system_message="""You are a creative and design specialist.
            Handle: creative problem solving, design thinking, brainstorming, UX/UI, innovation."""
        )
        
        self.routing_agents["research_specialist"] = autogen.AssistantAgent(
            name="research_specialist",
            llm_config=self.llm_config,
            system_message="""You are a research and information specialist.
            Handle: literature review, fact-checking, competitive analysis, market research, academic research."""
        )
        
        self.routing_agents["technical_specialist"] = autogen.AssistantAgent(
            name="technical_specialist",
            llm_config=self.llm_config,
            system_message="""You are a technical architecture and systems specialist.
            Handle: system design, infrastructure, DevOps, cloud architecture, security, scalability."""
        )
        
        self.routing_agents["business_specialist"] = autogen.AssistantAgent(
            name="business_specialist",
            llm_config=self.llm_config,
            system_message="""You are a business strategy and analysis specialist.
            Handle: business planning, financial analysis, market strategy, project management, risk assessment."""
        )
    
    def route_task(self, task_description: str, initial_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Route a task through appropriate specialists"""
        logger.info(f"Starting task routing for: {task_description[:100]}...")
        start_time = time.time()
        
        context = HandoffContext(
            task_description=task_description,
            conversation_history=[],
            accumulated_data=initial_context or {},
            routing_history=[],
            original_requester="user",
            timestamp=start_time,
            metadata={}
        )
        
        routing_decision = self._classify_task(task_description)
        logger.info(f"Initial routing: {routing_decision}")
        
        result = self._execute_with_handoffs(context, routing_decision)
        
        elapsed_time = time.time() - start_time
        
        final_result = {
            "task": task_description,
            "routing_path": context.routing_history,
            "final_response": result,
            "context": context.to_dict(),
            "execution_time": elapsed_time
        }
        
        self.handoff_history.append(final_result)
        return final_result
    
    def _classify_task(self, task_description: str) -> Dict[str, Any]:
        """Classify task and determine routing"""
        message = f"Classify and route this task: {task_description}"
        
        response = self.classifier_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        routing = self._parse_routing_decision(response)
        return routing
    
    def _parse_routing_decision(self, response: str) -> Dict[str, Any]:
        """Parse routing decision from classifier response"""
        routing = {
            "specialist": "code_specialist",
            "confidence": 0.5,
            "reasoning": "",
            "alternative": None
        }
        
        lines = response.split('\n')
        for line in lines:
            if "ROUTE_TO:" in line:
                specialist = line.split(":")[-1].strip()
                if specialist in self.routing_agents:
                    routing["specialist"] = specialist
            elif "CONFIDENCE:" in line:
                try:
                    routing["confidence"] = float(line.split(":")[-1].strip())
                except:
                    pass
            elif "REASONING:" in line:
                routing["reasoning"] = line.split(":")[-1].strip()
            elif "ALTERNATIVE:" in line:
                alt = line.split(":")[-1].strip()
                if alt in self.routing_agents:
                    routing["alternative"] = alt
        
        return routing
    
    def _execute_with_handoffs(
        self, 
        context: HandoffContext, 
        routing: Dict[str, Any]
    ) -> str:
        """Execute task with potential handoffs"""
        specialist = routing["specialist"]
        context.add_routing_step(specialist)
        
        logger.info(f"Handing off to {specialist}")
        
        if self.context_compression_enabled:
            compressed_context = self._compress_context(context)
        else:
            compressed_context = context.to_dict()
        
        specialist_agent = self.routing_agents[specialist]
        
        message = f"""Task: {context.task_description}

Context: {json.dumps(compressed_context, indent=2)}

Please handle this task according to your specialization."""
        
        response = specialist_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        context.conversation_history.append({
            "specialist": specialist,
            "response": response
        })
        
        if self._needs_further_handoff(response, routing["confidence"]):
            next_specialist = self._determine_next_specialist(response, context)
            if next_specialist and next_specialist != specialist:
                logger.info(f"Additional handoff needed to {next_specialist}")
                next_routing = {"specialist": next_specialist, "confidence": 0.7}
                return self._execute_with_handoffs(context, next_routing)
        
        return response
    
    def _compress_context(self, context: HandoffContext) -> Dict[str, Any]:
        """Compress context for token efficiency"""
        compressed = {
            "task": context.task_description[:200],
            "routing": context.routing_history,
            "key_data": {}
        }
        
        if context.conversation_history:
            compressed["last_response"] = context.conversation_history[-1]["response"][:500]
        
        for key, value in context.accumulated_data.items():
            if isinstance(value, (str, int, float, bool)):
                compressed["key_data"][key] = value
            elif isinstance(value, dict):
                compressed["key_data"][key] = {k: v for k, v in list(value.items())[:3]}
        
        return compressed
    
    def _needs_further_handoff(self, response: str, confidence: float) -> bool:
        """Determine if task needs another specialist"""
        handoff_indicators = [
            "need more expertise",
            "requires specialist",
            "beyond my scope",
            "would benefit from",
            "recommend consulting"
        ]
        
        if confidence < 0.6:
            return True
        
        return any(indicator in response.lower() for indicator in handoff_indicators)
    
    def _determine_next_specialist(
        self, 
        response: str, 
        context: HandoffContext
    ) -> Optional[str]:
        """Determine next specialist based on response"""
        if len(context.routing_history) >= 3:
            logger.info("Maximum handoff depth reached")
            return None
        
        for specialist_type in SpecialistType:
            if specialist_type.value in response.lower() and \
               specialist_type.value not in context.routing_history:
                return specialist_type.value
        
        return None
    
    def transfer_to_specialist(
        self, 
        task_description: str, 
        context: Dict[str, Any], 
        specialist_type: str
    ) -> Dict[str, Any]:
        """Direct transfer to specific specialist"""
        logger.info(f"Direct transfer to {specialist_type}")
        
        handoff_context = HandoffContext(
            task_description=task_description,
            conversation_history=[],
            accumulated_data=context,
            routing_history=[],
            original_requester="direct_transfer",
            timestamp=time.time(),
            metadata={"direct_transfer": True}
        )
        
        routing = {"specialist": specialist_type, "confidence": 1.0}
        result = self._execute_with_handoffs(handoff_context, routing)
        
        return {
            "specialist": specialist_type,
            "result": result,
            "context": handoff_context.to_dict()
        }


class EscalationWorkflow(HandoffsWorkflow):
    """Extended handoffs with escalation tiers"""
    
    def __init__(self, llm_config: Optional[Dict] = None):
        super().__init__(llm_config)
        self.escalation_tiers = {
            "L1": ["research_specialist", "writing_specialist"],
            "L2": ["code_specialist", "data_specialist", "creative_specialist"],
            "L3": ["technical_specialist", "business_specialist"]
        }
    
    def escalate_task(
        self, 
        task: str, 
        starting_tier: str = "L1"
    ) -> Dict[str, Any]:
        """Escalate task through support tiers"""
        logger.info(f"Starting escalation from {starting_tier}")
        
        context = HandoffContext(
            task_description=task,
            conversation_history=[],
            accumulated_data={"escalation_tier": starting_tier},
            routing_history=[],
            original_requester="escalation",
            timestamp=time.time(),
            metadata={"escalation": True}
        )
        
        current_tier = starting_tier
        escalation_path = []
        
        for tier in ["L1", "L2", "L3"]:
            if tier < current_tier:
                continue
            
            tier_specialists = self.escalation_tiers[tier]
            best_specialist = tier_specialists[0]
            
            logger.info(f"Escalating to {tier}: {best_specialist}")
            escalation_path.append(f"{tier}:{best_specialist}")
            
            routing = {"specialist": best_specialist, "confidence": 0.8}
            result = self._execute_with_handoffs(context, routing)
            
            if self._is_resolved(result):
                logger.info(f"Issue resolved at {tier}")
                break
        
        return {
            "task": task,
            "escalation_path": escalation_path,
            "final_tier": tier,
            "resolution": result,
            "context": context.to_dict()
        }
    
    def _is_resolved(self, response: str) -> bool:
        """Check if issue is resolved"""
        resolution_indicators = [
            "resolved",
            "completed",
            "fixed",
            "solution provided",
            "successfully"
        ]
        return any(indicator in response.lower() for indicator in resolution_indicators)


def run_handoffs_examples():
    """Demonstrate Handoffs pattern"""
    logger.info("Starting Handoffs Pattern Examples")
    
    logger.info("\n=== Basic Task Routing Example ===")
    
    workflow = HandoffsWorkflow()
    
    task1 = "Write a Python function to calculate fibonacci numbers and optimize it for performance"
    
    result1 = workflow.route_task(task1)
    
    logger.info(f"\nTask: {task1}")
    logger.info(f"Routing path: {' -> '.join(result1['routing_path'])}")
    logger.info(f"Execution time: {result1['execution_time']:.2f}s")
    
    logger.info("\n=== Multi-Specialist Handoff Example ===")
    
    task2 = "Analyze sales data to identify trends and create a business strategy presentation"
    
    result2 = workflow.route_task(task2)
    
    logger.info(f"\nTask: {task2}")
    logger.info(f"Routing path: {' -> '.join(result2['routing_path'])}")
    
    logger.info("\n=== Direct Transfer Example ===")
    
    task3 = "Create a creative marketing campaign"
    context3 = {"target_audience": "millennials", "product": "eco-friendly water bottle"}
    
    result3 = workflow.transfer_to_specialist(
        task3,
        context3,
        "creative_specialist"
    )
    
    logger.info(f"\nDirect transfer task: {task3}")
    logger.info(f"Specialist: {result3['specialist']}")
    
    logger.info("\n=== Escalation Workflow Example ===")
    
    escalation_workflow = EscalationWorkflow()
    
    issue = "Customer reporting critical bug in payment processing system"
    
    escalation_result = escalation_workflow.escalate_task(issue, starting_tier="L1")
    
    logger.info(f"\nEscalation issue: {issue}")
    logger.info(f"Escalation path: {' -> '.join(escalation_result['escalation_path'])}")
    logger.info(f"Resolved at tier: {escalation_result['final_tier']}")
    
    logger.info("\n=== Context Preservation Example ===")
    
    complex_task = "Research quantum computing applications and write technical documentation"
    initial_context = {
        "technical_level": "intermediate",
        "target_length": "5000 words",
        "include_examples": True
    }
    
    complex_result = workflow.route_task(complex_task, initial_context)
    
    logger.info(f"\nComplex task with context preservation")
    logger.info(f"Initial context keys: {list(initial_context.keys())}")
    logger.info(f"Preserved in final context: {list(complex_result['context']['accumulated_data'].keys())}")
    
    logger.info("\nHandoffs Pattern Examples Complete")


if __name__ == "__main__":
    run_handoffs_examples()