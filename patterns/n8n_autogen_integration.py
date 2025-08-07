"""
n8n + AutoGen Integration Pattern
Comprehensive example showing how to integrate n8n workflow orchestration
with AutoGen multi-agent systems for powerful automation pipelines
"""

import autogen
from typing import Dict, Any, Optional, List
import logging
from utils.logging_utils import setup_logging
import time
import json
import asyncio
import aiohttp
from dataclasses import dataclass, field
from enum import Enum
from flask import Flask, request, jsonify
import redis
import uuid
from datetime import datetime
import requests
import os

logger = setup_logging(__name__)


class WorkflowType(Enum):
    """Types of AutoGen workflows that n8n can trigger"""
    RESEARCH_ANALYSIS = "research_analysis"
    CONTENT_CREATION = "content_creation"
    CODE_REVIEW = "code_review"
    CUSTOMER_SUPPORT = "customer_support"
    DATA_PROCESSING = "data_processing"
    CREATIVE_BRAINSTORM = "creative_brainstorm"


@dataclass
class WorkflowRequest:
    """Request structure from n8n to AutoGen service"""
    workflow_id: str
    workflow_type: WorkflowType
    task: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    n8n_callback_url: Optional[str] = None
    priority: int = 1
    timeout: int = 300  # 5 minutes default


@dataclass
class WorkflowResponse:
    """Response structure from AutoGen back to n8n"""
    workflow_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0
    agent_interactions: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutoGenN8NService:
    """
    Main service that bridges n8n workflows with AutoGen multi-agent systems
    """
    
    def __init__(self, llm_config: Optional[Dict] = None):
        self.llm_config = llm_config or {
            "config_list": [
                {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY", "demo-key"),
                }
            ],
            "temperature": 0.7,
        }
        
        # State management
        self.state_manager = StateManager()
        self.workflow_registry = {}
        
        # Initialize workflow handlers
        self._initialize_workflow_handlers()
        
        # Track active workflows
        self.active_workflows = {}
    
    def _initialize_workflow_handlers(self):
        """Initialize different types of AutoGen workflows"""
        self.workflow_handlers = {
            WorkflowType.RESEARCH_ANALYSIS: self._research_analysis_workflow,
            WorkflowType.CONTENT_CREATION: self._content_creation_workflow,
            WorkflowType.CODE_REVIEW: self._code_review_workflow,
            WorkflowType.CUSTOMER_SUPPORT: self._customer_support_workflow,
            WorkflowType.DATA_PROCESSING: self._data_processing_workflow,
            WorkflowType.CREATIVE_BRAINSTORM: self._creative_brainstorm_workflow,
        }
    
    async def execute_workflow(self, workflow_request: WorkflowRequest) -> WorkflowResponse:
        """Execute an AutoGen workflow triggered by n8n"""
        logger.info(f"Starting workflow {workflow_request.workflow_id} of type {workflow_request.workflow_type.value}")
        
        start_time = time.time()
        
        # Store workflow state
        self.active_workflows[workflow_request.workflow_id] = {
            "status": "running",
            "start_time": start_time,
            "request": workflow_request
        }
        
        try:
            # Get appropriate workflow handler
            handler = self.workflow_handlers.get(workflow_request.workflow_type)
            if not handler:
                raise ValueError(f"Unknown workflow type: {workflow_request.workflow_type}")
            
            # Execute the AutoGen workflow
            result = await handler(workflow_request)
            
            execution_time = time.time() - start_time
            
            # Create response
            response = WorkflowResponse(
                workflow_id=workflow_request.workflow_id,
                success=True,
                result=result["output"],
                execution_time=execution_time,
                agent_interactions=result.get("interactions", []),
                metadata=result.get("metadata", {})
            )
            
            # Update state
            self.active_workflows[workflow_request.workflow_id]["status"] = "completed"
            
            # Send callback to n8n if URL provided
            if workflow_request.n8n_callback_url:
                await self._send_n8n_callback(workflow_request.n8n_callback_url, response)
            
            logger.info(f"Workflow {workflow_request.workflow_id} completed in {execution_time:.2f}s")
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Workflow {workflow_request.workflow_id} failed: {e}")
            
            error_response = WorkflowResponse(
                workflow_id=workflow_request.workflow_id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
            
            # Update state
            self.active_workflows[workflow_request.workflow_id]["status"] = "failed"
            
            return error_response
    
    async def _research_analysis_workflow(self, request: WorkflowRequest) -> Dict[str, Any]:
        """Multi-agent research and analysis workflow"""
        
        # Create specialized research agents
        researcher = autogen.AssistantAgent(
            name="researcher",
            llm_config=self.llm_config,
            system_message="""You are a research specialist. Your role is to:
            1. Gather comprehensive information on given topics
            2. Identify key sources and references  
            3. Organize findings systematically
            4. Highlight important insights and patterns"""
        )
        
        analyst = autogen.AssistantAgent(
            name="analyst",
            llm_config=self.llm_config,
            system_message="""You are a data analyst. Your role is to:
            1. Analyze research findings for patterns and trends
            2. Draw meaningful conclusions from data
            3. Identify implications and recommendations
            4. Create structured summaries of insights"""
        )
        
        synthesizer = autogen.AssistantAgent(
            name="synthesizer", 
            llm_config=self.llm_config,
            system_message="""You are a synthesis specialist. Your role is to:
            1. Combine research and analysis into coherent reports
            2. Create executive summaries and key takeaways
            3. Structure information for different audiences
            4. Ensure clarity and actionable insights"""
        )
        
        # Set up group chat
        groupchat = autogen.GroupChat(
            agents=[researcher, analyst, synthesizer],
            messages=[],
            max_round=8
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config
        )
        
        # Start the research workflow
        research_query = f"Research and analyze: {request.task}"
        if request.parameters.get("focus_areas"):
            research_query += f"\nFocus areas: {', '.join(request.parameters['focus_areas'])}"
        
        researcher.initiate_chat(manager, message=research_query)
        
        return {
            "output": {
                "summary": groupchat.messages[-1]["content"] if groupchat.messages else "No output generated",
                "full_conversation": [msg["content"] for msg in groupchat.messages],
                "agents_involved": ["researcher", "analyst", "synthesizer"]
            },
            "interactions": groupchat.messages,
            "metadata": {
                "total_rounds": len(groupchat.messages),
                "workflow_type": "research_analysis"
            }
        }
    
    async def _content_creation_workflow(self, request: WorkflowRequest) -> Dict[str, Any]:
        """Multi-agent content creation workflow"""
        
        # Content creation agents
        writer = autogen.AssistantAgent(
            name="writer",
            llm_config=self.llm_config,
            system_message="""You are a professional writer. Create engaging, well-structured content that:
            1. Captures the target audience's attention
            2. Delivers clear, valuable information
            3. Maintains consistent tone and style
            4. Includes compelling headlines and structure"""
        )
        
        editor = autogen.AssistantAgent(
            name="editor",
            llm_config=self.llm_config, 
            system_message="""You are an expert editor. Review and improve content by:
            1. Enhancing clarity and readability
            2. Correcting grammar and style issues
            3. Improving flow and structure
            4. Ensuring consistency and accuracy"""
        )
        
        reviewer = autogen.AssistantAgent(
            name="reviewer",
            llm_config=self.llm_config,
            system_message="""You are a content reviewer. Provide final quality assessment:
            1. Check for factual accuracy
            2. Ensure brand voice alignment
            3. Verify target audience appropriateness
            4. Recommend final improvements"""
        )
        
        # Group chat for collaborative content creation
        groupchat = autogen.GroupChat(
            agents=[writer, editor, reviewer],
            messages=[],
            max_round=6
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config
        )
        
        # Content brief
        content_brief = f"Create content: {request.task}"
        if request.parameters.get("target_audience"):
            content_brief += f"\nTarget audience: {request.parameters['target_audience']}"
        if request.parameters.get("tone"):
            content_brief += f"\nTone: {request.parameters['tone']}"
        if request.parameters.get("word_count"):
            content_brief += f"\nTarget length: {request.parameters['word_count']} words"
        
        writer.initiate_chat(manager, message=content_brief)
        
        return {
            "output": {
                "final_content": groupchat.messages[-1]["content"] if groupchat.messages else "No content generated",
                "creation_process": [msg["content"] for msg in groupchat.messages],
                "agents_involved": ["writer", "editor", "reviewer"]
            },
            "interactions": groupchat.messages,
            "metadata": {
                "content_type": request.parameters.get("content_type", "general"),
                "workflow_type": "content_creation"
            }
        }
    
    async def _code_review_workflow(self, request: WorkflowRequest) -> Dict[str, Any]:
        """Multi-agent code review workflow"""
        
        # Code review specialists
        security_reviewer = autogen.AssistantAgent(
            name="security_reviewer",
            llm_config=self.llm_config,
            system_message="""You are a security code reviewer. Focus on:
            1. Identifying security vulnerabilities
            2. Checking for input validation
            3. Reviewing authentication and authorization
            4. Finding potential injection attacks
            5. Assessing data handling security"""
        )
        
        performance_reviewer = autogen.AssistantAgent(
            name="performance_reviewer", 
            llm_config=self.llm_config,
            system_message="""You are a performance code reviewer. Focus on:
            1. Identifying performance bottlenecks
            2. Reviewing algorithm efficiency
            3. Checking memory usage patterns
            4. Assessing scalability concerns
            5. Suggesting optimization opportunities"""
        )
        
        style_reviewer = autogen.AssistantAgent(
            name="style_reviewer",
            llm_config=self.llm_config,
            system_message="""You are a code style and maintainability reviewer. Focus on:
            1. Code readability and clarity
            2. Adherence to coding standards
            3. Proper documentation and comments
            4. Code organization and structure
            5. Maintainability best practices"""
        )
        
        # Group review process
        groupchat = autogen.GroupChat(
            agents=[security_reviewer, performance_reviewer, style_reviewer],
            messages=[],
            max_round=8
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config
        )
        
        # Code review request
        code_to_review = request.task
        review_request = f"Please review this code:\n\n{code_to_review}"
        
        if request.parameters.get("language"):
            review_request += f"\nProgramming language: {request.parameters['language']}"
        if request.parameters.get("context"):
            review_request += f"\nContext: {request.parameters['context']}"
        
        security_reviewer.initiate_chat(manager, message=review_request)
        
        return {
            "output": {
                "review_summary": groupchat.messages[-1]["content"] if groupchat.messages else "No review generated",
                "detailed_reviews": [msg["content"] for msg in groupchat.messages],
                "reviewers": ["security", "performance", "style"]
            },
            "interactions": groupchat.messages,
            "metadata": {
                "code_language": request.parameters.get("language", "unknown"),
                "workflow_type": "code_review"
            }
        }
    
    async def _customer_support_workflow(self, request: WorkflowRequest) -> Dict[str, Any]:
        """Multi-agent customer support workflow"""
        
        # Customer support agents
        classifier = autogen.AssistantAgent(
            name="support_classifier",
            llm_config=self.llm_config,
            system_message="""You are a support ticket classifier. Analyze customer issues and:
            1. Categorize the type of issue (technical, billing, general inquiry, etc.)
            2. Assess urgency level (low, medium, high, critical)
            3. Identify required expertise area
            4. Suggest initial resolution approach"""
        )
        
        technical_agent = autogen.AssistantAgent(
            name="technical_support",
            llm_config=self.llm_config,
            system_message="""You are a technical support specialist. Help with:
            1. Technical troubleshooting and diagnostics
            2. Product feature explanations
            3. Integration and setup guidance
            4. Bug report analysis and workarounds"""
        )
        
        customer_success = autogen.AssistantAgent(
            name="customer_success",
            llm_config=self.llm_config,
            system_message="""You are a customer success specialist. Focus on:
            1. Customer satisfaction and retention
            2. Escalation handling and resolution
            3. Follow-up recommendations
            4. Process improvement suggestions"""
        )
        
        # Support workflow
        groupchat = autogen.GroupChat(
            agents=[classifier, technical_agent, customer_success],
            messages=[],
            max_round=6
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config
        )
        
        # Customer inquiry
        support_request = f"Customer inquiry: {request.task}"
        if request.parameters.get("customer_tier"):
            support_request += f"\nCustomer tier: {request.parameters['customer_tier']}"
        if request.parameters.get("previous_tickets"):
            support_request += f"\nPrevious tickets: {request.parameters['previous_tickets']}"
        
        classifier.initiate_chat(manager, message=support_request)
        
        return {
            "output": {
                "resolution": groupchat.messages[-1]["content"] if groupchat.messages else "No resolution generated",
                "support_process": [msg["content"] for msg in groupchat.messages],
                "agents_consulted": ["classifier", "technical_support", "customer_success"]
            },
            "interactions": groupchat.messages,
            "metadata": {
                "customer_tier": request.parameters.get("customer_tier", "standard"),
                "workflow_type": "customer_support"
            }
        }
    
    async def _data_processing_workflow(self, request: WorkflowRequest) -> Dict[str, Any]:
        """Multi-agent data processing workflow"""
        
        # Data processing agents
        data_analyzer = autogen.AssistantAgent(
            name="data_analyzer",
            llm_config=self.llm_config,
            system_message="""You are a data analyst. Analyze data by:
            1. Examining data structure and quality
            2. Identifying patterns and trends
            3. Detecting anomalies or outliers
            4. Calculating relevant statistics and metrics"""
        )
        
        insights_generator = autogen.AssistantAgent(
            name="insights_generator",
            llm_config=self.llm_config,
            system_message="""You are an insights specialist. Generate insights by:
            1. Interpreting analysis results
            2. Drawing meaningful conclusions
            3. Identifying business implications
            4. Recommending actionable next steps"""
        )
        
        # Data processing workflow
        groupchat = autogen.GroupChat(
            agents=[data_analyzer, insights_generator],
            messages=[],
            max_round=4
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config
        )
        
        # Data processing request
        data_request = f"Process and analyze this data: {request.task}"
        if request.parameters.get("analysis_type"):
            data_request += f"\nAnalysis type: {request.parameters['analysis_type']}"
        
        data_analyzer.initiate_chat(manager, message=data_request)
        
        return {
            "output": {
                "insights": groupchat.messages[-1]["content"] if groupchat.messages else "No insights generated",
                "analysis_process": [msg["content"] for msg in groupchat.messages],
                "processors": ["data_analyzer", "insights_generator"]
            },
            "interactions": groupchat.messages,
            "metadata": {
                "analysis_type": request.parameters.get("analysis_type", "general"),
                "workflow_type": "data_processing"
            }
        }
    
    async def _creative_brainstorm_workflow(self, request: WorkflowRequest) -> Dict[str, Any]:
        """Multi-agent creative brainstorming workflow"""
        
        # Creative agents
        ideator = autogen.AssistantAgent(
            name="creative_ideator",
            llm_config=self.llm_config,
            system_message="""You are a creative ideator. Generate ideas by:
            1. Thinking outside conventional boundaries
            2. Combining concepts in novel ways
            3. Exploring multiple creative directions
            4. Building on and expanding initial concepts"""
        )
        
        critic = autogen.AssistantAgent(
            name="creative_critic",
            llm_config=self.llm_config,
            system_message="""You are a constructive creative critic. Evaluate ideas by:
            1. Assessing feasibility and practicality
            2. Identifying strengths and potential improvements
            3. Suggesting refinements and variations
            4. Balancing creativity with realistic constraints"""
        )
        
        synthesizer = autogen.AssistantAgent(
            name="idea_synthesizer",
            llm_config=self.llm_config,
            system_message="""You are an idea synthesizer. Combine and refine by:
            1. Merging the best elements from multiple ideas
            2. Creating coherent final concepts
            3. Prioritizing ideas by impact and feasibility
            4. Presenting polished creative solutions"""
        )
        
        # Creative brainstorming session
        groupchat = autogen.GroupChat(
            agents=[ideator, critic, synthesizer],
            messages=[],
            max_round=8
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config
        )
        
        # Creative brief
        creative_brief = f"Brainstorm creative solutions for: {request.task}"
        if request.parameters.get("constraints"):
            creative_brief += f"\nConstraints: {request.parameters['constraints']}"
        if request.parameters.get("target_outcome"):
            creative_brief += f"\nDesired outcome: {request.parameters['target_outcome']}"
        
        ideator.initiate_chat(manager, message=creative_brief)
        
        return {
            "output": {
                "final_concepts": groupchat.messages[-1]["content"] if groupchat.messages else "No concepts generated",
                "brainstorm_process": [msg["content"] for msg in groupchat.messages],
                "creative_team": ["ideator", "critic", "synthesizer"]
            },
            "interactions": groupchat.messages,
            "metadata": {
                "creative_domain": request.parameters.get("domain", "general"),
                "workflow_type": "creative_brainstorm"
            }
        }
    
    async def _send_n8n_callback(self, callback_url: str, response: WorkflowResponse):
        """Send completion callback to n8n"""
        try:
            async with aiohttp.ClientSession() as session:
                callback_data = {
                    "workflow_id": response.workflow_id,
                    "success": response.success,
                    "result": response.result,
                    "error": response.error,
                    "execution_time": response.execution_time,
                    "agent_count": len(response.agent_interactions),
                    "completed_at": datetime.now().isoformat()
                }
                
                async with session.post(callback_url, json=callback_data) as resp:
                    if resp.status == 200:
                        logger.info(f"Callback sent successfully for workflow {response.workflow_id}")
                    else:
                        logger.warning(f"Callback failed with status {resp.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send callback: {e}")


class StateManager:
    """Manages workflow state using Redis for persistence"""
    
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=0,
                decode_responses=True
            )
            self.redis_client.ping()  # Test connection
        except:
            logger.warning("Redis not available, using in-memory state")
            self.redis_client = None
            self.memory_state = {}
    
    def save_workflow_state(self, workflow_id: str, state: Dict):
        """Save workflow state"""
        if self.redis_client:
            self.redis_client.setex(
                f"workflow:{workflow_id}",
                3600,  # 1 hour TTL
                json.dumps(state, default=str)
            )
        else:
            self.memory_state[workflow_id] = state
    
    def get_workflow_state(self, workflow_id: str) -> Dict:
        """Get workflow state"""
        if self.redis_client:
            state_data = self.redis_client.get(f"workflow:{workflow_id}")
            return json.loads(state_data) if state_data else {}
        else:
            return self.memory_state.get(workflow_id, {})


class N8NAutoGenApp:
    """Flask application for n8n integration"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.autogen_service = AutoGenN8NService()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes for n8n integration"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                "status": "healthy",
                "service": "autogen-n8n-integration",
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/workflow/execute', methods=['POST'])
        def execute_workflow():
            """Execute AutoGen workflow from n8n"""
            try:
                data = request.json
                
                # Create workflow request
                workflow_request = WorkflowRequest(
                    workflow_id=data.get('workflow_id', str(uuid.uuid4())),
                    workflow_type=WorkflowType(data.get('workflow_type')),
                    task=data.get('task'),
                    parameters=data.get('parameters', {}),
                    n8n_callback_url=data.get('callback_url'),
                    priority=data.get('priority', 1),
                    timeout=data.get('timeout', 300)
                )
                
                # Execute workflow asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    self.autogen_service.execute_workflow(workflow_request)
                )
                
                # Return response
                return jsonify({
                    "workflow_id": result.workflow_id,
                    "success": result.success,
                    "result": result.result,
                    "error": result.error,
                    "execution_time": result.execution_time,
                    "agent_interactions": len(result.agent_interactions),
                    "metadata": result.metadata
                })
                
            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/workflow/status/<workflow_id>', methods=['GET'])
        def get_workflow_status(workflow_id):
            """Get workflow status"""
            workflow_info = self.autogen_service.active_workflows.get(workflow_id)
            
            if not workflow_info:
                return jsonify({
                    "error": "Workflow not found"
                }), 404
            
            return jsonify({
                "workflow_id": workflow_id,
                "status": workflow_info["status"],
                "start_time": workflow_info["start_time"],
                "runtime": time.time() - workflow_info["start_time"]
            })
        
        @self.app.route('/workflows/active', methods=['GET'])
        def list_active_workflows():
            """List all active workflows"""
            active = []
            for wf_id, wf_info in self.autogen_service.active_workflows.items():
                active.append({
                    "workflow_id": wf_id,
                    "status": wf_info["status"],
                    "workflow_type": wf_info["request"].workflow_type.value,
                    "runtime": time.time() - wf_info["start_time"]
                })
            
            return jsonify({
                "active_workflows": active,
                "count": len(active)
            })
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        self.app.run(host=host, port=port, debug=debug)


def create_n8n_workflow_examples():
    """Create example n8n workflow JSON files"""
    
    # Example 1: Research Analysis Workflow
    research_workflow = {
        "name": "AutoGen Research Analysis",
        "nodes": [
            {
                "parameters": {
                    "httpMethod": "POST",
                    "path": "research-trigger",
                    "responseMode": "responseNode",
                    "responseData": "allEntries"
                },
                "name": "Research Webhook",
                "type": "n8n-nodes-base.webhook",
                "position": [240, 300]
            },
            {
                "parameters": {
                    "url": "http://autogen-service:5000/workflow/execute",
                    "sendBody": True,
                    "specifyBody": "json",
                    "jsonBody": """={
  "workflow_type": "research_analysis",
  "task": "{{ $json.research_topic }}",
  "parameters": {
    "focus_areas": {{ $json.focus_areas || [] }},
    "depth": "{{ $json.depth || 'comprehensive' }}"
  },
  "callback_url": "{{ $json.callback_url }}",
  "workflow_id": "{{ $json.workflow_id || $runId }}"
}"""
                },
                "name": "Execute AutoGen Research",
                "type": "n8n-nodes-base.httpRequest",
                "position": [460, 300]
            },
            {
                "parameters": {
                    "conditions": {
                        "boolean": [
                            {
                                "value1": "={{ $json.success }}",
                                "value2": True
                            }
                        ]
                    }
                },
                "name": "Check Success",
                "type": "n8n-nodes-base.if",
                "position": [680, 300]
            },
            {
                "parameters": {
                    "tableName": "research_results",
                    "operation": "create",
                    "dataToSend": "defineFields",
                    "fieldsUi": {
                        "fieldValues": [
                            {
                                "fieldName": "topic",
                                "fieldValue": "={{ $('Research Webhook').first().json.research_topic }}"
                            },
                            {
                                "fieldName": "results",
                                "fieldValue": "={{ $json.result.summary }}"
                            },
                            {
                                "fieldName": "agents_involved", 
                                "fieldValue": "={{ $json.result.agents_involved.join(', ') }}"
                            },
                            {
                                "fieldName": "execution_time",
                                "fieldValue": "={{ $json.execution_time }}"
                            },
                            {
                                "fieldName": "created_at",
                                "fieldValue": "={{ new Date().toISOString() }}"
                            }
                        ]
                    }
                },
                "name": "Save Research Results",
                "type": "n8n-nodes-base.postgres",
                "position": [900, 240]
            },
            {
                "parameters": {
                    "channel": "#research",
                    "text": "Research Analysis Complete!\n\nTopic: {{ $('Research Webhook').first().json.research_topic }}\nExecution Time: {{ $json.execution_time }}s\nAgents: {{ $json.result.agents_involved.join(', ') }}"
                },
                "name": "Notify Slack",
                "type": "n8n-nodes-base.slack",
                "position": [900, 360]
            }
        ],
        "connections": {
            "Research Webhook": {
                "main": [
                    [
                        {
                            "node": "Execute AutoGen Research",
                            "type": "main",
                            "index": 0
                        }
                    ]
                ]
            },
            "Execute AutoGen Research": {
                "main": [
                    [
                        {
                            "node": "Check Success",
                            "type": "main",
                            "index": 0
                        }
                    ]
                ]
            },
            "Check Success": {
                "main": [
                    [
                        {
                            "node": "Save Research Results",
                            "type": "main",
                            "index": 0
                        },
                        {
                            "node": "Notify Slack",
                            "type": "main",
                            "index": 0
                        }
                    ]
                ]
            }
        }
    }
    
    # Save workflow example
    with open('/Users/rkbotla/apps/vibe-experiments/agentic/n8n_research_workflow.json', 'w') as f:
        json.dump(research_workflow, f, indent=2)
    
    logger.info("Created n8n workflow examples")


async def run_integration_examples():
    """Run comprehensive integration examples"""
    logger.info("Starting n8n + AutoGen Integration Examples")
    
    logger.info("\n=== AutoGen Service Examples ===")
    
    # Initialize service
    service = AutoGenN8NService()
    
    # Example 1: Research Analysis
    research_request = WorkflowRequest(
        workflow_id="research_001",
        workflow_type=WorkflowType.RESEARCH_ANALYSIS,
        task="Analyze the impact of artificial intelligence on healthcare",
        parameters={
            "focus_areas": ["diagnostic accuracy", "cost reduction", "patient outcomes"],
            "depth": "comprehensive"
        }
    )
    
    research_result = await service.execute_workflow(research_request)
    logger.info(f"\nResearch Analysis Result:")
    logger.info(f"Success: {research_result.success}")
    logger.info(f"Execution Time: {research_result.execution_time:.2f}s")
    logger.info(f"Agent Interactions: {len(research_result.agent_interactions)}")
    
    # Example 2: Content Creation
    content_request = WorkflowRequest(
        workflow_id="content_001",
        workflow_type=WorkflowType.CONTENT_CREATION,
        task="Create a blog post about sustainable technology trends",
        parameters={
            "target_audience": "tech professionals",
            "tone": "informative yet engaging",
            "word_count": 800,
            "content_type": "blog_post"
        }
    )
    
    content_result = await service.execute_workflow(content_request)
    logger.info(f"\nContent Creation Result:")
    logger.info(f"Success: {content_result.success}")
    logger.info(f"Execution Time: {content_result.execution_time:.2f}s")
    
    # Example 3: Code Review
    code_review_request = WorkflowRequest(
        workflow_id="review_001", 
        workflow_type=WorkflowType.CODE_REVIEW,
        task="""
def process_user_data(data):
    # Process incoming user data
    result = []
    for item in data:
        if item['age'] > 18:
            result.append(item)
    return result
        """,
        parameters={
            "language": "python",
            "context": "user data processing function"
        }
    )
    
    review_result = await service.execute_workflow(code_review_request)
    logger.info(f"\nCode Review Result:")
    logger.info(f"Success: {review_result.success}")
    logger.info(f"Execution Time: {review_result.execution_time:.2f}s")
    
    # Example 4: Creative Brainstorm
    creative_request = WorkflowRequest(
        workflow_id="creative_001",
        workflow_type=WorkflowType.CREATIVE_BRAINSTORM,
        task="Generate innovative marketing ideas for a sustainable fashion brand",
        parameters={
            "constraints": ["limited budget", "digital-first approach"],
            "target_outcome": "increase brand awareness among millennials",
            "domain": "marketing"
        }
    )
    
    creative_result = await service.execute_workflow(creative_request)
    logger.info(f"\nCreative Brainstorm Result:")
    logger.info(f"Success: {creative_result.success}")
    logger.info(f"Execution Time: {creative_result.execution_time:.2f}s")
    
    logger.info("\n=== Integration Pattern Examples ===")
    
    # Demonstrate callback functionality
    logger.info("Testing callback functionality...")
    
    # Simulate n8n callback
    def simulate_n8n_callback(workflow_data):
        logger.info(f"n8n received callback: {workflow_data['workflow_id']}")
        logger.info(f"Result available: {workflow_data['success']}")
        return {"callback_received": True}
    
    # Create workflow examples  
    create_n8n_workflow_examples()
    
    logger.info("n8n + AutoGen Integration Examples Complete")


if __name__ == "__main__":
    # Can be run in two modes:
    
    # 1. As a standalone service (for production)
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        logger.info("Starting n8n-AutoGen integration service...")
        app = N8NAutoGenApp()
        app.run(host='0.0.0.0', port=5000, debug=True)
    
    # 2. As example demonstration
    else:
        logger.info("Running integration examples...")
        asyncio.run(run_integration_examples())