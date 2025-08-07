#!/usr/bin/env python3
"""
Sequential Workflow Pattern Implementation - AutoGen v0.7.1

This example demonstrates a research paper analysis pipeline:
1. Researcher Agent: Gathers information about a topic
2. Analyst Agent: Analyzes and structures the information
3. Writer Agent: Creates a final report
4. Reviewer Agent: Reviews and provides final feedback

Key Features (v0.7.1):
- Async-first architecture
- Modern agent creation with model clients
- State preservation between steps
- Error handling and recovery
- Progress tracking and checkpointing
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from config import Config
from utils.logging_utils import AgentLogger

class SequentialWorkflowState:
    """Manages state across sequential workflow steps"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.current_step = 0
        self.steps_completed = []
        self.accumulated_data = {}
        self.conversation_history = []
        self.checkpoints = []
    
    def checkpoint(self, step_name: str, data: Dict[str, Any]):
        """Create a checkpoint for recovery"""
        checkpoint = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "step_number": self.current_step
        }
        self.checkpoints.append(checkpoint)
        
        # Save to file for persistence
        Path("logs").mkdir(exist_ok=True)
        checkpoint_file = f"logs/checkpoint_{self.task_id}_{step_name}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def restore_from_checkpoint(self, step_name: str) -> Dict[str, Any]:
        """Restore state from a specific checkpoint"""
        checkpoint_file = f"logs/checkpoint_{self.task_id}_{step_name}.json"
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            return checkpoint['data']
        except FileNotFoundError:
            return {}

class SequentialResearchWorkflow:
    """Sequential workflow for research paper analysis using AutoGen v0.7.1"""
    
    def __init__(self):
        self.logger = AgentLogger("sequential_workflow_v071")
        self.model_client = Config.get_model_client()
        self.agents = self._create_agents()
        self.workflow_state = None
    
    def _create_agents(self) -> Dict[str, AssistantAgent]:
        """Create specialized agents for each workflow step"""
        
        # Step 1: Research Agent
        researcher = AssistantAgent(
            name="researcher",
            model_client=self.model_client,
            system_message="""You are a research specialist. Your job is to:
1. Gather comprehensive information about the given topic
2. Identify key concepts, trends, and important sources
3. Structure your findings in a clear, organized format
4. Provide a foundation for deeper analysis

Focus on factual information and cite your reasoning process."""
        )
        
        # Step 2: Analyst Agent
        analyst = AssistantAgent(
            name="analyst", 
            model_client=self.model_client,
            system_message="""You are an analytical specialist. Your job is to:
1. Take the research findings and analyze patterns and relationships
2. Identify strengths, weaknesses, gaps, and opportunities
3. Draw insights and conclusions from the data
4. Structure your analysis for clear presentation

Be thorough but concise in your analysis."""
        )
        
        # Step 3: Writer Agent
        writer = AssistantAgent(
            name="writer",
            model_client=self.model_client,
            system_message="""You are a technical writer. Your job is to:
1. Transform research and analysis into a coherent, well-structured report
2. Ensure clarity, flow, and logical organization
3. Include executive summary, main findings, and recommendations
4. Write for both technical and non-technical audiences

Focus on clear communication and actionable insights."""
        )
        
        # Step 4: Reviewer Agent
        reviewer = AssistantAgent(
            name="reviewer",
            model_client=self.model_client,
            system_message="""You are a quality reviewer. Your job is to:
1. Review the final report for accuracy, completeness, and clarity
2. Identify any gaps, inconsistencies, or areas for improvement
3. Provide specific, actionable feedback
4. Rate the overall quality and make final recommendations

Be constructive and thorough in your review."""
        )
        
        return {
            "researcher": researcher,
            "analyst": analyst,
            "writer": writer,
            "reviewer": reviewer
        }
    
    async def execute_workflow(self, research_topic: str, task_id: str = None) -> Dict[str, Any]:
        """Execute the complete sequential workflow"""
        
        if not task_id:
            task_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.workflow_state = SequentialWorkflowState(task_id)
        self.logger.log_decision("coordinator", f"Starting workflow for: {research_topic}", f"Task ID: {task_id}")
        
        try:
            # Step 1: Research Phase
            research_results = await self._execute_research_step(research_topic)
            
            # Step 2: Analysis Phase
            analysis_results = await self._execute_analysis_step(research_results)
            
            # Step 3: Writing Phase
            writing_results = await self._execute_writing_step(research_results, analysis_results)
            
            # Step 4: Review Phase
            final_results = await self._execute_review_step(writing_results)
            
            # Compile final output
            final_output = {
                "task_id": task_id,
                "topic": research_topic,
                "research": research_results,
                "analysis": analysis_results,
                "report": writing_results,
                "review": final_results,
                "completed_at": datetime.now().isoformat(),
                "workflow_state": self.workflow_state.__dict__
            }
            
            self.logger.save_conversation(f"logs/workflow_{task_id}.json")
            return final_output
            
        except Exception as e:
            self.logger.logger.error(f"Workflow failed at step {self.workflow_state.current_step}: {e}")
            raise
    
    async def _execute_research_step(self, topic: str) -> str:
        """Step 1: Research phase"""
        self.logger.logger.info("=== Step 1: Research Phase ===")
        self.workflow_state.current_step = 1
        
        prompt = f"""Please research the topic: "{topic}"

Provide a comprehensive overview including:
1. Key concepts and definitions
2. Current state and recent developments
3. Main challenges and opportunities
4. Important sources and references
5. Structured summary for analysis

Focus on gathering factual, actionable information."""
        
        # Execute research using v0.7.1 async pattern
        from autogen_core import CancellationToken
        message = TextMessage(content=prompt, source="user")
        cancellation_token = CancellationToken()
        response = await self.agents["researcher"].on_messages([message], cancellation_token)
        
        research_output = response.chat_message.content
        
        # Checkpoint the results
        self.workflow_state.checkpoint("research", {"output": research_output, "topic": topic})
        self.workflow_state.steps_completed.append("research")
        
        self.logger.log_message("coordinator", "researcher", f"Research completed for: {topic}", "workflow")
        
        return research_output
    
    async def _execute_analysis_step(self, research_data: str) -> str:
        """Step 2: Analysis phase"""
        self.logger.logger.info("=== Step 2: Analysis Phase ===")
        self.workflow_state.current_step = 2
        
        prompt = f"""Please analyze the following research data:

{research_data}

Provide a thorough analysis including:
1. Key patterns and relationships identified
2. Strengths and weaknesses of current approaches
3. Gaps in knowledge or implementation
4. Opportunities for improvement or innovation
5. Strategic insights and implications

Structure your analysis clearly with headings and bullet points."""
        
        # Execute analysis
        from autogen_core import CancellationToken
        message = TextMessage(content=prompt, source="user")
        cancellation_token = CancellationToken()
        response = await self.agents["analyst"].on_messages([message], cancellation_token)
        
        analysis_output = response.chat_message.content
        
        # Checkpoint the results
        self.workflow_state.checkpoint("analysis", {"output": analysis_output})
        self.workflow_state.steps_completed.append("analysis")
        
        self.logger.log_message("analyst", "coordinator", "Analysis completed", "workflow")
        
        return analysis_output
    
    async def _execute_writing_step(self, research_data: str, analysis_data: str) -> str:
        """Step 3: Writing phase"""
        self.logger.logger.info("=== Step 3: Writing Phase ===")
        self.workflow_state.current_step = 3
        
        prompt = f"""Please write a comprehensive report based on the following research and analysis:

RESEARCH DATA:
{research_data}

ANALYSIS DATA:
{analysis_data}

Create a well-structured report with:
1. Executive Summary
2. Introduction and Background
3. Key Findings
4. Analysis and Insights
5. Recommendations
6. Conclusion

Write clearly and professionally for both technical and business audiences."""
        
        # Execute writing
        from autogen_core import CancellationToken
        message = TextMessage(content=prompt, source="user")
        cancellation_token = CancellationToken()
        response = await self.agents["writer"].on_messages([message], cancellation_token)
        
        writing_output = response.chat_message.content
        
        # Checkpoint the results
        self.workflow_state.checkpoint("writing", {"output": writing_output})
        self.workflow_state.steps_completed.append("writing")
        
        self.logger.log_message("writer", "coordinator", "Report writing completed", "workflow")
        
        return writing_output
    
    async def _execute_review_step(self, report: str) -> str:
        """Step 4: Review phase"""
        self.logger.logger.info("=== Step 4: Review Phase ===")
        self.workflow_state.current_step = 4
        
        prompt = f"""Please review the following report for quality, accuracy, and completeness:

{report}

Provide a comprehensive review including:
1. Overall quality assessment (1-10 scale)
2. Strengths of the report
3. Areas for improvement
4. Factual accuracy check
5. Clarity and organization assessment
6. Final recommendations
7. Summary of key takeaways

Be specific and constructive in your feedback."""
        
        # Execute review
        from autogen_core import CancellationToken
        message = TextMessage(content=prompt, source="user")
        cancellation_token = CancellationToken()
        response = await self.agents["reviewer"].on_messages([message], cancellation_token)
        
        review_output = response.chat_message.content
        
        # Checkpoint the results
        self.workflow_state.checkpoint("review", {"output": review_output})
        self.workflow_state.steps_completed.append("review")
        
        self.logger.log_message("reviewer", "coordinator", "Review completed", "workflow")
        
        return review_output

async def main():
    """Example usage of sequential workflow"""
    if not Config.validate_config():
        print("Please configure your API key in .env file")
        return
    
    # Create workflow
    workflow = SequentialResearchWorkflow()
    
    # Example topic
    topic = "Multi-Agent AI Systems in Production: Current Challenges and Best Practices"
    
    print(f"🚀 Starting Sequential Workflow (AutoGen v0.7.1) for topic: {topic}")
    print("This will execute 4 sequential steps: Research → Analysis → Writing → Review")
    print("Check logs/ directory for detailed progress and checkpoints\n")
    
    try:
        results = await workflow.execute_workflow(topic)
        
        print("✅ Sequential workflow completed successfully!")
        print(f"Task ID: {results['task_id']}")
        print(f"Steps completed: {', '.join(results['workflow_state']['steps_completed'])}")
        
        # Save final results
        Path("outputs").mkdir(exist_ok=True)
        output_file = f"outputs/sequential_workflow_v071_{results['task_id']}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Full results saved to: {output_file}")
        
        # Display key sections
        print("\n=== EXECUTIVE SUMMARY ===")
        print(results['report'][:500] + "..." if len(results['report']) > 500 else results['report'])
        
        print(f"\n=== QUALITY REVIEW ===")
        print(results['review'][:500] + "..." if len(results['review']) > 500 else results['review'])
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        print("Check logs for detailed error information")

if __name__ == "__main__":
    asyncio.run(main())