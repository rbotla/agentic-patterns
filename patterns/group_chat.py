#!/usr/bin/env python3
"""
Group Chat Pattern Implementation - AutoGen v0.7.1

This example demonstrates collaborative problem-solving using AutoGen's v0.7.1 team architecture:
- Product Manager: Defines requirements and priorities
- Software Engineer: Provides technical implementation details  
- Data Scientist: Handles ML/AI aspects and data requirements
- DevOps Engineer: Addresses deployment and infrastructure
- QA Tester: Ensures quality and testing strategies

Key Features (v0.7.1):
- Modern team-based architecture (RoundRobinGroupChat/SelectorGroupChat)
- Async-first collaboration
- Advanced termination conditions
- Conversation flow control and analysis
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from config import Config
from utils.logging_utils import AgentLogger

class GroupChatWorkflow:
    """Group chat workflow for collaborative problem solving using AutoGen v0.7.1"""
    
    def __init__(self, max_rounds: int = 25):
        self.logger = AgentLogger("group_chat_v071")
        self.model_client = Config.get_model_client()
        self.max_rounds = max_rounds
        self.agents = self._create_team_agents()
    
    def _create_team_agents(self) -> Dict[str, AssistantAgent]:
        """Create a diverse team of specialist agents"""
        
        # Product Manager - Requirements and Strategy
        product_manager = AssistantAgent(
            name="product_manager",
            model_client=self.model_client,
            system_message="""You are a Product Manager with expertise in:
- Requirements gathering and prioritization
- User experience design
- Market analysis and competitive positioning
- Stakeholder communication
- Roadmap planning and feature prioritization

Focus on business value, user needs, and strategic alignment. Ask clarifying questions about requirements and suggest solutions that maximize user value while being technically feasible."""
        )
        
        # Software Engineer - Technical Implementation
        software_engineer = AssistantAgent(
            name="software_engineer",
            model_client=self.model_client,
            system_message="""You are a Senior Software Engineer with expertise in:
- System architecture and design patterns
- Full-stack development (frontend, backend, APIs)
- Performance optimization and scalability
- Code quality and best practices
- Technology selection and trade-offs

Provide technical insights, identify implementation challenges, suggest architectural solutions, and estimate development effort. Focus on building robust, maintainable systems."""
        )
        
        # Data Scientist - ML/AI and Analytics
        data_scientist = AssistantAgent(
            name="data_scientist",
            model_client=self.model_client,
            system_message="""You are a Data Scientist specializing in:
- Machine learning model design and implementation
- Data pipeline architecture and ETL processes
- Statistical analysis and experimentation
- Model deployment and monitoring
- Data quality and governance

Contribute insights on data requirements, ML opportunities, analytics strategies, and measurement frameworks. Focus on data-driven solutions and scientific rigor."""
        )
        
        # DevOps Engineer - Infrastructure and Deployment
        devops_engineer = AssistantAgent(
            name="devops_engineer", 
            model_client=self.model_client,
            system_message="""You are a DevOps Engineer with expertise in:
- Cloud infrastructure (AWS, Azure, GCP)
- CI/CD pipelines and automation
- Containerization and orchestration (Docker, Kubernetes)
- Monitoring, logging, and observability
- Security and compliance
- Cost optimization

Focus on deployment strategies, scalability, reliability, security, and operational excellence. Consider infrastructure costs and maintenance overhead."""
        )
        
        # QA Tester - Quality Assurance
        qa_tester = AssistantAgent(
            name="qa_tester",
            model_client=self.model_client,
            system_message="""You are a QA Engineer specializing in:
- Test strategy and planning
- Automated testing frameworks
- Performance and load testing
- Security testing
- User acceptance testing
- Bug tracking and quality metrics

Focus on ensuring quality, identifying edge cases, suggesting testing strategies, and preventing defects. Consider both functional and non-functional requirements."""
        )
        
        return {
            "product_manager": product_manager,
            "software_engineer": software_engineer,
            "data_scientist": data_scientist,
            "devops_engineer": devops_engineer,
            "qa_tester": qa_tester
        }
    
    def _create_round_robin_team(self, problem_statement: str) -> RoundRobinGroupChat:
        """Create a round-robin team for structured discussion"""
        
        # Create agent list for team
        team_agents = list(self.agents.values())
        
        # Create termination conditions
        max_messages_condition = MaxMessageTermination(max_messages=self.max_rounds)
        consensus_condition = TextMentionTermination("CONSENSUS_REACHED")
        
        # Create round-robin team
        team = RoundRobinGroupChat(
            participants=team_agents,
            termination_condition=max_messages_condition
        )
        
        return team
    
    def _create_selector_team(self, problem_statement: str) -> SelectorGroupChat:
        """Create a selector team for dynamic speaker selection"""
        
        # Create agent list for team
        team_agents = list(self.agents.values())
        
        # Create termination conditions
        max_messages_condition = MaxMessageTermination(max_messages=self.max_rounds)
        
        # Create selector team with model-based speaker selection
        team = SelectorGroupChat(
            participants=team_agents,
            model_client=self.model_client,
            termination_condition=max_messages_condition
        )
        
        return team
    
    async def facilitate_round_robin_discussion(self, problem_statement: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Facilitate a structured round-robin group discussion"""
        
        task_id = f"roundrobin_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.log_decision("facilitator", f"Starting round-robin discussion", f"Problem: {problem_statement[:100]}...")
        
        # Create round-robin team
        team = self._create_round_robin_team(problem_statement)
        
        # Prepare initial context message
        context_info = ""
        if context:
            context_info = f"""
            
ADDITIONAL CONTEXT:
- Business Domain: {context.get('domain', 'Not specified')}
- Timeline: {context.get('timeline', 'Flexible')}  
- Budget Constraints: {context.get('budget', 'To be determined')}
- Technical Constraints: {context.get('constraints', 'None specified')}
- Success Metrics: {context.get('metrics', 'To be defined')}"""
        
        initial_message = f"""Welcome team! Let's collaborate on solving this challenge:

PROBLEM STATEMENT: {problem_statement}{context_info}

We'll use a round-robin discussion format where each team member contributes their expertise in turn:

1. Product Manager - Business requirements and user needs
2. Software Engineer - Technical implementation approach
3. Data Scientist - Data and ML/AI opportunities  
4. DevOps Engineer - Infrastructure and deployment considerations
5. QA Tester - Quality assurance and testing strategy

Remember:
- Share your expertise and unique perspective
- Ask clarifying questions when needed
- Build on others' ideas constructively
- Focus on creating practical, implementable solutions

Let's begin with the Product Manager's perspective on requirements."""
        
        try:
            # Start the team discussion
            start_message = TextMessage(content=initial_message, source="facilitator")
            result = await team.run(task=start_message)
            
            # Extract conversation and analyze
            conversation_analysis = self._analyze_team_conversation(result.messages, task_id)
            
            # Compile results
            discussion_results = {
                "task_id": task_id,
                "team_type": "round_robin",
                "problem_statement": problem_statement,
                "context": context or {},
                "conversation_messages": [msg.dict() if hasattr(msg, 'dict') else str(msg) for msg in result.messages],
                "analysis": conversation_analysis,
                "participants": list(self.agents.keys()),
                "total_messages": len(result.messages),
                "completed_at": datetime.now().isoformat()
            }
            
            return discussion_results
            
        except Exception as e:
            self.logger.logger.error(f"Round-robin discussion failed: {e}")
            raise
    
    async def facilitate_selector_discussion(self, problem_statement: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Facilitate a dynamic selector-based group discussion"""
        
        task_id = f"selector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.log_decision("facilitator", f"Starting selector discussion", f"Problem: {problem_statement[:100]}...")
        
        # Create selector team
        team = self._create_selector_team(problem_statement)
        
        # Prepare initial context message
        context_info = ""
        if context:
            context_info = f"""
            
ADDITIONAL CONTEXT:
- Business Domain: {context.get('domain', 'Not specified')}
- Timeline: {context.get('timeline', 'Flexible')}  
- Budget Constraints: {context.get('budget', 'To be determined')}
- Technical Constraints: {context.get('constraints', 'None specified')}
- Success Metrics: {context.get('metrics', 'To be defined')}"""
        
        initial_message = f"""Welcome team! Let's collaborate on solving this challenge:

PROBLEM STATEMENT: {problem_statement}{context_info}

This will be a dynamic discussion where the next speaker is selected based on conversation context and expertise needs:

Team Members:
- Product Manager: Requirements, user needs, business strategy
- Software Engineer: Technical implementation, architecture
- Data Scientist: ML/AI, data analysis, experimentation
- DevOps Engineer: Infrastructure, deployment, operations  
- QA Tester: Quality assurance, testing, risk mitigation

The system will automatically select who should speak next based on the conversation flow and expertise requirements.

Let's begin the collaborative discussion!"""
        
        try:
            # Start the team discussion
            start_message = TextMessage(content=initial_message, source="facilitator")
            result = await team.run(task=start_message)
            
            # Extract conversation and analyze
            conversation_analysis = self._analyze_team_conversation(result.messages, task_id)
            
            # Compile results
            discussion_results = {
                "task_id": task_id,
                "team_type": "selector",
                "problem_statement": problem_statement,
                "context": context or {},
                "conversation_messages": [msg.dict() if hasattr(msg, 'dict') else str(msg) for msg in result.messages],
                "analysis": conversation_analysis,
                "participants": list(self.agents.keys()),
                "total_messages": len(result.messages),
                "completed_at": datetime.now().isoformat()
            }
            
            return discussion_results
            
        except Exception as e:
            self.logger.logger.error(f"Selector discussion failed: {e}")
            raise
    
    def _analyze_team_conversation(self, messages: List, task_id: str) -> Dict[str, Any]:
        """Analyze the team conversation for insights"""
        
        # Count contributions by participant
        contributions = {}
        decision_points = []
        consensus_indicators = []
        
        for msg in messages:
            # Extract speaker and content based on message structure
            if hasattr(msg, 'source'):
                speaker = msg.source
                content = str(msg.content) if hasattr(msg, 'content') else str(msg)
            else:
                # Fallback for different message formats
                speaker = "unknown"
                content = str(msg)
            
            # Count contributions
            if speaker in contributions:
                contributions[speaker] += 1
            else:
                contributions[speaker] = 1
            
            # Identify decision points
            if any(keyword in content.lower() for keyword in ["decide", "agree", "recommend", "propose"]):
                decision_points.append({
                    "speaker": speaker,
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "message_index": len(decision_points)
                })
            
            # Identify consensus building
            if any(keyword in content.lower() for keyword in ["consensus", "agree with", "support", "sounds good"]):
                consensus_indicators.append({
                    "speaker": speaker,
                    "content": content[:150] + "..." if len(content) > 150 else content
                })
        
        # Calculate participation balance
        total_contributions = sum(contributions.values())
        participation_balance = {
            name: (count / total_contributions * 100) if total_contributions > 0 else 0
            for name, count in contributions.items()
        }
        
        analysis = {
            "participation_stats": {
                "total_messages": len(messages),
                "unique_participants": len(contributions),
                "contributions_by_participant": contributions,
                "participation_balance": participation_balance
            },
            "conversation_dynamics": {
                "decision_points": len(decision_points),
                "consensus_indicators": len(consensus_indicators),
                "key_decisions": decision_points[:5],  # Top 5 decisions
                "consensus_moments": consensus_indicators[:3]  # Top 3 consensus moments
            },
            "collaboration_quality": {
                "balanced_participation": max(participation_balance.values()) - min(participation_balance.values()) < 40 if participation_balance else False,
                "constructive_dialogue": len(consensus_indicators) > len(decision_points) * 0.3,
                "productive_length": 10 <= len(messages) <= 50
            }
        }
        
        return analysis
    
    async def extract_action_items(self, discussion_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract action items from the team discussion"""
        
        # Create an analyzer agent to extract action items
        action_extractor = AssistantAgent(
            name="action_extractor",
            model_client=self.model_client,
            system_message="""You extract action items from team discussions. Analyze the conversation and identify:

1. Concrete tasks that need to be completed
2. Who should be responsible for each task
3. Dependencies between tasks
4. Estimated timelines
5. Success criteria

Format as a structured list with clear ownership and deadlines."""
        )
        
        # Prepare conversation summary for analysis
        conversation_text = "\n\n".join([
            f"Message: {msg}" for msg in discussion_results["conversation_messages"]
        ])
        
        extraction_prompt = f"""Please analyze the following team discussion and extract specific action items:

DISCUSSION TRANSCRIPT:
{conversation_text}

Extract and format action items with:
1. Task description
2. Responsible team member
3. Dependencies (if any)
4. Estimated timeline
5. Success criteria
6. Priority level

Focus on concrete, actionable tasks that were agreed upon during the discussion."""
        
        from autogen_core import CancellationToken
        message = TextMessage(content=extraction_prompt, source="coordinator")
        cancellation_token = CancellationToken()
        response = await action_extractor.on_messages([message], cancellation_token)
        action_items_text = response.chat_message.content
        
        return {
            "extracted_actions": action_items_text,
            "extraction_timestamp": datetime.now().isoformat()
        }

async def main():
    """Example usage of group chat pattern"""
    if not Config.validate_config():
        print("Please configure your API key in .env file")
        return
    
    # Example problem statement
    problem_statement = """
    Our company wants to build an AI-powered code review assistant that can:
    1. Automatically review pull requests for code quality, security vulnerabilities, and best practices
    2. Provide intelligent suggestions for code improvements
    3. Learn from the team's coding patterns and preferences over time
    4. Integrate with GitHub, GitLab, and Bitbucket
    5. Support multiple programming languages (Python, JavaScript, Java, Go)
    
    We need to determine the technical approach, architecture, implementation plan, and go-to-market strategy.
    """
    
    context = {
        "domain": "Developer Tools - AI Code Review",
        "timeline": "MVP in 6 months, full product in 12 months",
        "budget": "$500K-1M development budget",
        "constraints": "Must handle enterprise security requirements, GDPR compliance",
        "metrics": "Reduce code review time by 50%, catch 90% of common issues automatically"
    }
    
    print("🗣️  Starting Group Chat Discussion (AutoGen v0.7.1)")
    print("Participants: Product Manager, Software Engineer, Data Scientist, DevOps Engineer, QA Tester")
    print(f"Topic: AI-Powered Code Review Assistant")
    print("Check logs/ directory for detailed conversation tracking\n")
    
    try:
        # Create group chat workflow
        group_workflow = GroupChatWorkflow(max_rounds=20)
        
        # Test both discussion types
        print("🔄 Testing Round-Robin Discussion...")
        roundrobin_results = await group_workflow.facilitate_round_robin_discussion(problem_statement, context)
        
        print("🎯 Testing Selector-based Discussion...")
        selector_results = await group_workflow.facilitate_selector_discussion(problem_statement, context)
        
        # Display results for both
        for results, discussion_type in [(roundrobin_results, "Round-Robin"), (selector_results, "Selector")]:
            print(f"\n✅ {discussion_type} discussion completed successfully!")
            print(f"Task ID: {results['task_id']}")
            print(f"Total messages: {results['total_messages']}")
            print(f"Participants: {', '.join(results['participants'])}")
            
            # Display participation stats
            participation = results['analysis']['participation_stats']
            print(f"\n📊 {discussion_type} Participation Summary:")
            for participant, percentage in participation['participation_balance'].items():
                print(f"   {participant}: {percentage:.1f}% of messages")
            
            # Display collaboration quality
            quality = results['analysis']['collaboration_quality']
            print(f"\n🤝 {discussion_type} Collaboration Quality:")
            print(f"   Balanced participation: {'✅' if quality['balanced_participation'] else '❌'}")
            print(f"   Constructive dialogue: {'✅' if quality['constructive_dialogue'] else '❌'}")
            print(f"   Productive length: {'✅' if quality['productive_length'] else '❌'}")
        
        # Extract action items from selector discussion
        print(f"\n⚡ Extracting action items from selector discussion...")
        action_items = await group_workflow.extract_action_items(selector_results)
        
        # Save results
        Path("outputs").mkdir(exist_ok=True)
        
        # Save both discussion results
        for results, suffix in [(roundrobin_results, "roundrobin"), (selector_results, "selector")]:
            output_file = f"outputs/group_discussion_v071_{suffix}_{results['task_id']}.json"
            
            if suffix == "selector":
                results["action_items"] = action_items
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"📄 {suffix.title()} results saved to: {output_file}")
        
        # Display key insights
        print(f"\n📋 Action Items:")
        action_preview = action_items['extracted_actions'][:600] + "..." if len(action_items['extracted_actions']) > 600 else action_items['extracted_actions']
        print(action_preview)
        
    except Exception as e:
        print(f"❌ Group discussion failed: {e}")
        print("Check logs for detailed error information")

if __name__ == "__main__":
    asyncio.run(main())