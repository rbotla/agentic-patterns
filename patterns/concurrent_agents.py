#!/usr/bin/env python3
"""
Concurrent Agents Pattern Implementation - AutoGen v0.7.1

This example demonstrates parallel analysis of a business scenario from multiple perspectives:
- Market Analyst: Analyzes market conditions and competition
- Technical Analyst: Evaluates technical feasibility and requirements
- Financial Analyst: Assesses financial implications and ROI
- Risk Analyst: Identifies potential risks and mitigation strategies

Key Features (v0.7.1):
- Async-first concurrent execution
- Modern agent creation with model clients
- Result aggregation and synthesis
- Performance comparison vs sequential execution
"""

import sys
import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from config import Config
from utils.logging_utils import AgentLogger

class ConcurrentAnalysisResult:
    """Container for concurrent analysis results"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.results = {}
        self.timing_data = {}
        self.errors = {}
        self.completed_at = None
        self.total_execution_time = 0
    
    def add_result(self, agent_name: str, result: str, execution_time: float):
        """Add result from an agent"""
        self.results[agent_name] = result
        self.timing_data[agent_name] = execution_time
    
    def add_error(self, agent_name: str, error: Exception):
        """Record an error from an agent"""
        self.errors[agent_name] = str(error)
    
    def is_complete(self, expected_agents: List[str]) -> bool:
        """Check if all expected agents have reported"""
        return all(agent in self.results for agent in expected_agents if agent not in self.errors)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        return {
            "task_id": self.task_id,
            "completed_agents": len(self.results),
            "failed_agents": len(self.errors),
            "total_execution_time": self.total_execution_time,
            "average_execution_time": sum(self.timing_data.values()) / len(self.timing_data) if self.timing_data else 0,
            "fastest_agent": min(self.timing_data.items(), key=lambda x: x[1]) if self.timing_data else None,
            "slowest_agent": max(self.timing_data.items(), key=lambda x: x[1]) if self.timing_data else None,
        }

class ConcurrentBusinessAnalysis:
    """Concurrent analysis system for business scenarios using AutoGen v0.7.1"""
    
    def __init__(self):
        self.logger = AgentLogger("concurrent_analysis_v071")
        self.model_client = Config.get_model_client()
        self.agents = self._create_specialist_agents()
        self.synthesizer = self._create_synthesizer_agent()
    
    def _create_specialist_agents(self) -> Dict[str, AssistantAgent]:
        """Create specialized analysis agents"""
        
        # Market Analysis Specialist
        market_analyst = AssistantAgent(
            name="market_analyst",
            model_client=self.model_client,
            system_message="""You are a market analysis specialist. For any business scenario, provide:

1. Market Size and Opportunity Assessment
2. Competitive Landscape Analysis
3. Target Customer Segmentation
4. Market Trends and Dynamics
5. Go-to-Market Strategy Recommendations
6. Market Entry Barriers and Advantages

Be data-driven and specific in your analysis. Focus on actionable market insights."""
        )
        
        # Technical Analysis Specialist
        technical_analyst = AssistantAgent(
            name="technical_analyst",
            model_client=self.model_client,
            system_message="""You are a technical feasibility specialist. For any business scenario, provide:

1. Technical Architecture Requirements
2. Technology Stack Recommendations
3. Development Timeline and Milestones
4. Scalability and Performance Considerations
5. Integration Requirements and Challenges
6. Technical Risk Assessment

Focus on practical implementation details and technical trade-offs."""
        )
        
        # Financial Analysis Specialist
        financial_analyst = AssistantAgent(
            name="financial_analyst",
            model_client=self.model_client,
            system_message="""You are a financial analysis specialist. For any business scenario, provide:

1. Revenue Model and Projections
2. Cost Structure Analysis
3. Investment Requirements and Timeline
4. ROI and Break-even Analysis
5. Financial Risk Assessment
6. Funding Strategy Recommendations

Provide concrete numbers and financial modeling where possible."""
        )
        
        # Risk Analysis Specialist
        risk_analyst = AssistantAgent(
            name="risk_analyst",
            model_client=self.model_client,
            system_message="""You are a risk management specialist. For any business scenario, provide:

1. Business Risk Identification and Assessment
2. Operational Risk Analysis
3. Regulatory and Compliance Considerations
4. Market Risk Factors
5. Risk Mitigation Strategies
6. Contingency Planning Recommendations

Focus on identifying potential failure points and mitigation strategies."""
        )
        
        return {
            "market_analyst": market_analyst,
            "technical_analyst": technical_analyst,
            "financial_analyst": financial_analyst,
            "risk_analyst": risk_analyst
        }
    
    def _create_synthesizer_agent(self) -> AssistantAgent:
        """Create an agent to synthesize concurrent results"""
        return AssistantAgent(
            name="synthesis_agent",
            model_client=self.model_client,
            system_message="""You are a strategic synthesis specialist. Your role is to:

1. Integrate analysis from multiple specialists (market, technical, financial, risk)
2. Identify synergies and conflicts between different perspectives
3. Prioritize recommendations based on overall impact
4. Create a cohesive strategic roadmap
5. Highlight critical success factors
6. Provide executive-level summary and next steps

Focus on creating actionable, integrated insights from diverse analytical perspectives."""
        )
    
    async def _analyze_with_agent(self, agent_name: str, agent: AssistantAgent, 
                                 scenario: str, context: Dict[str, Any]) -> Tuple[str, str, float]:
        """Execute analysis with a single agent (async)"""
        start_time = time.time()
        
        try:
            # Construct detailed prompt with context
            prompt = f"""Please analyze the following business scenario from your specialty perspective:

SCENARIO: {scenario}

ADDITIONAL CONTEXT:
- Business Domain: {context.get('domain', 'Not specified')}
- Timeline: {context.get('timeline', 'Not specified')}
- Budget Range: {context.get('budget', 'Not specified')}
- Key Constraints: {context.get('constraints', 'None specified')}

Provide a comprehensive analysis following your specialty guidelines. Be specific and actionable."""
            
            # Execute analysis using v0.7.1 async pattern
            from autogen_core import CancellationToken
            message = TextMessage(content=prompt, source="user")
            cancellation_token = CancellationToken()
            response = await agent.on_messages([message], cancellation_token)
            analysis_output = response.chat_message.content
            
            execution_time = time.time() - start_time
            
            self.logger.log_message("coordinator", agent_name, f"Analysis completed in {execution_time:.2f}s", "concurrent")
            
            return agent_name, analysis_output, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.logger.error(f"Agent {agent_name} failed after {execution_time:.2f}s: {e}")
            raise e
    
    async def execute_concurrent_analysis(self, business_scenario: str, context: Dict[str, Any] = None) -> ConcurrentAnalysisResult:
        """Execute concurrent analysis with multiple specialist agents"""
        
        if context is None:
            context = {}
        
        task_id = f"concurrent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result_container = ConcurrentAnalysisResult(task_id)
        
        self.logger.log_decision("coordinator", "Starting concurrent analysis", 
                                f"Scenario: {business_scenario[:100]}...")
        
        start_time = time.time()
        
        # Execute all agents concurrently using asyncio.gather
        try:
            # Create tasks for all agents
            tasks = [
                self._analyze_with_agent(agent_name, agent, business_scenario, context)
                for agent_name, agent in self.agents.items()
            ]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    # Handle exceptions
                    self.logger.logger.error(f"Agent failed: {result}")
                    result_container.add_error("unknown_agent", result)
                else:
                    agent_name, analysis, execution_time = result
                    result_container.add_result(agent_name, analysis, execution_time)
                    self.logger.logger.info(f"✅ {agent_name} completed in {execution_time:.2f}s")
                    
        except Exception as e:
            self.logger.logger.error(f"Concurrent execution failed: {e}")
            raise
        
        result_container.total_execution_time = time.time() - start_time
        result_container.completed_at = datetime.now().isoformat()
        
        self.logger.logger.info(f"Concurrent analysis completed in {result_container.total_execution_time:.2f}s")
        
        return result_container
    
    async def synthesize_results(self, analysis_results: ConcurrentAnalysisResult) -> str:
        """Synthesize concurrent analysis results into unified insights"""
        
        self.logger.logger.info("=== Synthesizing Concurrent Results ===")
        
        # Prepare synthesis prompt
        synthesis_prompt = f"""Please synthesize the following specialist analyses into a unified strategic assessment:

MARKET ANALYSIS:
{analysis_results.results.get('market_analyst', 'Not available')}

TECHNICAL ANALYSIS:
{analysis_results.results.get('technical_analyst', 'Not available')}

FINANCIAL ANALYSIS:
{analysis_results.results.get('financial_analyst', 'Not available')}

RISK ANALYSIS:
{analysis_results.results.get('risk_analyst', 'Not available')}

EXECUTION SUMMARY:
- Total Analysis Time: {analysis_results.total_execution_time:.2f} seconds
- Completed Analyses: {len(analysis_results.results)}/4
- Failed Analyses: {len(analysis_results.errors)}

Please provide:
1. Executive Summary integrating all perspectives
2. Key Strategic Insights and Recommendations
3. Critical Success Factors
4. Priority Action Items
5. Risk-Adjusted Implementation Roadmap
6. Resource Requirements Summary
7. Success Metrics and KPIs

Focus on creating a cohesive strategy that balances all analytical perspectives."""
        
        # Execute synthesis
        from autogen_core import CancellationToken
        message = TextMessage(content=synthesis_prompt, source="user")
        cancellation_token = CancellationToken()
        response = await self.synthesizer.on_messages([message], cancellation_token)
        synthesis_output = response.chat_message.content
        
        self.logger.log_message("synthesizer", "coordinator", "Synthesis completed", "concurrent")
        
        return synthesis_output
    
    async def run_full_analysis(self, business_scenario: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run complete concurrent analysis with synthesis"""
        
        # Step 1: Execute concurrent analysis
        concurrent_results = await self.execute_concurrent_analysis(business_scenario, context)
        
        # Step 2: Synthesize results
        synthesis = await self.synthesize_results(concurrent_results)
        
        # Step 3: Compile final output
        final_results = {
            "task_id": concurrent_results.task_id,
            "scenario": business_scenario,
            "context": context or {},
            "individual_analyses": concurrent_results.results,
            "synthesis": synthesis,
            "execution_summary": concurrent_results.get_summary(),
            "timing_data": concurrent_results.timing_data,
            "errors": concurrent_results.errors,
            "completed_at": concurrent_results.completed_at
        }
        
        return final_results

async def compare_concurrent_vs_sequential(business_scenario: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Compare concurrent vs sequential execution performance"""
    
    concurrent_system = ConcurrentBusinessAnalysis()
    
    print("🔄 Running performance comparison...")
    
    # Test concurrent execution
    print("⚡ Testing concurrent execution...")
    concurrent_start = time.time()
    concurrent_results = await concurrent_system.execute_concurrent_analysis(business_scenario, context)
    concurrent_time = time.time() - concurrent_start
    
    # Simulate sequential execution time (sum of individual times)
    sequential_time_estimate = sum(concurrent_results.timing_data.values())
    
    # Calculate performance improvement
    speedup = sequential_time_estimate / concurrent_time if concurrent_time > 0 else 0
    efficiency = (speedup / len(concurrent_results.results)) * 100 if concurrent_results.results else 0
    
    comparison = {
        "concurrent_execution_time": concurrent_time,
        "estimated_sequential_time": sequential_time_estimate,
        "speedup_factor": speedup,
        "efficiency_percentage": efficiency,
        "parallel_agents": len(concurrent_results.results),
        "failed_agents": len(concurrent_results.errors)
    }
    
    return comparison

async def main():
    """Example usage of concurrent agents pattern"""
    if not Config.validate_config():
        print("Please configure your API key in .env file")
        return
    
    # Example business scenario
    business_scenario = """
    Parents want their children to spend less time on social media and more time on educational content.
    Develop a B2B SaaS solution that provides a platform that can help reduce social media usage and increase motivation for educational activities.
    The platform should include features like gamification, progress tracking, and integration with existing educational tools.
    Target market: Schools and educational institutions.
    Key requirements: Must be user-friendly, scalable, and support multiple languages.
    """
    
    context = {
        "Target market": "Schools and educational institutions.",
        "Key requirements": "Must be user-friendly, scalable, and support multiple languages.",
        "Budget": "$2-5 million, with an 18-month timeline to market.",
        "Constraints": "Must integrate with Salesforce, HubSpot, and Zendesk. GDPR compliance required.",
        "domain": "EdTech",
        "timeline": "18 months"
    }
    
    print("🚀 Starting Concurrent Business Analysis (AutoGen v0.7.1)")
    print("This will run 4 specialist agents in parallel: Market, Technical, Financial, and Risk Analysis")
    print("Check logs/ directory for detailed progress\n")
    
    try:
        # Create analysis system
        analysis_system = ConcurrentBusinessAnalysis()
        
        # Run performance comparison
        performance_comparison = await compare_concurrent_vs_sequential(business_scenario, context)
        
        print(f"⚡ Performance Results:")
        print(f"   Concurrent execution: {performance_comparison['concurrent_execution_time']:.2f}s")
        print(f"   Estimated sequential: {performance_comparison['estimated_sequential_time']:.2f}s")
        print(f"   Speedup: {performance_comparison['speedup_factor']:.2f}x")
        print(f"   Efficiency: {performance_comparison['efficiency_percentage']:.1f}%\n")
        
        # Run full analysis
        results = await analysis_system.run_full_analysis(business_scenario, context)
        
        print("✅ Concurrent analysis completed successfully!")
        print(f"Task ID: {results['task_id']}")
        print(f"Execution Summary: {results['execution_summary']}")
        
        # Save results
        Path("outputs").mkdir(exist_ok=True)
        output_file = f"outputs/concurrent_analysis_v071_{results['task_id']}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Full results saved to: {output_file}")
        
        # Display synthesis summary
        print("\n=== STRATEGIC SYNTHESIS ===")
        synthesis_preview = results['synthesis'][:800] + "..." if len(results['synthesis']) > 800 else results['synthesis']
        print(synthesis_preview)
        
        # Display individual analysis previews
        print(f"\n=== INDIVIDUAL ANALYSES ({len(results['individual_analyses'])} completed) ===")
        for agent_name, analysis in results['individual_analyses'].items():
            print(f"\n{agent_name.upper()}:")
            preview = analysis[:300] + "..." if len(analysis) > 300 else analysis
            print(preview)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("Check logs for detailed error information")

if __name__ == "__main__":
    asyncio.run(main())