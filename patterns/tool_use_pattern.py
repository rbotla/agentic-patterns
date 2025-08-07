"""
Tool Use Pattern Implementation
Agents equipped with external tools for enhanced capabilities
"""

import autogen
from typing import Dict, Any, Optional, List, Callable
import logging
from utils.logging_utils import setup_logging
import json
import time
import random
import math

logger = setup_logging(__name__)


class ToolRegistry:
    """Registry for managing tool definitions and implementations"""
    
    def __init__(self):
        self.tools = {}
        self.tool_descriptions = {}
        self.tool_usage_stats = {}
    
    def register_tool(
        self, 
        name: str, 
        func: Callable,
        description: str,
        parameters: Dict[str, Any]
    ):
        """Register a new tool"""
        self.tools[name] = func
        self.tool_descriptions[name] = {
            "description": description,
            "parameters": parameters
        }
        self.tool_usage_stats[name] = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0
        }
        logger.info(f"Registered tool: {name}")
    
    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a registered tool"""
        if name not in self.tools:
            return {"error": f"Tool {name} not found"}
        
        start_time = time.time()
        self.tool_usage_stats[name]["calls"] += 1
        
        try:
            result = self.tools[name](**kwargs)
            self.tool_usage_stats[name]["successes"] += 1
            status = "success"
        except Exception as e:
            result = str(e)
            self.tool_usage_stats[name]["failures"] += 1
            status = "error"
            logger.error(f"Tool {name} failed: {e}")
        
        elapsed = time.time() - start_time
        self.tool_usage_stats[name]["total_time"] += elapsed
        
        return {
            "tool": name,
            "status": status,
            "result": result,
            "execution_time": elapsed
        }
    
    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools"""
        descriptions = []
        for name, info in self.tool_descriptions.items():
            desc = f"- {name}: {info['description']}\n"
            desc += f"  Parameters: {json.dumps(info['parameters'], indent=2)}"
            descriptions.append(desc)
        return "\n".join(descriptions)


class ToolUseWorkflow:
    """Main workflow for tool-using agents"""
    
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
        
        self.tool_registry = ToolRegistry()
        self._register_default_tools()
        self._initialize_agents()
    
    def _register_default_tools(self):
        """Register default tools"""
        
        self.tool_registry.register_tool(
            name="calculator",
            func=self._calculator_tool,
            description="Perform mathematical calculations",
            parameters={
                "expression": "Mathematical expression to evaluate (e.g., '2 + 2 * 3')"
            }
        )
        
        self.tool_registry.register_tool(
            name="web_search",
            func=self._web_search_tool,
            description="Search the web for information",
            parameters={
                "query": "Search query string",
                "max_results": "Maximum number of results (default: 5)"
            }
        )
        
        self.tool_registry.register_tool(
            name="data_analyzer",
            func=self._data_analyzer_tool,
            description="Analyze data and compute statistics",
            parameters={
                "data": "List of numerical values",
                "operation": "Type of analysis (mean, median, std, min, max)"
            }
        )
        
        self.tool_registry.register_tool(
            name="text_processor",
            func=self._text_processor_tool,
            description="Process and analyze text",
            parameters={
                "text": "Text to process",
                "operation": "Operation to perform (word_count, char_count, summary)"
            }
        )
        
        self.tool_registry.register_tool(
            name="code_executor",
            func=self._code_executor_tool,
            description="Execute Python code snippets",
            parameters={
                "code": "Python code to execute"
            }
        )
    
    def _initialize_agents(self):
        """Initialize tool-using agents"""
        
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        
        self.planner_agent = autogen.AssistantAgent(
            name="planner",
            llm_config=self.llm_config,
            system_message=f"""You are a planning agent that analyzes tasks and determines 
            which tools to use. Available tools:
            
{tool_descriptions}

For each task, create a plan specifying:
1. Which tools to use
2. In what order
3. What parameters to pass
4. How to combine results

Format your response as:
PLAN:
Step 1: [tool_name] with [parameters]
Step 2: [tool_name] with [parameters]
..."""
        )
        
        self.executor_agent = autogen.AssistantAgent(
            name="executor",
            llm_config=self.llm_config,
            system_message="""You are an execution agent that carries out tool operations
            based on plans. Parse tool calls, execute them, and handle results."""
        )
        
        self.synthesizer_agent = autogen.AssistantAgent(
            name="synthesizer",
            llm_config=self.llm_config,
            system_message="""You are a synthesis agent that combines tool outputs
            into coherent final responses. Interpret results and provide clear answers."""
        )
    
    def _calculator_tool(self, expression: str) -> float:
        """Simple calculator tool"""
        try:
            result = eval(expression, {"__builtins__": {}}, 
                        {"sin": math.sin, "cos": math.cos, "sqrt": math.sqrt, 
                         "pi": math.pi, "e": math.e})
            return result
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")
    
    def _web_search_tool(self, query: str, max_results: int = 5) -> List[Dict]:
        """Simulated web search tool"""
        results = []
        for i in range(max_results):
            results.append({
                "title": f"Result {i+1} for '{query}'",
                "snippet": f"This is a simulated search result about {query}...",
                "url": f"https://example.com/result{i+1}"
            })
        return results
    
    def _data_analyzer_tool(self, data: List[float], operation: str) -> float:
        """Data analysis tool"""
        if not data:
            raise ValueError("Empty data list")
        
        operations = {
            "mean": lambda d: sum(d) / len(d),
            "median": lambda d: sorted(d)[len(d)//2],
            "std": lambda d: math.sqrt(sum((x - sum(d)/len(d))**2 for x in d) / len(d)),
            "min": min,
            "max": max
        }
        
        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")
        
        return operations[operation](data)
    
    def _text_processor_tool(self, text: str, operation: str) -> Any:
        """Text processing tool"""
        operations = {
            "word_count": lambda t: len(t.split()),
            "char_count": lambda t: len(t),
            "summary": lambda t: t[:100] + "..." if len(t) > 100 else t
        }
        
        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")
        
        return operations[operation](text)
    
    def _code_executor_tool(self, code: str) -> Any:
        """Simulated code execution tool"""
        logger.warning("Code execution is simulated for safety")
        return f"[Simulated execution of: {code[:50]}...]"
    
    def execute_with_tools(self, task: str) -> Dict[str, Any]:
        """Execute a task using available tools"""
        logger.info(f"Executing task with tools: {task[:100]}...")
        start_time = time.time()
        
        plan = self._create_plan(task)
        logger.info(f"Created plan: {plan}")
        
        tool_results = self._execute_plan(plan)
        logger.info(f"Executed {len(tool_results)} tool calls")
        
        final_result = self._synthesize_results(task, tool_results)
        
        elapsed_time = time.time() - start_time
        
        return {
            "task": task,
            "plan": plan,
            "tool_results": tool_results,
            "final_answer": final_result,
            "execution_time": elapsed_time,
            "tool_usage_stats": self.tool_registry.tool_usage_stats
        }
    
    def _create_plan(self, task: str) -> List[Dict]:
        """Create execution plan for task"""
        message = f"Create a tool execution plan for this task: {task}"
        
        response = self.planner_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        plan = self._parse_plan(response)
        return plan
    
    def _parse_plan(self, plan_text: str) -> List[Dict]:
        """Parse plan text into structured format"""
        steps = []
        
        lines = plan_text.split('\n')
        for line in lines:
            if 'Step' in line or 'step' in line:
                step = {
                    "description": line,
                    "tool": None,
                    "parameters": {}
                }
                
                for tool_name in self.tool_registry.tools.keys():
                    if tool_name in line.lower():
                        step["tool"] = tool_name
                        break
                
                if step["tool"]:
                    steps.append(step)
        
        if not steps:
            steps = [{"description": "No specific tools needed", "tool": None, "parameters": {}}]
        
        return steps
    
    def _execute_plan(self, plan: List[Dict]) -> List[Dict]:
        """Execute the planned tool calls"""
        results = []
        
        for step in plan:
            if step["tool"]:
                result = self.tool_registry.execute_tool(
                    step["tool"],
                    **step["parameters"]
                )
                results.append(result)
                logger.info(f"Executed tool: {step['tool']} - Status: {result['status']}")
        
        return results
    
    def _synthesize_results(self, task: str, tool_results: List[Dict]) -> str:
        """Synthesize tool results into final answer"""
        message = f"""Task: {task}

Tool Results:
{json.dumps(tool_results, indent=2)}

Synthesize these results into a clear, comprehensive answer to the task."""
        
        response = self.synthesizer_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        return response


class ReActAgent(ToolUseWorkflow):
    """
    ReAct (Reasoning + Acting) pattern implementation
    Interleaves reasoning and tool use
    """
    
    def __init__(self, llm_config: Optional[Dict] = None):
        super().__init__(llm_config)
        self.reasoning_traces = []
    
    def execute_with_react(self, task: str, max_steps: int = 5) -> Dict[str, Any]:
        """Execute task using ReAct pattern"""
        logger.info(f"Starting ReAct execution for: {task[:100]}...")
        
        current_context = {"task": task, "observations": []}
        steps_taken = []
        
        for step_num in range(max_steps):
            logger.info(f"\n=== ReAct Step {step_num + 1} ===")
            
            thought = self._think(current_context)
            self.reasoning_traces.append({"step": step_num + 1, "thought": thought})
            logger.info(f"Thought: {thought}")
            
            if "final answer" in thought.lower() or "complete" in thought.lower():
                break
            
            action = self._decide_action(thought)
            logger.info(f"Action: {action}")
            
            if action and action.get("tool"):
                observation = self.tool_registry.execute_tool(
                    action["tool"],
                    **action.get("parameters", {})
                )
                current_context["observations"].append(observation)
                logger.info(f"Observation: {observation['result']}")
            
            steps_taken.append({
                "step": step_num + 1,
                "thought": thought,
                "action": action,
                "observation": current_context["observations"][-1] if current_context["observations"] else None
            })
        
        final_answer = self._generate_final_answer(current_context)
        
        return {
            "task": task,
            "steps": steps_taken,
            "reasoning_traces": self.reasoning_traces,
            "final_answer": final_answer
        }
    
    def _think(self, context: Dict) -> str:
        """Generate reasoning about current state"""
        prompt = f"""Task: {context['task']}

Previous observations: {json.dumps(context['observations'], indent=2)}

What should I think about or consider next? Reason step by step."""
        
        response = self.planner_agent.generate_reply(
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response
    
    def _decide_action(self, thought: str) -> Optional[Dict]:
        """Decide next action based on reasoning"""
        for tool_name in self.tool_registry.tools.keys():
            if tool_name in thought.lower():
                return {
                    "tool": tool_name,
                    "parameters": {}
                }
        
        return None
    
    def _generate_final_answer(self, context: Dict) -> str:
        """Generate final answer from context"""
        prompt = f"""Task: {context['task']}

Observations: {json.dumps(context['observations'], indent=2)}

Provide the final answer to the task based on all observations."""
        
        response = self.synthesizer_agent.generate_reply(
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response


def run_tool_use_examples():
    """Demonstrate Tool Use patterns"""
    logger.info("Starting Tool Use Pattern Examples")
    
    logger.info("\n=== Basic Tool Use Example ===")
    
    workflow = ToolUseWorkflow()
    
    task = "Calculate the average of [15, 23, 42, 8, 19] and then find its square root"
    
    result = workflow.execute_with_tools(task)
    
    logger.info(f"\nTask: {task}")
    logger.info(f"Final Answer: {result['final_answer']}")
    logger.info(f"Execution Time: {result['execution_time']:.2f}s")
    
    logger.info("\n=== Custom Tool Registration Example ===")
    
    def custom_fibonacci_tool(n: int) -> int:
        """Calculate nth Fibonacci number"""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    workflow.tool_registry.register_tool(
        name="fibonacci",
        func=custom_fibonacci_tool,
        description="Calculate nth Fibonacci number",
        parameters={"n": "Position in Fibonacci sequence"}
    )
    
    fib_task = "What is the 10th Fibonacci number?"
    fib_result = workflow.execute_with_tools(fib_task)
    
    logger.info(f"\nFibonacci Task: {fib_task}")
    logger.info(f"Result: {fib_result['final_answer']}")
    
    logger.info("\n=== ReAct Pattern Example ===")
    
    react_agent = ReActAgent()
    
    react_task = "Search for information about Python, then count the words in the first result"
    
    react_result = react_agent.execute_with_react(react_task, max_steps=3)
    
    logger.info(f"\nReAct Task: {react_task}")
    logger.info(f"Number of steps: {len(react_result['steps'])}")
    logger.info(f"Final Answer: {react_result['final_answer']}")
    
    logger.info("\n=== Tool Usage Statistics ===")
    
    for tool_name, stats in workflow.tool_registry.tool_usage_stats.items():
        if stats["calls"] > 0:
            avg_time = stats["total_time"] / stats["calls"]
            success_rate = stats["successes"] / stats["calls"] * 100
            logger.info(f"{tool_name}: {stats['calls']} calls, "
                       f"{success_rate:.1f}% success, {avg_time:.3f}s avg time")
    
    logger.info("\nTool Use Pattern Examples Complete")


if __name__ == "__main__":
    run_tool_use_examples()