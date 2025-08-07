"""
MCP (Model Context Protocol) Pattern Implementation
Standardized protocol for model context management and tool integration
Enables seamless communication between AI models and external tools/services
"""

import autogen
from typing import Dict, Any, Optional, List, Callable, Union
import logging
from utils.logging_utils import setup_logging
import time
import json
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from abc import ABC, abstractmethod

logger = setup_logging(__name__)


class MCPMessageType(Enum):
    """Types of MCP messages"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONTEXT_UPDATE = "context_update"
    CAPABILITY_QUERY = "capability_query"


class MCPCapability(Enum):
    """Standard MCP capabilities"""
    TOOLS = "tools"
    PROMPTS = "prompts"
    RESOURCES = "resources"
    SAMPLING = "sampling"
    ROOTS = "roots"
    LOGGING = "logging"


@dataclass
class MCPMessage:
    """Standard MCP message format"""
    id: str
    type: MCPMessageType
    method: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_json(self) -> str:
        """Convert to JSON-RPC format"""
        msg = {
            "jsonrpc": "2.0",
            "id": self.id,
            "type": self.type.value
        }
        
        if self.method:
            msg["method"] = self.method
        if self.params:
            msg["params"] = self.params
        if self.result is not None:
            msg["result"] = self.result
        if self.error:
            msg["error"] = self.error
        if self.metadata:
            msg["metadata"] = self.metadata
            
        return json.dumps(msg)
    
    @classmethod
    def from_json(cls, json_str: str) -> "MCPMessage":
        """Create from JSON-RPC format"""
        data = json.loads(json_str)
        return cls(
            id=data.get("id", ""),
            type=MCPMessageType(data.get("type", "request")),
            method=data.get("method"),
            params=data.get("params", {}),
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )


class MCPServer(ABC):
    """Abstract base class for MCP servers"""
    
    @abstractmethod
    async def handle_request(self, message: MCPMessage) -> MCPMessage:
        """Handle incoming MCP request"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[MCPCapability]:
        """Return server capabilities"""
        pass
    
    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """Return available tools"""
        pass


class MCPToolServer(MCPServer):
    """MCP server that provides tool access"""
    
    def __init__(self, name: str = "tool_server"):
        self.name = name
        self.tools = {}
        self.tool_descriptions = {}
        self.request_history = []
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools"""
        self.register_tool(
            "calculate",
            self._calculate_tool,
            {
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            }
        )
        
        self.register_tool(
            "search",
            self._search_tool,
            {
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        )
        
        self.register_tool(
            "read_file",
            self._read_file_tool,
            {
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"]
                }
            }
        )
    
    def register_tool(self, name: str, func: Callable, schema: Dict[str, Any]):
        """Register a new tool"""
        self.tools[name] = func
        self.tool_descriptions[name] = schema
        logger.info(f"Registered MCP tool: {name}")
    
    async def handle_request(self, message: MCPMessage) -> MCPMessage:
        """Handle incoming MCP request"""
        self.request_history.append(message)
        
        if message.method == "tools/list":
            return self._handle_list_tools(message)
        elif message.method == "tools/call":
            return await self._handle_tool_call(message)
        elif message.method == "capabilities":
            return self._handle_capabilities(message)
        else:
            return MCPMessage(
                id=message.id,
                type=MCPMessageType.ERROR,
                error={"code": -32601, "message": "Method not found"}
            )
    
    def _handle_list_tools(self, message: MCPMessage) -> MCPMessage:
        """Handle tools/list request"""
        tools_list = []
        for name, schema in self.tool_descriptions.items():
            tools_list.append({
                "name": name,
                "description": schema.get("description", ""),
                "inputSchema": schema.get("parameters", {})
            })
        
        return MCPMessage(
            id=message.id,
            type=MCPMessageType.RESPONSE,
            result={"tools": tools_list}
        )
    
    async def _handle_tool_call(self, message: MCPMessage) -> MCPMessage:
        """Handle tools/call request"""
        tool_name = message.params.get("name")
        tool_args = message.params.get("arguments", {})
        
        if tool_name not in self.tools:
            return MCPMessage(
                id=message.id,
                type=MCPMessageType.ERROR,
                error={"code": -32602, "message": f"Tool {tool_name} not found"}
            )
        
        try:
            if asyncio.iscoroutinefunction(self.tools[tool_name]):
                result = await self.tools[tool_name](**tool_args)
            else:
                result = self.tools[tool_name](**tool_args)
            
            return MCPMessage(
                id=message.id,
                type=MCPMessageType.TOOL_RESULT,
                result={"content": [{"type": "text", "text": str(result)}]}
            )
        except Exception as e:
            return MCPMessage(
                id=message.id,
                type=MCPMessageType.ERROR,
                error={"code": -32603, "message": str(e)}
            )
    
    def _handle_capabilities(self, message: MCPMessage) -> MCPMessage:
        """Handle capabilities request"""
        return MCPMessage(
            id=message.id,
            type=MCPMessageType.RESPONSE,
            result={
                "capabilities": {
                    "tools": {},
                    "prompts": {},
                    "resources": {}
                }
            }
        )
    
    def get_capabilities(self) -> List[MCPCapability]:
        """Return server capabilities"""
        return [MCPCapability.TOOLS, MCPCapability.LOGGING]
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Return available tools"""
        return list(self.tool_descriptions.values())
    
    def _calculate_tool(self, expression: str) -> float:
        """Calculate mathematical expression"""
        try:
            result = eval(expression, {"__builtins__": {}})
            return result
        except Exception as e:
            raise ValueError(f"Calculation error: {e}")
    
    def _search_tool(self, query: str, limit: int = 5) -> List[str]:
        """Simulated search tool"""
        results = []
        for i in range(limit):
            results.append(f"Result {i+1} for '{query}'")
        return results
    
    def _read_file_tool(self, path: str) -> str:
        """Simulated file reading"""
        return f"[Simulated content of {path}]"


class MCPClient:
    """MCP client for interacting with MCP servers"""
    
    def __init__(self, name: str = "mcp_client"):
        self.name = name
        self.servers: Dict[str, MCPServer] = {}
        self.message_id_counter = 0
        self.pending_requests = {}
    
    def connect_server(self, server_name: str, server: MCPServer):
        """Connect to an MCP server"""
        self.servers[server_name] = server
        logger.info(f"Connected to MCP server: {server_name}")
    
    def _next_message_id(self) -> str:
        """Generate next message ID"""
        self.message_id_counter += 1
        return f"{self.name}_{self.message_id_counter}"
    
    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """List available tools from a server"""
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not connected")
        
        message = MCPMessage(
            id=self._next_message_id(),
            type=MCPMessageType.REQUEST,
            method="tools/list"
        )
        
        response = await self.servers[server_name].handle_request(message)
        
        if response.error:
            raise Exception(f"Error listing tools: {response.error}")
        
        return response.result.get("tools", [])
    
    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """Call a tool on a server"""
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not connected")
        
        message = MCPMessage(
            id=self._next_message_id(),
            type=MCPMessageType.TOOL_CALL,
            method="tools/call",
            params={"name": tool_name, "arguments": arguments}
        )
        
        response = await self.servers[server_name].handle_request(message)
        
        if response.error:
            raise Exception(f"Tool call error: {response.error}")
        
        return response.result


class MCPAgentWorkflow:
    """Workflow integrating Autogen agents with MCP"""
    
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
        
        self.mcp_client = MCPClient("agent_client")
        self.mcp_server = MCPToolServer("agent_server")
        self.mcp_client.connect_server("main_server", self.mcp_server)
        
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize MCP-aware agents"""
        self.coordinator_agent = autogen.AssistantAgent(
            name="mcp_coordinator",
            llm_config=self.llm_config,
            system_message="""You are an MCP coordinator agent.
            
Your responsibilities:
1. Identify when tools are needed
2. Format proper MCP tool requests
3. Interpret tool results
4. Coordinate between multiple MCP servers
5. Maintain context across tool calls"""
        )
        
        self.executor_agent = autogen.AssistantAgent(
            name="mcp_executor",
            llm_config=self.llm_config,
            system_message="""You are an MCP executor agent.
            
Your responsibilities:
1. Execute MCP protocol operations
2. Handle tool calls through MCP
3. Process responses and errors
4. Maintain MCP session state"""
        )
    
    async def execute_with_mcp(self, task: str) -> Dict[str, Any]:
        """Execute task using MCP protocol"""
        logger.info(f"Executing task with MCP: {task[:100]}...")
        start_time = time.time()
        
        # List available tools
        tools = await self.mcp_client.list_tools("main_server")
        logger.info(f"Available MCP tools: {[t['name'] for t in tools]}")
        
        # Determine which tools to use
        tool_plan = self._plan_tool_usage(task, tools)
        
        # Execute tool calls
        results = []
        for tool_call in tool_plan:
            result = await self.mcp_client.call_tool(
                "main_server",
                tool_call["tool"],
                tool_call["arguments"]
            )
            results.append({
                "tool": tool_call["tool"],
                "result": result
            })
            logger.info(f"MCP tool {tool_call['tool']} executed")
        
        # Synthesize results
        final_answer = self._synthesize_results(task, results)
        
        elapsed_time = time.time() - start_time
        
        return {
            "task": task,
            "tools_used": [r["tool"] for r in results],
            "tool_results": results,
            "final_answer": final_answer,
            "execution_time": elapsed_time
        }
    
    def _plan_tool_usage(self, task: str, tools: List[Dict]) -> List[Dict]:
        """Plan which tools to use for the task"""
        tool_names = [t["name"] for t in tools]
        
        message = f"""Task: {task}

Available MCP tools: {tool_names}

Determine which tools to use and with what arguments."""
        
        response = self.coordinator_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        # Simple parsing for demonstration
        tool_plan = []
        if "calculate" in task.lower() or "math" in task.lower():
            tool_plan.append({
                "tool": "calculate",
                "arguments": {"expression": "2 + 2 * 3"}
            })
        
        if "search" in task.lower() or "find" in task.lower():
            tool_plan.append({
                "tool": "search",
                "arguments": {"query": "relevant information", "limit": 3}
            })
        
        if not tool_plan:
            tool_plan.append({
                "tool": "search",
                "arguments": {"query": task, "limit": 5}
            })
        
        return tool_plan
    
    def _synthesize_results(self, task: str, results: List[Dict]) -> str:
        """Synthesize tool results into final answer"""
        message = f"""Task: {task}

MCP Tool Results:
{json.dumps(results, indent=2)}

Synthesize these results into a comprehensive answer."""
        
        response = self.executor_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        return response


class MCPResourceServer(MCPServer):
    """MCP server providing resource access"""
    
    def __init__(self, name: str = "resource_server"):
        self.name = name
        self.resources = {}
        self._initialize_resources()
    
    def _initialize_resources(self):
        """Initialize available resources"""
        self.resources["database"] = {
            "uri": "db://example",
            "name": "Example Database",
            "mimeType": "application/sql"
        }
        
        self.resources["api"] = {
            "uri": "api://example",
            "name": "Example API",
            "mimeType": "application/json"
        }
        
        self.resources["filesystem"] = {
            "uri": "file://example",
            "name": "File System",
            "mimeType": "text/plain"
        }
    
    async def handle_request(self, message: MCPMessage) -> MCPMessage:
        """Handle resource requests"""
        if message.method == "resources/list":
            return self._handle_list_resources(message)
        elif message.method == "resources/read":
            return self._handle_read_resource(message)
        else:
            return MCPMessage(
                id=message.id,
                type=MCPMessageType.ERROR,
                error={"code": -32601, "message": "Method not found"}
            )
    
    def _handle_list_resources(self, message: MCPMessage) -> MCPMessage:
        """List available resources"""
        resources_list = []
        for name, info in self.resources.items():
            resources_list.append({
                "uri": info["uri"],
                "name": info["name"],
                "mimeType": info["mimeType"]
            })
        
        return MCPMessage(
            id=message.id,
            type=MCPMessageType.RESPONSE,
            result={"resources": resources_list}
        )
    
    def _handle_read_resource(self, message: MCPMessage) -> MCPMessage:
        """Read resource content"""
        uri = message.params.get("uri")
        
        # Simulated resource reading
        content = f"[Content from {uri}]"
        
        return MCPMessage(
            id=message.id,
            type=MCPMessageType.RESPONSE,
            result={
                "contents": [
                    {"uri": uri, "mimeType": "text/plain", "text": content}
                ]
            }
        )
    
    def get_capabilities(self) -> List[MCPCapability]:
        """Return server capabilities"""
        return [MCPCapability.RESOURCES]
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Return available tools (none for resource server)"""
        return []


async def run_mcp_examples():
    """Demonstrate MCP pattern"""
    logger.info("Starting MCP Pattern Examples")
    
    logger.info("\n=== Basic MCP Tool Server Example ===")
    
    # Create MCP server and client
    server = MCPToolServer("example_server")
    client = MCPClient("example_client")
    client.connect_server("server1", server)
    
    # List tools
    tools = await client.list_tools("server1")
    logger.info(f"Available tools: {[t['name'] for t in tools]}")
    
    # Call a tool
    calc_result = await client.call_tool(
        "server1",
        "calculate",
        {"expression": "10 * 5 + 3"}
    )
    logger.info(f"Calculation result: {calc_result}")
    
    search_result = await client.call_tool(
        "server1",
        "search",
        {"query": "MCP protocol", "limit": 3}
    )
    logger.info(f"Search results: {search_result}")
    
    logger.info("\n=== MCP Agent Workflow Example ===")
    
    workflow = MCPAgentWorkflow()
    
    task1 = "Calculate the result of 15 * 8 and then search for information about it"
    result1 = await workflow.execute_with_mcp(task1)
    
    logger.info(f"\nTask: {task1}")
    logger.info(f"Tools used: {result1['tools_used']}")
    logger.info(f"Execution time: {result1['execution_time']:.2f}s")
    
    logger.info("\n=== MCP Resource Server Example ===")
    
    resource_server = MCPResourceServer()
    client.connect_server("resource_server", resource_server)
    
    # List resources
    list_msg = MCPMessage(
        id="req_1",
        type=MCPMessageType.REQUEST,
        method="resources/list"
    )
    
    resources_response = await resource_server.handle_request(list_msg)
    logger.info(f"Available resources: {resources_response.result}")
    
    # Read a resource
    read_msg = MCPMessage(
        id="req_2",
        type=MCPMessageType.REQUEST,
        method="resources/read",
        params={"uri": "db://example"}
    )
    
    content_response = await resource_server.handle_request(read_msg)
    logger.info(f"Resource content: {content_response.result}")
    
    logger.info("\n=== Custom Tool Registration Example ===")
    
    # Register custom tool
    def custom_analyzer(text: str, mode: str = "basic") -> Dict[str, Any]:
        """Custom text analyzer"""
        return {
            "length": len(text),
            "words": len(text.split()),
            "mode": mode
        }
    
    server.register_tool(
        "analyze_text",
        custom_analyzer,
        {
            "description": "Analyze text properties",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "mode": {"type": "string", "default": "basic"}
                },
                "required": ["text"]
            }
        }
    )
    
    analysis_result = await client.call_tool(
        "server1",
        "analyze_text",
        {"text": "This is a sample text for analysis", "mode": "detailed"}
    )
    logger.info(f"Text analysis result: {analysis_result}")
    
    logger.info("\nMCP Pattern Examples Complete")


if __name__ == "__main__":
    asyncio.run(run_mcp_examples())