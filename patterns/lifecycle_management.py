"""
Agent Lifecycle Management Pattern Implementation
Complete lifecycle management for agents including initialization, deployment,
scaling, health monitoring, updates, and graceful shutdown
"""

import autogen
from typing import Dict, Any, Optional, List, Callable, Union
import logging
from utils.logging_utils import setup_logging
import time
import json
import threading
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import signal
import sys
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod

logger = setup_logging(__name__)


class AgentState(Enum):
    """Agent lifecycle states"""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    BUSY = "busy"
    SCALING = "scaling"
    UPDATING = "updating"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    TERMINATED = "terminated"


class HealthStatus(Enum):
    """Health check statuses"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class AgentMetadata:
    """Metadata for agent lifecycle management"""
    id: str
    name: str
    version: str
    created_at: float
    state: AgentState
    health_status: HealthStatus = HealthStatus.UNKNOWN
    tags: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    last_health_check: Optional[float] = None
    error_count: int = 0
    restart_count: int = 0
    
    def update_state(self, new_state: AgentState):
        """Update agent state with timestamp"""
        self.state = new_state
        logger.info(f"Agent {self.name} ({self.id}) state changed to {new_state.value}")


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    status: HealthStatus
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    latency: float = 0.0
    
    def is_healthy(self) -> bool:
        """Check if status indicates health"""
        return self.status == HealthStatus.HEALTHY


class LifecycleHook(ABC):
    """Abstract base class for lifecycle hooks"""
    
    @abstractmethod
    async def execute(self, agent_id: str, context: Dict[str, Any]) -> bool:
        """Execute the hook. Return True for success, False for failure."""
        pass


class InitializationHook(LifecycleHook):
    """Hook executed during agent initialization"""
    
    def __init__(self, init_func: Callable):
        self.init_func = init_func
    
    async def execute(self, agent_id: str, context: Dict[str, Any]) -> bool:
        """Execute initialization"""
        try:
            if asyncio.iscoroutinefunction(self.init_func):
                await self.init_func(agent_id, context)
            else:
                self.init_func(agent_id, context)
            return True
        except Exception as e:
            logger.error(f"Initialization hook failed for {agent_id}: {e}")
            return False


class ShutdownHook(LifecycleHook):
    """Hook executed during agent shutdown"""
    
    def __init__(self, shutdown_func: Callable):
        self.shutdown_func = shutdown_func
    
    async def execute(self, agent_id: str, context: Dict[str, Any]) -> bool:
        """Execute shutdown"""
        try:
            if asyncio.iscoroutinefunction(self.shutdown_func):
                await self.shutdown_func(agent_id, context)
            else:
                self.shutdown_func(agent_id, context)
            return True
        except Exception as e:
            logger.error(f"Shutdown hook failed for {agent_id}: {e}")
            return False


class ManagedAgent:
    """Agent wrapper with lifecycle management"""
    
    def __init__(
        self,
        agent: autogen.ConversableAgent,
        metadata: AgentMetadata,
        health_check_func: Optional[Callable] = None
    ):
        self.agent = agent
        self.metadata = metadata
        self.health_check_func = health_check_func or self._default_health_check
        
        # Lifecycle hooks
        self.initialization_hooks: List[LifecycleHook] = []
        self.shutdown_hooks: List[LifecycleHook] = []
        self.update_hooks: List[LifecycleHook] = []
        
        # Runtime state
        self.start_time: Optional[float] = None
        self.last_activity: float = time.time()
        self.task_queue: deque = deque()
        self.active_tasks: Dict[str, Dict] = {}
        
        # Performance metrics
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        
        # Wrap agent methods for lifecycle management
        self._wrap_agent_methods()
    
    def add_initialization_hook(self, hook: LifecycleHook):
        """Add initialization hook"""
        self.initialization_hooks.append(hook)
    
    def add_shutdown_hook(self, hook: LifecycleHook):
        """Add shutdown hook"""
        self.shutdown_hooks.append(hook)
    
    def add_update_hook(self, hook: LifecycleHook):
        """Add update hook"""
        self.update_hooks.append(hook)
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        self.metadata.update_state(AgentState.INITIALIZING)
        
        try:
            # Execute initialization hooks
            for hook in self.initialization_hooks:
                success = await hook.execute(self.metadata.id, {"metadata": self.metadata})
                if not success:
                    self.metadata.update_state(AgentState.FAILED)
                    return False
            
            self.start_time = time.time()
            self.metadata.update_state(AgentState.READY)
            return True
            
        except Exception as e:
            logger.error(f"Agent {self.metadata.name} initialization failed: {e}")
            self.metadata.update_state(AgentState.FAILED)
            return False
    
    async def start(self) -> bool:
        """Start the agent"""
        if self.metadata.state != AgentState.READY:
            if not await self.initialize():
                return False
        
        self.metadata.update_state(AgentState.RUNNING)
        logger.info(f"Agent {self.metadata.name} started successfully")
        return True
    
    async def stop(self, graceful: bool = True) -> bool:
        """Stop the agent"""
        self.metadata.update_state(AgentState.STOPPING)
        
        try:
            if graceful:
                # Wait for active tasks to complete
                timeout = 30  # seconds
                start_wait = time.time()
                
                while self.active_tasks and (time.time() - start_wait) < timeout:
                    await asyncio.sleep(0.1)
                
                if self.active_tasks:
                    logger.warning(f"Agent {self.metadata.name} stopped with {len(self.active_tasks)} active tasks")
            
            # Execute shutdown hooks
            for hook in self.shutdown_hooks:
                try:
                    await hook.execute(self.metadata.id, {"metadata": self.metadata})
                except Exception as e:
                    logger.error(f"Shutdown hook error for {self.metadata.name}: {e}")
            
            self.metadata.update_state(AgentState.STOPPED)
            logger.info(f"Agent {self.metadata.name} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Agent {self.metadata.name} shutdown failed: {e}")
            self.metadata.update_state(AgentState.FAILED)
            return False
    
    async def restart(self) -> bool:
        """Restart the agent"""
        logger.info(f"Restarting agent {self.metadata.name}")
        
        if not await self.stop(graceful=True):
            logger.error(f"Failed to stop agent {self.metadata.name} for restart")
            return False
        
        self.metadata.restart_count += 1
        
        # Reset error count on successful restart
        self.metadata.error_count = 0
        
        return await self.start()
    
    async def update(self, new_config: Dict[str, Any]) -> bool:
        """Update agent configuration"""
        self.metadata.update_state(AgentState.UPDATING)
        
        try:
            old_config = self.metadata.config.copy()
            
            # Execute update hooks
            context = {
                "old_config": old_config,
                "new_config": new_config,
                "metadata": self.metadata
            }
            
            for hook in self.update_hooks:
                success = await hook.execute(self.metadata.id, context)
                if not success:
                    logger.error(f"Update failed for agent {self.metadata.name}")
                    self.metadata.update_state(AgentState.RUNNING)  # Revert state
                    return False
            
            # Update configuration
            self.metadata.config.update(new_config)
            
            # Apply configuration to underlying agent if applicable
            if hasattr(self.agent, 'llm_config') and 'llm_config' in new_config:
                self.agent.llm_config.update(new_config['llm_config'])
            
            self.metadata.update_state(AgentState.RUNNING)
            logger.info(f"Agent {self.metadata.name} updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Agent {self.metadata.name} update failed: {e}")
            self.metadata.update_state(AgentState.FAILED)
            return False
    
    async def health_check(self) -> HealthCheckResult:
        """Perform health check"""
        start_time = time.time()
        
        try:
            result = await self._execute_health_check()
            result.latency = time.time() - start_time
            result.timestamp = time.time()
            
            self.metadata.health_status = result.status
            self.metadata.last_health_check = result.timestamp
            
            return result
            
        except Exception as e:
            logger.error(f"Health check failed for {self.metadata.name}: {e}")
            error_result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                timestamp=time.time(),
                latency=time.time() - start_time,
                details={"error": str(e)}
            )
            self.metadata.health_status = HealthStatus.UNHEALTHY
            return error_result
    
    async def _execute_health_check(self) -> HealthCheckResult:
        """Execute the health check function"""
        if asyncio.iscoroutinefunction(self.health_check_func):
            return await self.health_check_func(self)
        else:
            return self.health_check_func(self)
    
    def _default_health_check(self, managed_agent) -> HealthCheckResult:
        """Default health check implementation"""
        details = {
            "state": self.metadata.state.value,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "request_count": self.request_count,
            "error_rate": self.metadata.error_count / max(self.request_count, 1),
            "active_tasks": len(self.active_tasks),
            "queue_size": len(self.task_queue)
        }
        
        # Determine health status
        if self.metadata.state == AgentState.FAILED:
            status = HealthStatus.UNHEALTHY
        elif self.metadata.state in [AgentState.RUNNING, AgentState.READY]:
            error_rate = details["error_rate"]
            if error_rate > 0.5:  # More than 50% errors
                status = HealthStatus.UNHEALTHY
            elif error_rate > 0.1:  # More than 10% errors
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
        else:
            status = HealthStatus.DEGRADED
        
        return HealthCheckResult(status=status, timestamp=time.time(), details=details)
    
    def _wrap_agent_methods(self):
        """Wrap agent methods for lifecycle tracking"""
        if hasattr(self.agent, 'generate_reply'):
            original_generate_reply = self.agent.generate_reply
            
            def tracked_generate_reply(*args, **kwargs):
                task_id = str(uuid.uuid4())
                self.active_tasks[task_id] = {
                    "start_time": time.time(),
                    "args": args,
                    "kwargs": kwargs
                }
                
                self.metadata.update_state(AgentState.BUSY)
                start_time = time.time()
                
                try:
                    result = original_generate_reply(*args, **kwargs)
                    self.request_count += 1
                    self.last_activity = time.time()
                    return result
                    
                except Exception as e:
                    self.metadata.error_count += 1
                    self.error_count += 1
                    logger.error(f"Agent {self.metadata.name} request failed: {e}")
                    raise
                    
                finally:
                    processing_time = time.time() - start_time
                    self.total_processing_time += processing_time
                    
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]
                    
                    # Update state based on remaining tasks
                    if not self.active_tasks:
                        self.metadata.update_state(AgentState.RUNNING)
            
            self.agent.generate_reply = tracked_generate_reply
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            "uptime": uptime,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_processing_time": self.total_processing_time / max(self.request_count, 1),
            "active_tasks": len(self.active_tasks),
            "queue_size": len(self.task_queue),
            "restart_count": self.metadata.restart_count,
            "last_activity": self.last_activity,
            "health_status": self.metadata.health_status.value
        }


class AgentLifecycleManager:
    """Central manager for agent lifecycles"""
    
    def __init__(self):
        self.managed_agents: Dict[str, ManagedAgent] = {}
        self.agent_groups: Dict[str, List[str]] = defaultdict(list)
        self.health_check_interval = 30  # seconds
        self.health_monitor_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Scaling configuration
        self.auto_scaling_enabled = False
        self.scaling_policies: Dict[str, Dict] = {}
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.shutdown_all_agents())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def register_agent(
        self,
        agent: autogen.ConversableAgent,
        version: str = "1.0.0",
        tags: Dict[str, str] = None,
        config: Dict[str, Any] = None,
        health_check_func: Optional[Callable] = None,
        group: Optional[str] = None
    ) -> str:
        """Register an agent for lifecycle management"""
        
        agent_id = str(uuid.uuid4())
        metadata = AgentMetadata(
            id=agent_id,
            name=agent.name,
            version=version,
            created_at=time.time(),
            state=AgentState.CREATED,
            tags=tags or {},
            config=config or {},
            dependencies=[],
            resource_requirements={}
        )
        
        managed_agent = ManagedAgent(agent, metadata, health_check_func)
        self.managed_agents[agent_id] = managed_agent
        
        if group:
            self.agent_groups[group].append(agent_id)
        
        logger.info(f"Registered agent {agent.name} with ID {agent_id}")
        return agent_id
    
    async def start_agent(self, agent_id: str) -> bool:
        """Start a specific agent"""
        managed_agent = self.managed_agents.get(agent_id)
        if not managed_agent:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        return await managed_agent.start()
    
    async def stop_agent(self, agent_id: str, graceful: bool = True) -> bool:
        """Stop a specific agent"""
        managed_agent = self.managed_agents.get(agent_id)
        if not managed_agent:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        return await managed_agent.stop(graceful)
    
    async def restart_agent(self, agent_id: str) -> bool:
        """Restart a specific agent"""
        managed_agent = self.managed_agents.get(agent_id)
        if not managed_agent:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        return await managed_agent.restart()
    
    async def update_agent(self, agent_id: str, new_config: Dict[str, Any]) -> bool:
        """Update agent configuration"""
        managed_agent = self.managed_agents.get(agent_id)
        if not managed_agent:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        return await managed_agent.update(new_config)
    
    async def start_all_agents(self, group: Optional[str] = None) -> Dict[str, bool]:
        """Start all agents or agents in a specific group"""
        if group:
            agent_ids = self.agent_groups.get(group, [])
        else:
            agent_ids = list(self.managed_agents.keys())
        
        results = {}
        for agent_id in agent_ids:
            results[agent_id] = await self.start_agent(agent_id)
        
        return results
    
    async def shutdown_all_agents(self, group: Optional[str] = None) -> Dict[str, bool]:
        """Shutdown all agents or agents in a specific group"""
        if group:
            agent_ids = self.agent_groups.get(group, [])
        else:
            agent_ids = list(self.managed_agents.keys())
        
        results = {}
        for agent_id in agent_ids:
            results[agent_id] = await self.stop_agent(agent_id)
        
        return results
    
    async def start_health_monitoring(self):
        """Start health monitoring for all agents"""
        if self.health_monitor_running:
            return
        
        self.health_monitor_running = True
        self.monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Health monitoring started")
    
    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        self.health_monitor_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _health_monitor_loop(self):
        """Health monitoring loop"""
        while self.health_monitor_running:
            try:
                health_results = await self.check_all_agents_health()
                
                # Handle unhealthy agents
                for agent_id, result in health_results.items():
                    if not result.is_healthy():
                        await self._handle_unhealthy_agent(agent_id, result)
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def check_all_agents_health(self) -> Dict[str, HealthCheckResult]:
        """Check health of all agents"""
        results = {}
        
        tasks = []
        agent_ids = []
        
        for agent_id, managed_agent in self.managed_agents.items():
            if managed_agent.metadata.state in [AgentState.RUNNING, AgentState.BUSY, AgentState.READY]:
                tasks.append(managed_agent.health_check())
                agent_ids.append(agent_id)
        
        if tasks:
            health_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for agent_id, result in zip(agent_ids, health_results):
                if isinstance(result, Exception):
                    results[agent_id] = HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        timestamp=time.time(),
                        details={"error": str(result)}
                    )
                else:
                    results[agent_id] = result
        
        return results
    
    async def _handle_unhealthy_agent(self, agent_id: str, health_result: HealthCheckResult):
        """Handle an unhealthy agent"""
        managed_agent = self.managed_agents.get(agent_id)
        if not managed_agent:
            return
        
        agent_name = managed_agent.metadata.name
        
        if health_result.status == HealthStatus.UNHEALTHY:
            # Attempt restart if not too many recent restarts
            if managed_agent.metadata.restart_count < 3:
                logger.warning(f"Agent {agent_name} is unhealthy, attempting restart")
                success = await managed_agent.restart()
                if success:
                    logger.info(f"Agent {agent_name} restarted successfully")
                else:
                    logger.error(f"Failed to restart agent {agent_name}")
                    managed_agent.metadata.update_state(AgentState.FAILED)
            else:
                logger.error(f"Agent {agent_name} has too many restart attempts, marking as failed")
                managed_agent.metadata.update_state(AgentState.FAILED)
        
        elif health_result.status == HealthStatus.DEGRADED:
            logger.warning(f"Agent {agent_name} is degraded: {health_result.details}")
            managed_agent.metadata.update_state(AgentState.DEGRADED)
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent"""
        managed_agent = self.managed_agents.get(agent_id)
        if not managed_agent:
            return None
        
        return {
            "metadata": {
                "id": managed_agent.metadata.id,
                "name": managed_agent.metadata.name,
                "version": managed_agent.metadata.version,
                "state": managed_agent.metadata.state.value,
                "health_status": managed_agent.metadata.health_status.value,
                "created_at": managed_agent.metadata.created_at,
                "tags": managed_agent.metadata.tags,
                "restart_count": managed_agent.metadata.restart_count,
                "error_count": managed_agent.metadata.error_count
            },
            "metrics": managed_agent.get_metrics()
        }
    
    def get_all_agents_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents"""
        return {
            agent_id: self.get_agent_status(agent_id)
            for agent_id in self.managed_agents
        }
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system overview"""
        states = defaultdict(int)
        health_statuses = defaultdict(int)
        total_requests = 0
        total_errors = 0
        
        for managed_agent in self.managed_agents.values():
            states[managed_agent.metadata.state.value] += 1
            health_statuses[managed_agent.metadata.health_status.value] += 1
            metrics = managed_agent.get_metrics()
            total_requests += metrics["request_count"]
            total_errors += metrics["error_count"]
        
        return {
            "total_agents": len(self.managed_agents),
            "states": dict(states),
            "health_statuses": dict(health_statuses),
            "system_metrics": {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "system_error_rate": total_errors / max(total_requests, 1),
                "groups": {group: len(agents) for group, agents in self.agent_groups.items()}
            }
        }


async def run_lifecycle_examples():
    """Demonstrate agent lifecycle management patterns"""
    logger.info("Starting Agent Lifecycle Management Pattern Examples")
    
    logger.info("\n=== Basic Agent Lifecycle Example ===")
    
    # Create lifecycle manager
    lifecycle_mgr = AgentLifecycleManager()
    
    # Create test agents
    test_agents = []
    for i in range(3):
        agent = autogen.ConversableAgent(
            name=f"test_agent_{i}",
            llm_config=False,
            human_input_mode="NEVER"
        )
        
        # Simple reply function
        agent.register_reply(
            trigger=lambda sender: True,
            reply_func=lambda **kwargs: (True, f"Response from {agent.name}"),
            position=0
        )
        
        test_agents.append(agent)
    
    # Register agents
    agent_ids = []
    for i, agent in enumerate(test_agents):
        agent_id = await lifecycle_mgr.register_agent(
            agent,
            version="1.0.0",
            tags={"type": "test", "index": str(i)},
            config={"timeout": 30, "max_retries": 3},
            group="test_group"
        )
        agent_ids.append(agent_id)
    
    # Start all agents
    start_results = await lifecycle_mgr.start_all_agents(group="test_group")
    logger.info(f"Agent start results: {start_results}")
    
    # Get system overview
    overview = lifecycle_mgr.get_system_overview()
    logger.info(f"System overview: {json.dumps(overview, indent=2)}")
    
    logger.info("\n=== Health Monitoring Example ===")
    
    # Start health monitoring
    await lifecycle_mgr.start_health_monitoring()
    
    # Simulate some agent activity
    for agent_id in agent_ids[:2]:  # Use first 2 agents
        managed_agent = lifecycle_mgr.managed_agents[agent_id]
        
        # Simulate requests
        for j in range(5):
            try:
                response = managed_agent.agent.generate_reply(
                    messages=[{"role": "user", "content": f"Test message {j}"}]
                )
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Request failed: {e}")
    
    # Wait for health check
    await asyncio.sleep(2)
    
    # Check health of all agents
    health_results = await lifecycle_mgr.check_all_agents_health()
    logger.info("Health check results:")
    for agent_id, result in health_results.items():
        agent_name = lifecycle_mgr.managed_agents[agent_id].metadata.name
        logger.info(f"  {agent_name}: {result.status.value} (latency: {result.latency:.3f}s)")
    
    logger.info("\n=== Configuration Update Example ===")
    
    # Update agent configuration
    if agent_ids:
        first_agent_id = agent_ids[0]
        new_config = {
            "timeout": 60,
            "max_retries": 5,
            "llm_config": {"temperature": 0.8}
        }
        
        update_success = await lifecycle_mgr.update_agent(first_agent_id, new_config)
        logger.info(f"Configuration update successful: {update_success}")
        
        # Check updated configuration
        agent_status = lifecycle_mgr.get_agent_status(first_agent_id)
        logger.info(f"Updated config: {agent_status['metadata']}")
    
    logger.info("\n=== Lifecycle Hooks Example ===")
    
    # Create agent with custom hooks
    hook_agent = autogen.ConversableAgent(
        name="hook_agent",
        llm_config=False,
        human_input_mode="NEVER"
    )
    
    # Custom initialization hook
    def custom_init(agent_id: str, context: Dict[str, Any]):
        logger.info(f"Custom initialization for {agent_id}")
        logger.info(f"Context: {context}")
    
    # Custom shutdown hook
    def custom_shutdown(agent_id: str, context: Dict[str, Any]):
        logger.info(f"Custom shutdown for {agent_id}")
        logger.info(f"Final metrics: {context}")
    
    hook_agent_id = await lifecycle_mgr.register_agent(
        hook_agent,
        version="1.0.0",
        tags={"type": "hook_test"}
    )
    
    managed_hook_agent = lifecycle_mgr.managed_agents[hook_agent_id]
    managed_hook_agent.add_initialization_hook(InitializationHook(custom_init))
    managed_hook_agent.add_shutdown_hook(ShutdownHook(custom_shutdown))
    
    # Start and stop to demonstrate hooks
    await lifecycle_mgr.start_agent(hook_agent_id)
    await lifecycle_mgr.stop_agent(hook_agent_id)
    
    logger.info("\n=== Agent Restart Example ===")
    
    # Simulate agent failure and restart
    if agent_ids:
        failing_agent_id = agent_ids[-1]
        managed_failing_agent = lifecycle_mgr.managed_agents[failing_agent_id]
        
        # Simulate failures
        managed_failing_agent.metadata.error_count = 5
        managed_failing_agent.metadata.update_state(AgentState.DEGRADED)
        
        logger.info(f"Agent state before restart: {managed_failing_agent.metadata.state.value}")
        
        # Restart the agent
        restart_success = await lifecycle_mgr.restart_agent(failing_agent_id)
        logger.info(f"Restart successful: {restart_success}")
        logger.info(f"Agent state after restart: {managed_failing_agent.metadata.state.value}")
        logger.info(f"Restart count: {managed_failing_agent.metadata.restart_count}")
    
    logger.info("\n=== Final System Status ===")
    
    # Get final status of all agents
    final_status = lifecycle_mgr.get_all_agents_status()
    
    logger.info("Final agent statuses:")
    for agent_id, status in final_status.items():
        if status:
            metadata = status["metadata"]
            metrics = status["metrics"]
            logger.info(f"  {metadata['name']}:")
            logger.info(f"    State: {metadata['state']}")
            logger.info(f"    Health: {metadata['health_status']}")
            logger.info(f"    Requests: {metrics['request_count']}")
            logger.info(f"    Error Rate: {metrics['error_rate']:.2%}")
            logger.info(f"    Uptime: {metrics['uptime']:.1f}s")
    
    # Cleanup
    await lifecycle_mgr.stop_health_monitoring()
    await lifecycle_mgr.shutdown_all_agents()
    
    final_overview = lifecycle_mgr.get_system_overview()
    logger.info(f"Final system overview: {json.dumps(final_overview, indent=2)}")
    
    logger.info("\nAgent Lifecycle Management Pattern Examples Complete")


if __name__ == "__main__":
    asyncio.run(run_lifecycle_examples())