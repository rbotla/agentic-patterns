"""
A2A (Agent-to-Agent) Communication Pattern Implementation
Direct communication protocols between agents without human intervention
Includes message passing, event-driven communication, and shared memory patterns
"""

import autogen
from typing import Dict, Any, Optional, List, Callable, Tuple
import logging
from utils.logging_utils import setup_logging
import time
import json
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
from abc import ABC, abstractmethod

logger = setup_logging(__name__)


class A2AMessageType(Enum):
    """Types of A2A messages"""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    ACKNOWLEDGMENT = "ack"
    ERROR = "error"


class A2AProtocol(Enum):
    """A2A communication protocols"""
    DIRECT = "direct"  # Direct point-to-point
    PUBSUB = "pubsub"  # Publish-Subscribe
    BROADCAST = "broadcast"  # One-to-many
    REQUEST_REPLY = "request_reply"  # Synchronous request-response
    STREAMING = "streaming"  # Continuous data stream


@dataclass
class A2AMessage:
    """Agent-to-Agent message"""
    id: str
    sender: str
    recipient: str
    type: A2AMessageType
    protocol: A2AProtocol
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    requires_ack: bool = False
    ttl: Optional[int] = None  # Time to live (hops)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "type": self.type.value,
            "protocol": self.protocol.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "requires_ack": self.requires_ack,
            "ttl": self.ttl
        }


class MessageBus:
    """Central message bus for A2A communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_queue: Dict[str, deque] = {}
        self.message_history = []
        self.lock = threading.Lock()
        self.message_counter = 0
    
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic"""
        with self.lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(callback)
            logger.info(f"Subscribed to topic: {topic}")
    
    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from a topic"""
        with self.lock:
            if topic in self.subscribers and callback in self.subscribers[topic]:
                self.subscribers[topic].remove(callback)
                logger.info(f"Unsubscribed from topic: {topic}")
    
    def publish(self, topic: str, message: A2AMessage):
        """Publish message to a topic"""
        with self.lock:
            self.message_history.append(message)
            
            if topic in self.subscribers:
                for callback in self.subscribers[topic]:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Error in subscriber callback: {e}")
    
    def send_direct(self, recipient: str, message: A2AMessage):
        """Send direct message to specific agent"""
        with self.lock:
            if recipient not in self.message_queue:
                self.message_queue[recipient] = deque(maxlen=100)
            
            self.message_queue[recipient].append(message)
            self.message_history.append(message)
    
    def receive(self, agent_id: str) -> Optional[A2AMessage]:
        """Receive messages for an agent"""
        with self.lock:
            if agent_id in self.message_queue and self.message_queue[agent_id]:
                return self.message_queue[agent_id].popleft()
        return None
    
    def broadcast(self, message: A2AMessage, exclude: Optional[List[str]] = None):
        """Broadcast message to all agents"""
        with self.lock:
            exclude = exclude or []
            for agent_id in self.message_queue.keys():
                if agent_id not in exclude:
                    self.message_queue[agent_id].append(message)
            self.message_history.append(message)


class A2AAgent(ABC):
    """Base class for A2A-capable agents"""
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.received_messages = []
        self.sent_messages = []
        self.message_handlers = {}
        self.running = False
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.register_handler(A2AMessageType.HEARTBEAT, self._handle_heartbeat)
        self.register_handler(A2AMessageType.ERROR, self._handle_error)
    
    def register_handler(self, message_type: A2AMessageType, handler: Callable):
        """Register handler for message type"""
        self.message_handlers[message_type] = handler
    
    @abstractmethod
    async def process_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Process incoming message - to be implemented by subclasses"""
        pass
    
    def send_message(
        self,
        recipient: str,
        content: Any,
        message_type: A2AMessageType = A2AMessageType.REQUEST,
        protocol: A2AProtocol = A2AProtocol.DIRECT
    ) -> str:
        """Send message to another agent"""
        message_id = f"{self.agent_id}_{len(self.sent_messages)}"
        
        message = A2AMessage(
            id=message_id,
            sender=self.agent_id,
            recipient=recipient,
            type=message_type,
            protocol=protocol,
            content=content
        )
        
        self.sent_messages.append(message)
        
        if protocol == A2AProtocol.DIRECT:
            self.message_bus.send_direct(recipient, message)
        elif protocol == A2AProtocol.BROADCAST:
            self.message_bus.broadcast(message, exclude=[self.agent_id])
        elif protocol == A2AProtocol.PUBSUB:
            self.message_bus.publish(recipient, message)  # recipient as topic
        
        logger.info(f"{self.agent_id} sent {message_type.value} to {recipient}")
        return message_id
    
    def receive_messages(self) -> List[A2AMessage]:
        """Receive all pending messages"""
        messages = []
        while True:
            msg = self.message_bus.receive(self.agent_id)
            if msg is None:
                break
            messages.append(msg)
            self.received_messages.append(msg)
        return messages
    
    async def run(self):
        """Main agent loop"""
        self.running = True
        logger.info(f"{self.agent_id} started")
        
        while self.running:
            messages = self.receive_messages()
            
            for message in messages:
                if message.type in self.message_handlers:
                    await self.message_handlers[message.type](message)
                else:
                    response = await self.process_message(message)
                    if response:
                        self.send_message(
                            response.recipient,
                            response.content,
                            response.type,
                            response.protocol
                        )
            
            await asyncio.sleep(0.1)
    
    def stop(self):
        """Stop the agent"""
        self.running = False
        logger.info(f"{self.agent_id} stopped")
    
    async def _handle_heartbeat(self, message: A2AMessage):
        """Handle heartbeat message"""
        logger.debug(f"{self.agent_id} received heartbeat from {message.sender}")
    
    async def _handle_error(self, message: A2AMessage):
        """Handle error message"""
        logger.error(f"{self.agent_id} received error from {message.sender}: {message.content}")


class TaskCoordinatorAgent(A2AAgent):
    """Coordinator agent that delegates tasks"""
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, message_bus)
        self.task_queue = deque()
        self.worker_status = {}
        self.task_results = {}
    
    async def process_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Process incoming messages"""
        if message.type == A2AMessageType.REQUEST:
            # Task request from user
            task_id = f"task_{len(self.task_queue)}"
            self.task_queue.append({
                "id": task_id,
                "content": message.content,
                "requester": message.sender
            })
            
            # Find available worker
            worker = self._find_available_worker()
            if worker:
                self._assign_task(worker, task_id, message.content)
            
            return A2AMessage(
                id=f"{message.id}_ack",
                sender=self.agent_id,
                recipient=message.sender,
                type=A2AMessageType.ACKNOWLEDGMENT,
                protocol=A2AProtocol.DIRECT,
                content={"status": "accepted", "task_id": task_id}
            )
        
        elif message.type == A2AMessageType.RESPONSE:
            # Task result from worker
            task_id = message.metadata.get("task_id")
            if task_id:
                self.task_results[task_id] = message.content
                self.worker_status[message.sender] = "available"
                logger.info(f"Received result for {task_id} from {message.sender}")
        
        return None
    
    def _find_available_worker(self) -> Optional[str]:
        """Find an available worker"""
        for worker, status in self.worker_status.items():
            if status == "available":
                return worker
        return None
    
    def _assign_task(self, worker: str, task_id: str, content: Any):
        """Assign task to worker"""
        self.worker_status[worker] = "busy"
        
        self.send_message(
            worker,
            content,
            A2AMessageType.REQUEST,
            A2AProtocol.DIRECT
        )
        
        logger.info(f"Assigned {task_id} to {worker}")


class WorkerAgent(A2AAgent):
    """Worker agent that processes tasks"""
    
    def __init__(self, agent_id: str, message_bus: MessageBus, specialization: str = "general"):
        super().__init__(agent_id, message_bus)
        self.specialization = specialization
        self.tasks_completed = 0
    
    async def process_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Process task requests"""
        if message.type == A2AMessageType.REQUEST:
            # Process the task
            result = await self._process_task(message.content)
            self.tasks_completed += 1
            
            # Send result back
            return A2AMessage(
                id=f"{message.id}_result",
                sender=self.agent_id,
                recipient=message.sender,
                type=A2AMessageType.RESPONSE,
                protocol=A2AProtocol.DIRECT,
                content=result,
                metadata={"task_id": message.metadata.get("task_id")}
            )
        
        return None
    
    async def _process_task(self, task: Any) -> Dict[str, Any]:
        """Process a task based on specialization"""
        logger.info(f"{self.agent_id} processing task: {task}")
        
        # Simulate task processing
        await asyncio.sleep(0.5)
        
        return {
            "status": "completed",
            "result": f"Processed by {self.agent_id} ({self.specialization})",
            "task": task,
            "timestamp": time.time()
        }


class EventDrivenWorkflow:
    """Event-driven A2A communication workflow"""
    
    def __init__(self):
        self.message_bus = MessageBus()
        self.agents = {}
        self.event_log = []
    
    def add_agent(self, agent: A2AAgent):
        """Add agent to workflow"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent: {agent.agent_id}")
    
    def emit_event(self, event_type: str, data: Any):
        """Emit an event to all subscribed agents"""
        event_message = A2AMessage(
            id=f"event_{len(self.event_log)}",
            sender="system",
            recipient="*",  # Broadcast
            type=A2AMessageType.EVENT,
            protocol=A2AProtocol.BROADCAST,
            content={
                "event_type": event_type,
                "data": data
            }
        )
        
        self.message_bus.broadcast(event_message)
        self.event_log.append(event_message)
        logger.info(f"Emitted event: {event_type}")
    
    async def run_workflow(self, duration: float = 5.0):
        """Run the workflow for specified duration"""
        # Start all agents
        tasks = []
        for agent in self.agents.values():
            tasks.append(asyncio.create_task(agent.run()))
        
        # Run for specified duration
        await asyncio.sleep(duration)
        
        # Stop all agents
        for agent in self.agents.values():
            agent.stop()
        
        # Cancel tasks
        for task in tasks:
            task.cancel()


class SharedMemoryProtocol:
    """Shared memory communication for agents"""
    
    def __init__(self):
        self.memory = {}
        self.locks = {}
        self.access_log = []
    
    def write(self, key: str, value: Any, agent_id: str):
        """Write to shared memory"""
        if key not in self.locks:
            self.locks[key] = threading.Lock()
        
        with self.locks[key]:
            self.memory[key] = value
            self.access_log.append({
                "operation": "write",
                "key": key,
                "agent": agent_id,
                "timestamp": time.time()
            })
            logger.debug(f"{agent_id} wrote to {key}")
    
    def read(self, key: str, agent_id: str) -> Any:
        """Read from shared memory"""
        if key not in self.locks:
            self.locks[key] = threading.Lock()
        
        with self.locks[key]:
            value = self.memory.get(key)
            self.access_log.append({
                "operation": "read",
                "key": key,
                "agent": agent_id,
                "timestamp": time.time()
            })
            logger.debug(f"{agent_id} read from {key}")
            return value
    
    def atomic_update(self, key: str, update_func: Callable, agent_id: str):
        """Atomically update a value"""
        if key not in self.locks:
            self.locks[key] = threading.Lock()
        
        with self.locks[key]:
            current_value = self.memory.get(key)
            new_value = update_func(current_value)
            self.memory[key] = new_value
            self.access_log.append({
                "operation": "atomic_update",
                "key": key,
                "agent": agent_id,
                "timestamp": time.time()
            })
            logger.debug(f"{agent_id} atomically updated {key}")


class A2AWorkflow:
    """Main A2A communication workflow with Autogen integration"""
    
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
        
        self.message_bus = MessageBus()
        self.shared_memory = SharedMemoryProtocol()
        self.agents = {}
        self._initialize_autogen_agents()
    
    def _initialize_autogen_agents(self):
        """Initialize Autogen agents with A2A capabilities"""
        self.coordinator = autogen.AssistantAgent(
            name="a2a_coordinator",
            llm_config=self.llm_config,
            system_message="""You coordinate agent-to-agent communication.
            
Responsibilities:
1. Route messages between agents
2. Manage communication protocols
3. Handle failures and retries
4. Monitor communication patterns"""
        )
        
        self.analyst = autogen.AssistantAgent(
            name="a2a_analyst",
            llm_config=self.llm_config,
            system_message="""You analyze A2A communication patterns.
            
Responsibilities:
1. Monitor message flow
2. Identify bottlenecks
3. Optimize routing
4. Detect anomalies"""
        )
    
    def demonstrate_direct_communication(self) -> Dict[str, Any]:
        """Demonstrate direct A2A communication"""
        logger.info("Demonstrating direct A2A communication")
        
        # Create worker agents
        worker1 = WorkerAgent("worker1", self.message_bus, "data_processing")
        worker2 = WorkerAgent("worker2", self.message_bus, "analysis")
        
        # Direct message exchange
        msg_id = worker1.send_message(
            "worker2",
            {"task": "analyze", "data": [1, 2, 3]},
            A2AMessageType.REQUEST
        )
        
        # Worker2 receives and processes
        messages = worker2.receive_messages()
        
        results = {
            "sent_message_id": msg_id,
            "received_messages": len(messages),
            "message_history": len(self.message_bus.message_history)
        }
        
        logger.info(f"Direct communication results: {results}")
        return results
    
    def demonstrate_pubsub_pattern(self) -> Dict[str, Any]:
        """Demonstrate publish-subscribe pattern"""
        logger.info("Demonstrating publish-subscribe pattern")
        
        results = {"subscriptions": 0, "published": 0, "received": {}}
        
        # Create callback functions for subscribers
        def subscriber1_callback(msg: A2AMessage):
            results["received"]["subscriber1"] = msg.content
        
        def subscriber2_callback(msg: A2AMessage):
            results["received"]["subscriber2"] = msg.content
        
        # Subscribe to topics
        self.message_bus.subscribe("data_updates", subscriber1_callback)
        self.message_bus.subscribe("data_updates", subscriber2_callback)
        results["subscriptions"] = 2
        
        # Publish message
        publisher_msg = A2AMessage(
            id="pub_1",
            sender="publisher",
            recipient="data_updates",
            type=A2AMessageType.EVENT,
            protocol=A2AProtocol.PUBSUB,
            content={"update": "new_data", "value": 42}
        )
        
        self.message_bus.publish("data_updates", publisher_msg)
        results["published"] = 1
        
        logger.info(f"Pub-sub results: {results}")
        return results
    
    def demonstrate_shared_memory(self) -> Dict[str, Any]:
        """Demonstrate shared memory communication"""
        logger.info("Demonstrating shared memory communication")
        
        # Write to shared memory
        self.shared_memory.write("task_queue", ["task1", "task2"], "producer")
        self.shared_memory.write("results", {}, "producer")
        
        # Read from shared memory
        tasks = self.shared_memory.read("task_queue", "consumer")
        
        # Atomic update
        def update_results(current):
            if current is None:
                current = {}
            current["task1"] = "completed"
            return current
        
        self.shared_memory.atomic_update("results", update_results, "consumer")
        
        results = {
            "tasks_read": tasks,
            "final_results": self.shared_memory.read("results", "observer"),
            "access_log_entries": len(self.shared_memory.access_log)
        }
        
        logger.info(f"Shared memory results: {results}")
        return results


async def run_a2a_examples():
    """Demonstrate A2A communication patterns"""
    logger.info("Starting A2A Communication Pattern Examples")
    
    logger.info("\n=== Basic Message Bus Example ===")
    
    message_bus = MessageBus()
    
    # Create agents
    coordinator = TaskCoordinatorAgent("coordinator", message_bus)
    worker1 = WorkerAgent("worker1", message_bus, "computation")
    worker2 = WorkerAgent("worker2", message_bus, "validation")
    
    # Register workers with coordinator
    coordinator.worker_status["worker1"] = "available"
    coordinator.worker_status["worker2"] = "available"
    
    # Send task to coordinator
    task_msg = A2AMessage(
        id="task_001",
        sender="client",
        recipient="coordinator",
        type=A2AMessageType.REQUEST,
        protocol=A2AProtocol.DIRECT,
        content={"operation": "process", "data": [1, 2, 3, 4, 5]}
    )
    
    message_bus.send_direct("coordinator", task_msg)
    
    # Process messages
    await coordinator.process_message(task_msg)
    
    logger.info(f"Task queue size: {len(coordinator.task_queue)}")
    logger.info(f"Worker status: {coordinator.worker_status}")
    
    logger.info("\n=== Event-Driven Workflow Example ===")
    
    workflow = EventDrivenWorkflow()
    
    # Add agents
    workflow.add_agent(WorkerAgent("event_worker1", workflow.message_bus))
    workflow.add_agent(WorkerAgent("event_worker2", workflow.message_bus))
    
    # Emit events
    workflow.emit_event("task_available", {"task_id": "001"})
    workflow.emit_event("priority_change", {"level": "high"})
    
    # Run workflow
    await workflow.run_workflow(duration=2.0)
    
    logger.info(f"Events emitted: {len(workflow.event_log)}")
    
    logger.info("\n=== A2A Workflow Demonstrations ===")
    
    a2a_workflow = A2AWorkflow()
    
    # Direct communication
    direct_results = a2a_workflow.demonstrate_direct_communication()
    logger.info(f"Direct communication: {direct_results}")
    
    # Publish-Subscribe
    pubsub_results = a2a_workflow.demonstrate_pubsub_pattern()
    logger.info(f"Publish-Subscribe: {pubsub_results}")
    
    # Shared Memory
    shared_mem_results = a2a_workflow.demonstrate_shared_memory()
    logger.info(f"Shared Memory: {shared_mem_results}")
    
    logger.info("\n=== Broadcast Communication Example ===")
    
    # Broadcast message to all agents
    broadcast_msg = A2AMessage(
        id="broadcast_001",
        sender="system",
        recipient="*",
        type=A2AMessageType.BROADCAST,
        protocol=A2AProtocol.BROADCAST,
        content={"announcement": "System maintenance in 5 minutes"}
    )
    
    message_bus.broadcast(broadcast_msg)
    
    # Check reception
    worker1_msgs = worker1.receive_messages()
    worker2_msgs = worker2.receive_messages()
    
    logger.info(f"Worker1 received {len(worker1_msgs)} broadcast messages")
    logger.info(f"Worker2 received {len(worker2_msgs)} broadcast messages")
    
    logger.info("\nA2A Communication Pattern Examples Complete")


if __name__ == "__main__":
    asyncio.run(run_a2a_examples())