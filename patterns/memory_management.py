"""
Memory Management Pattern Implementation
Advanced memory patterns for agents including episodic, semantic, and working memory
Context compression, memory consolidation, and retrieval strategies
"""

import autogen
from typing import Dict, Any, Optional, List, Tuple, Set
import logging
from utils.logging_utils import setup_logging
import time
import json
import hashlib
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import pickle
import sqlite3
import threading
from abc import ABC, abstractmethod

logger = setup_logging(__name__)


class MemoryType(Enum):
    """Types of memory systems"""
    WORKING = "working"        # Short-term, limited capacity
    EPISODIC = "episodic"      # Experiences and events
    SEMANTIC = "semantic"      # Facts and knowledge
    PROCEDURAL = "procedural"  # Skills and procedures
    CONTEXTUAL = "contextual"  # Context-dependent memories


class MemoryImportance(Enum):
    """Memory importance levels"""
    CRITICAL = 1.0
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    MINIMAL = 0.2


@dataclass
class MemoryItem:
    """Individual memory item"""
    id: str
    content: Any
    memory_type: MemoryType
    importance: float
    tags: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def decay_importance(self, decay_rate: float = 0.01):
        """Apply temporal decay to importance"""
        time_since_access = time.time() - self.last_accessed
        decay = np.exp(-decay_rate * time_since_access)
        self.importance *= decay
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "type": self.memory_type.value,
            "importance": self.importance,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "context": self.context
        }


class MemoryStore(ABC):
    """Abstract base class for memory stores"""
    
    @abstractmethod
    def store(self, memory: MemoryItem) -> str:
        """Store a memory item"""
        pass
    
    @abstractmethod
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a specific memory"""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[MemoryItem]:
        """Search memories by content similarity"""
        pass
    
    @abstractmethod
    def get_by_type(self, memory_type: MemoryType) -> List[MemoryItem]:
        """Get memories by type"""
        pass


class InMemoryStore(MemoryStore):
    """Simple in-memory storage"""
    
    def __init__(self, max_size: int = 1000):
        self.memories: Dict[str, MemoryItem] = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def store(self, memory: MemoryItem) -> str:
        """Store memory item"""
        with self.lock:
            if len(self.memories) >= self.max_size:
                self._evict_least_important()
            
            self.memories[memory.id] = memory
            return memory.id
    
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve memory by ID"""
        with self.lock:
            memory = self.memories.get(memory_id)
            if memory:
                memory.update_access()
            return memory
    
    def search(self, query: str, k: int = 5) -> List[MemoryItem]:
        """Simple text-based search"""
        with self.lock:
            scored_memories = []
            query_lower = query.lower()
            
            for memory in self.memories.values():
                score = self._calculate_relevance(memory, query_lower)
                if score > 0:
                    scored_memories.append((memory, score))
            
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            
            results = [memory for memory, _ in scored_memories[:k]]
            for memory in results:
                memory.update_access()
            
            return results
    
    def get_by_type(self, memory_type: MemoryType) -> List[MemoryItem]:
        """Get memories by type"""
        with self.lock:
            return [m for m in self.memories.values() 
                   if m.memory_type == memory_type]
    
    def _evict_least_important(self):
        """Evict least important memory"""
        if not self.memories:
            return
        
        least_important = min(
            self.memories.values(),
            key=lambda m: m.importance + (m.access_count * 0.1)
        )
        del self.memories[least_important.id]
        logger.debug(f"Evicted memory: {least_important.id}")
    
    def _calculate_relevance(self, memory: MemoryItem, query: str) -> float:
        """Calculate relevance score"""
        content_str = str(memory.content).lower()
        
        # Simple keyword matching
        words = query.split()
        score = 0.0
        
        for word in words:
            if word in content_str:
                score += 1.0
            
            for tag in memory.tags:
                if word in tag.lower():
                    score += 0.5
        
        # Boost by importance and access count
        score *= (memory.importance + memory.access_count * 0.01)
        
        return score


class SQLiteStore(MemoryStore):
    """SQLite-based persistent memory store"""
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    memory_type TEXT,
                    importance REAL,
                    tags TEXT,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    context TEXT,
                    embedding BLOB
                )
            """)
            conn.commit()
    
    def store(self, memory: MemoryItem) -> str:
        """Store memory in SQLite"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories VALUES 
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id,
                    json.dumps(memory.content),
                    memory.memory_type.value,
                    memory.importance,
                    json.dumps(list(memory.tags)),
                    memory.created_at,
                    memory.last_accessed,
                    memory.access_count,
                    json.dumps(memory.context),
                    pickle.dumps(memory.embedding) if memory.embedding else None
                ))
                conn.commit()
        return memory.id
    
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve memory from SQLite"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                memory = self._row_to_memory(row)
                
                # Update access
                memory.update_access()
                conn.execute("""
                    UPDATE memories SET last_accessed = ?, access_count = ? 
                    WHERE id = ?
                """, (memory.last_accessed, memory.access_count, memory_id))
                conn.commit()
                
                return memory
    
    def search(self, query: str, k: int = 5) -> List[MemoryItem]:
        """Search memories using FTS"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM memories 
                    WHERE content LIKE ? OR tags LIKE ?
                    ORDER BY importance DESC, access_count DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", k))
                
                results = []
                for row in cursor.fetchall():
                    memory = self._row_to_memory(row)
                    memory.update_access()
                    results.append(memory)
                
                # Update access counts
                for memory in results:
                    conn.execute("""
                        UPDATE memories SET last_accessed = ?, access_count = ? 
                        WHERE id = ?
                    """, (memory.last_accessed, memory.access_count, memory.id))
                
                conn.commit()
                return results
    
    def get_by_type(self, memory_type: MemoryType) -> List[MemoryItem]:
        """Get memories by type"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM memories WHERE memory_type = ?", 
                    (memory_type.value,)
                )
                return [self._row_to_memory(row) for row in cursor.fetchall()]
    
    def _row_to_memory(self, row) -> MemoryItem:
        """Convert database row to MemoryItem"""
        return MemoryItem(
            id=row[0],
            content=json.loads(row[1]),
            memory_type=MemoryType(row[2]),
            importance=row[3],
            tags=set(json.loads(row[4])),
            created_at=row[5],
            last_accessed=row[6],
            access_count=row[7],
            context=json.loads(row[8]),
            embedding=pickle.loads(row[9]) if row[9] else None
        )


class MemoryManager:
    """Central memory management system"""
    
    def __init__(
        self,
        store: MemoryStore,
        working_memory_size: int = 10,
        consolidation_threshold: int = 100
    ):
        self.store = store
        self.working_memory_size = working_memory_size
        self.consolidation_threshold = consolidation_threshold
        
        # Working memory (fast access)
        self.working_memory: deque = deque(maxlen=working_memory_size)
        
        # Memory statistics
        self.stats = {
            "memories_created": 0,
            "memories_retrieved": 0,
            "consolidations": 0,
            "evictions": 0
        }
        
        # Background consolidation
        self._last_consolidation = time.time()
    
    def remember(
        self,
        content: Any,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        tags: Set[str] = None,
        context: Dict[str, Any] = None
    ) -> str:
        """Store a new memory"""
        memory_id = self._generate_memory_id(content)
        
        memory = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or set(),
            context=context or {}
        )
        
        # Store in persistent storage
        self.store.store(memory)
        
        # Add to working memory
        self.working_memory.append(memory)
        
        self.stats["memories_created"] += 1
        
        # Trigger consolidation if needed
        if self.stats["memories_created"] % self.consolidation_threshold == 0:
            self._consolidate_memories()
        
        logger.debug(f"Stored memory: {memory_id}")
        return memory_id
    
    def recall(
        self,
        query: str,
        k: int = 5,
        memory_types: List[MemoryType] = None
    ) -> List[MemoryItem]:
        """Retrieve relevant memories"""
        # First search working memory
        working_results = []
        for memory in self.working_memory:
            if memory_types and memory.memory_type not in memory_types:
                continue
            
            if self._matches_query(memory, query):
                working_results.append(memory)
        
        # Then search persistent storage
        persistent_results = self.store.search(query, k)
        
        # Combine and deduplicate
        all_results = {}
        for memory in working_results + persistent_results:
            if memory.id not in all_results:
                all_results[memory.id] = memory
        
        # Sort by relevance (importance + recency)
        results = list(all_results.values())
        results.sort(
            key=lambda m: m.importance * np.exp(-(time.time() - m.last_accessed) / 3600),
            reverse=True
        )
        
        self.stats["memories_retrieved"] += len(results)
        
        return results[:k]
    
    def forget(self, memory_id: str) -> bool:
        """Remove a memory"""
        memory = self.store.retrieve(memory_id)
        if memory:
            # Remove from working memory
            self.working_memory = deque(
                [m for m in self.working_memory if m.id != memory_id],
                maxlen=self.working_memory_size
            )
            
            # Note: In a real implementation, you'd remove from persistent storage
            logger.debug(f"Forgot memory: {memory_id}")
            return True
        
        return False
    
    def consolidate_context(self, context_window: List[str]) -> str:
        """Consolidate a context window into compressed form"""
        if not context_window:
            return ""
        
        # Simple consolidation: extract key themes and summarize
        word_freq = defaultdict(int)
        for text in context_window:
            for word in text.lower().split():
                if len(word) > 3:  # Skip short words
                    word_freq[word] += 1
        
        # Get top keywords
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        keywords = [word for word, _ in top_words]
        
        # Create summary
        summary = f"Key themes: {', '.join(keywords)}. "
        summary += f"Context length: {len(context_window)} items. "
        summary += f"Most frequent: {top_words[0][0] if top_words else 'N/A'}"
        
        return summary
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        memory_counts = defaultdict(int)
        total_importance = 0.0
        
        for memory_type in MemoryType:
            memories = self.store.get_by_type(memory_type)
            memory_counts[memory_type.value] = len(memories)
            total_importance += sum(m.importance for m in memories)
        
        return {
            "total_memories": sum(memory_counts.values()),
            "working_memory_size": len(self.working_memory),
            "memory_by_type": dict(memory_counts),
            "average_importance": total_importance / max(sum(memory_counts.values()), 1),
            "stats": self.stats
        }
    
    def _generate_memory_id(self, content: Any) -> str:
        """Generate unique memory ID"""
        content_str = json.dumps(content, sort_keys=True, default=str)
        timestamp = str(time.time())
        combined = content_str + timestamp
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _matches_query(self, memory: MemoryItem, query: str) -> bool:
        """Check if memory matches query"""
        content_str = str(memory.content).lower()
        query_lower = query.lower()
        
        # Check content
        if query_lower in content_str:
            return True
        
        # Check tags
        for tag in memory.tags:
            if query_lower in tag.lower():
                return True
        
        return False
    
    def _consolidate_memories(self):
        """Consolidate and optimize memories"""
        logger.info("Starting memory consolidation")
        
        # Apply importance decay to old memories
        for memory_type in MemoryType:
            memories = self.store.get_by_type(memory_type)
            for memory in memories:
                memory.decay_importance()
                self.store.store(memory)  # Update with decayed importance
        
        self.stats["consolidations"] += 1
        self._last_consolidation = time.time()
        
        logger.info("Memory consolidation complete")


class AgentWithMemory:
    """Agent enhanced with memory capabilities"""
    
    def __init__(
        self,
        name: str,
        llm_config: Optional[Dict] = None,
        memory_store: Optional[MemoryStore] = None
    ):
        self.name = name
        self.llm_config = llm_config or {
            "config_list": [
                {
                    "model": "gpt-3.5-turbo",
                    "api_key": "dummy_key_for_demonstration",
                }
            ],
            "temperature": 0.7,
        }
        
        # Initialize memory system
        self.memory_manager = MemoryManager(
            store=memory_store or InMemoryStore(),
            working_memory_size=15,
            consolidation_threshold=50
        )
        
        # Create Autogen agent
        self.agent = autogen.AssistantAgent(
            name=name,
            llm_config=self.llm_config,
            system_message=f"""You are {name}, an AI assistant with advanced memory capabilities.
            
Your memory system allows you to:
1. Remember important information from conversations
2. Recall relevant context from past interactions  
3. Learn and adapt from experience
4. Maintain long-term knowledge

Use your memory to provide more personalized and contextual responses."""
        )
        
        self.conversation_history = []
    
    def process_with_memory(self, message: str, context: Dict[str, Any] = None) -> str:
        """Process message using memory-enhanced reasoning"""
        # Remember the input
        input_memory_id = self.memory_manager.remember(
            content={"type": "user_input", "message": message},
            memory_type=MemoryType.EPISODIC,
            importance=0.7,
            tags={"user_input", "conversation"},
            context=context or {}
        )
        
        # Recall relevant memories
        relevant_memories = self.memory_manager.recall(
            query=message,
            k=5
        )
        
        # Prepare context with memories
        memory_context = ""
        if relevant_memories:
            memory_context = "Relevant memories:\n"
            for i, memory in enumerate(relevant_memories):
                memory_context += f"{i+1}. {memory.content} (importance: {memory.importance:.2f})\n"
        
        # Generate response with memory context
        full_message = f"{memory_context}\n\nCurrent message: {message}"
        
        response = self.agent.generate_reply(
            messages=[{"role": "user", "content": full_message}]
        )
        
        # Remember the response
        self.memory_manager.remember(
            content={"type": "agent_response", "response": response},
            memory_type=MemoryType.EPISODIC,
            importance=0.6,
            tags={"agent_response", "conversation"}
        )
        
        # Store conversation turn
        self.conversation_history.append({
            "input": message,
            "output": response,
            "memories_recalled": len(relevant_memories),
            "timestamp": time.time()
        })
        
        return response
    
    def learn_fact(self, fact: str, importance: float = 0.8, tags: Set[str] = None):
        """Explicitly learn a fact"""
        self.memory_manager.remember(
            content={"type": "fact", "fact": fact},
            memory_type=MemoryType.SEMANTIC,
            importance=importance,
            tags=tags or {"fact", "learned"}
        )
        logger.info(f"{self.name} learned fact: {fact}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of agent's memory state"""
        stats = self.memory_manager.get_memory_stats()
        
        return {
            "agent_name": self.name,
            "conversation_turns": len(self.conversation_history),
            "memory_stats": stats,
            "last_consolidation": self.memory_manager._last_consolidation
        }


def run_memory_examples():
    """Demonstrate memory management patterns"""
    logger.info("Starting Memory Management Pattern Examples")
    
    logger.info("\n=== Basic Memory Store Example ===")
    
    # Test in-memory store
    memory_store = InMemoryStore(max_size=5)
    
    # Store some memories
    for i in range(7):  # More than max_size to test eviction
        memory = MemoryItem(
            id=f"mem_{i}",
            content=f"This is memory content {i}",
            memory_type=MemoryType.EPISODIC,
            importance=0.5 + (i * 0.1),
            tags={f"tag_{i}", "test"}
        )
        memory_store.store(memory)
    
    logger.info(f"Stored 7 memories, actual count: {len(memory_store.memories)}")
    
    # Search memories
    results = memory_store.search("memory content", k=3)
    logger.info(f"Search results: {len(results)} memories found")
    
    logger.info("\n=== SQLite Memory Store Example ===")
    
    # Test SQLite store
    sqlite_store = SQLiteStore("test_memory.db")
    
    # Store different types of memories
    memory_types = [
        (MemoryType.SEMANTIC, "Python is a programming language", {"programming", "python"}),
        (MemoryType.EPISODIC, "User asked about machine learning", {"user", "ml"}),
        (MemoryType.PROCEDURAL, "To sort a list, use sorted() function", {"programming", "procedure"})
    ]
    
    for mem_type, content, tags in memory_types:
        memory = MemoryItem(
            id=f"sql_{mem_type.value}",
            content=content,
            memory_type=mem_type,
            importance=0.8,
            tags=tags
        )
        sqlite_store.store(memory)
    
    # Test retrieval
    semantic_memories = sqlite_store.get_by_type(MemoryType.SEMANTIC)
    logger.info(f"Semantic memories: {len(semantic_memories)}")
    
    logger.info("\n=== Memory Manager Example ===")
    
    # Test memory manager with consolidation
    manager = MemoryManager(
        store=InMemoryStore(max_size=20),
        consolidation_threshold=5
    )
    
    # Remember various things
    facts = [
        "The capital of France is Paris",
        "Machine learning uses algorithms to learn patterns",
        "Python was created by Guido van Rossum",
        "Neural networks are inspired by the brain",
        "Data science combines statistics and programming"
    ]
    
    for fact in facts:
        manager.remember(
            content=fact,
            memory_type=MemoryType.SEMANTIC,
            importance=0.9,
            tags={"fact", "knowledge"}
        )
    
    # Test recall
    ml_memories = manager.recall("machine learning", k=3)
    logger.info(f"ML-related memories: {len(ml_memories)}")
    
    for memory in ml_memories:
        logger.info(f"  - {memory.content} (importance: {memory.importance:.2f})")
    
    # Check statistics
    stats = manager.get_memory_stats()
    logger.info(f"Memory statistics: {stats}")
    
    logger.info("\n=== Agent with Memory Example ===")
    
    # Create memory-enhanced agent
    memory_agent = AgentWithMemory("memory_assistant")
    
    # Have a conversation with memory
    responses = []
    
    # First interaction
    resp1 = memory_agent.process_with_memory(
        "My name is Alice and I work in data science"
    )
    responses.append(resp1)
    
    # Teach the agent something
    memory_agent.learn_fact(
        "Alice prefers Python over R for data analysis",
        importance=0.9,
        tags={"alice", "preference", "programming"}
    )
    
    # Second interaction - should remember Alice
    resp2 = memory_agent.process_with_memory(
        "What programming language should I use for my project?"
    )
    responses.append(resp2)
    
    # Third interaction - test context
    resp3 = memory_agent.process_with_memory(
        "Tell me about my programming preferences"
    )
    responses.append(resp3)
    
    logger.info(f"Conversation responses generated: {len(responses)}")
    
    # Check agent memory summary
    summary = memory_agent.get_memory_summary()
    logger.info(f"Agent memory summary: {summary}")
    
    logger.info("\n=== Context Consolidation Example ===")
    
    # Test context consolidation
    long_context = [
        "User is working on a machine learning project",
        "They need help with data preprocessing",
        "The dataset contains customer information", 
        "Goal is to predict customer churn",
        "They prefer using Python and scikit-learn",
        "Previous experience with pandas and numpy",
        "Timeline is 2 weeks for completion"
    ]
    
    consolidated = manager.consolidate_context(long_context)
    logger.info(f"Consolidated context: {consolidated}")
    
    # Store consolidated memory
    manager.remember(
        content=consolidated,
        memory_type=MemoryType.CONTEXTUAL,
        importance=0.8,
        tags={"context", "project", "ml"}
    )
    
    logger.info("\nMemory Management Pattern Examples Complete")


if __name__ == "__main__":
    run_memory_examples()