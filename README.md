# Agentic Workflow Learning Repository

A comprehensive implementation of multi-agent AI system design patterns using **AutoGen v0.7.1**, CrewAI, and GPT-4o-mini.

## Quick Start

1. **Setup Environment**
   ```bash
   # AutoGen v0.7.1 (Latest - Recommended)
   pip install -r requirements-minimal.txt
   
   # Legacy: AutoGen 0.2 (for compatibility with old examples)
   pip install -r requirements-v02.txt
   
   # Setup environment
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

2. **Test Installation**
   ```bash
   # Test AutoGen v0.7.1 setup
   python basic_setup_test_v071.py
   
   # Test AutoGen 0.2 setup (legacy)
   python basic_setup_test.py
   ```

3. **Run Example Patterns (AutoGen v0.7.1)**
   ```bash
   # Sequential workflow (Research → Analysis → Writing → Review)
   python patterns/sequential_workflow_v071.py
   
   # Concurrent analysis (Market, Technical, Financial, Risk in parallel)
   python patterns/concurrent_agents_v071.py
   
   # Group chat discussion (RoundRobin and Selector teams)
   python patterns/group_chat_v071.py
   ```

## AutoGen v0.7.1 Features

🚀 **Latest Architecture Benefits:**
- **Async-first design** - All operations use modern async/await patterns
- **Model client architecture** - Clean separation of model providers and agents
- **Team-based collaboration** - RoundRobinGroupChat and SelectorGroupChat
- **Advanced termination conditions** - Flexible conversation control
- **Streaming support** - Real-time message processing
- **Multi-model support** - OpenAI, Anthropic, Azure, Ollama integration

## Learning Path

📚 **See [PLAN.md](PLAN.md)** for the complete 14-week learning plan

### Implemented Patterns (v0.7.1)

✅ **Phase 1: Foundation**
- [x] AutoGen v0.7.1 setup with GPT-4o-mini
- [x] Async model client configuration
- [x] Modern logging and monitoring utilities
- [x] Cost tracking and optimization

✅ **Phase 2: Core Patterns (v0.7.1)**
- [x] **Sequential Workflow Pattern** - Async linear task dependencies with checkpointing
- [x] **Concurrent Agents Pattern** - Parallel execution with asyncio.gather optimization
- [x] **Group Chat Pattern** - Team-based collaboration (RoundRobin & Selector)

🚧 **Coming Next**
- [ ] Handoffs Pattern - Dynamic task routing
- [ ] Mixture of Agents (MoA) - Layered processing approach
- [ ] Multi-Agent Debate - Structured argumentation
- [ ] Reflection Pattern - Self-improvement cycles
- [ ] CrewAI Integration - Advanced workflow orchestration

## Key Features

### 🏗️ Production-Ready Architecture
- Comprehensive error handling and recovery
- Checkpointing for long-running workflows
- Cost tracking and optimization
- Detailed logging and monitoring

### ⚡ Performance Optimization
- Concurrent execution with 2-4x speedup
- Token usage optimization strategies
- Timeout handling and circuit breakers
- Memory-efficient state management

### 🔍 Advanced Debugging
- Conversation tracking and analysis
- Agent decision logging
- Performance metrics collection
- Visual workflow monitoring

### 💰 Cost Management
- Token usage tracking per agent
- Daily cost limits and alerts
- Model selection optimization
- Resource usage analytics

## Directory Structure

```
agentic/
├── PLAN.md                 # 14-week learning plan
├── config.py              # Central configuration
├── basic_setup_test.py     # Environment validation
├── patterns/               # Agent pattern implementations
│   ├── sequential_workflow.py
│   ├── concurrent_agents.py
│   ├── group_chat.py
│   └── ...
├── utils/                  # Utilities and helpers
│   └── logging_utils.py
├── logs/                   # Execution logs and checkpoints
├── outputs/                # Generated results and reports
└── requirements.txt        # Dependencies
```

## Example Use Cases

### 📊 Sequential Workflow
**Research Paper Analysis Pipeline**
- Researcher gathers information
- Analyst structures and analyzes
- Writer creates comprehensive report
- Reviewer provides quality assessment

### ⚡ Concurrent Analysis  
**Multi-Perspective Business Analysis**
- Market analyst (competition, opportunities)
- Technical analyst (feasibility, architecture)
- Financial analyst (ROI, costs)
- Risk analyst (threats, mitigation)

### 🗣️ Group Chat Discussion
**Product Development Team**
- Product Manager (requirements)
- Software Engineer (implementation)
- Data Scientist (ML/AI aspects)
- DevOps Engineer (infrastructure)
- QA Tester (quality assurance)

## Performance Benchmarks

Based on research findings and implementations:

| Pattern | Speedup vs Sequential | Token Usage | Best Use Case |
|---------|----------------------|-------------|---------------|
| Concurrent | 2-4x faster | 15x increase | Independent parallel tasks |
| Group Chat | Variable | 10-20x increase | Collaborative decision making |
| Sequential | Baseline | 5-10x increase | Dependent task chains |

## Getting Started Examples

### Simple Sequential Analysis
```python
from patterns.sequential_workflow import SequentialResearchWorkflow

workflow = SequentialResearchWorkflow()
results = workflow.execute_workflow("AI Ethics in Healthcare")
```

### Concurrent Business Analysis
```python  
from patterns.concurrent_agents import ConcurrentBusinessAnalysis

analyzer = ConcurrentBusinessAnalysis()
results = analyzer.run_full_analysis("Launch AI-powered customer service platform")
```

### Team Collaboration Discussion
```python
from patterns.group_chat import GroupChatWorkflow

team_chat = GroupChatWorkflow()
results = team_chat.facilitate_discussion("Build code review AI assistant")
```

## Research Foundation

This implementation is based on the latest research in multi-agent AI systems:
- **90.2% performance improvements** over single agents
- **15x token usage increase** - requires careful cost management
- **Mixture of Agents** achieving 65.1% on AlpacaEval 2.0
- Production deployments at Anthropic, Microsoft, and others

## Cost Considerations

- **GPT-4o-mini** recommended for learning (10x cheaper than GPT-4)
- Set daily cost limits in configuration
- Monitor token usage per pattern
- Consider batch processing for efficiency

## Next Steps

1. Complete the remaining patterns (Handoffs, MoA, Debate, Reflection)
2. Add CrewAI integration examples
3. Implement communication protocols (MCP, ACP)
4. Build real-world production examples
5. Add advanced debugging and monitoring tools

## Contributing

Follow the PLAN.md learning path and contribute implementations of:
- Advanced agent patterns
- Communication protocols  
- Production deployment examples
- Performance optimizations
- Domain-specific applications

---

🚀 **Ready to learn agentic workflows?** Start with `python basic_setup_test.py` and follow the PLAN.md!