# Installation Guide

## Quick Setup

```bash
# 1. Clone or navigate to the repository
cd agentic

# 2. Install dependencies (choose one option)
pip install -r requirements-v02.txt     # AutoGen 0.2 (Recommended - stable)
# OR
pip install -r requirements-minimal.txt # AutoGen 0.4 (Latest - experimental)
# OR  
pip install -r requirements.txt         # AutoGen 0.4 Full setup

# 3. Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key

# 4. Test installation
python basic_setup_test.py
```

## AutoGen Version Guide

### AutoGen 0.2 (Recommended for Learning)
- **Package:** `pyautogen>=0.2.33`
- **Status:** Stable, well-documented
- **Import:** `import autogen`
- **Use Case:** Learning, stable production workloads
- **Install:** `pip install -r requirements-v02.txt`

### AutoGen 0.4 (Latest)
- **Package:** `autogen-agentchat>=0.4.0` + `autogen-ext[openai]>=0.4.0`
- **Status:** New architecture, breaking changes from 0.2
- **Import:** `from autogen_agentchat.agents import AssistantAgent`
- **Use Case:** Latest features, experimental
- **Install:** `pip install -r requirements-minimal.txt`

### Recommendation
**Start with AutoGen 0.2** (`requirements-v02.txt`) as all examples in this repository are built for the 0.2 API. AutoGen 0.4 has significant breaking changes and the examples would need to be updated.

## Package Details

### Core Dependencies (AutoGen 0.2)
- **pyautogen>=0.2.33** - Microsoft AutoGen framework
- **openai>=1.12.0** - OpenAI API client
- **python-dotenv>=1.0.0** - Environment variable management
- **pydantic>=2.5.0** - Data validation and settings

### Optional Dependencies
- **crewai>=0.28.8** - CrewAI framework for advanced workflows
- **langchain>=0.1.0** - LangChain for additional integrations
- **pandas, numpy** - Data processing utilities

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'autogen'
**Solution:** Use AutoGen 0.2 (recommended):
```bash
# Recommended approach - use stable AutoGen 0.2
pip install -r requirements-v02.txt

# This installs: pyautogen>=0.2.33
# Import with: import autogen
```

**Alternative:** If you need AutoGen 0.4:
```bash
pip install autogen-agentchat autogen-ext[openai]
# Import with: from autogen_agentchat.agents import AssistantAgent
# Note: All examples need to be updated for 0.4 API
```

#### 2. OpenAI API Key Issues
**Symptoms:** 
- "Authentication Error" 
- "Invalid API key"

**Solutions:**
```bash
# Check your .env file has:
OPENAI_API_KEY=sk-your-actual-api-key-here

# Verify API key is loaded:
python -c "from config import Config; print('API Key loaded:', bool(Config.OPENAI_API_KEY))"
```

#### 3. Version Conflicts
**Solution:** Use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-minimal.txt
```

#### 4. CrewAI Installation Issues
**Solution:** Skip CrewAI initially:
```bash
# Install core dependencies only
pip install pyautogen>=0.2.33 openai>=1.12.0 python-dotenv>=1.0.0 pydantic>=2.5.0

# Test core functionality first
python basic_setup_test.py
```

#### 5. M1/M2 Mac Issues
**Solution:** Some packages may need specific installation:
```bash
# If you encounter build issues
pip install --no-build-isolation pyautogen
```

### Installation Verification

Run this script to verify everything is working:

```python
# test_installation.py
try:
    import autogen
    print(f"✅ AutoGen {autogen.__version__} installed")
except ImportError as e:
    print(f"❌ AutoGen not found: {e}")

try:
    import openai
    print(f"✅ OpenAI {openai.__version__} installed")
except ImportError as e:
    print(f"❌ OpenAI not found: {e}")

try:
    from config import Config
    if Config.OPENAI_API_KEY:
        print("✅ OpenAI API key configured")
    else:
        print("❌ OpenAI API key not found - check .env file")
except Exception as e:
    print(f"❌ Configuration error: {e}")

print("\nRun 'python basic_setup_test.py' for full validation")
```

## Alternative Installation Methods

### Using Conda
```bash
conda create -n agentic python=3.9
conda activate agentic
pip install -r requirements-minimal.txt
```

### Using Poetry
```bash
# Create pyproject.toml first, then:
poetry install
poetry shell
```

### Docker Setup (Advanced)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements-minimal.txt .
RUN pip install -r requirements-minimal.txt
COPY . .
CMD ["python", "basic_setup_test.py"]
```

## Development Setup

For contributors and advanced users:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

## API Key Setup

### Getting OpenAI API Key
1. Visit https://platform.openai.com/api-keys
2. Create new API key
3. Copy the key (starts with `sk-`)
4. Add to `.env` file

### Cost Management
```env
# In .env file
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini  # Cheaper option for learning
MAX_DAILY_COST=10.00     # Safety limit
ENABLE_COST_TRACKING=True
```

### Testing API Connection
```bash
python -c "
from config import Config
import openai
client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'Hello!'}],
    max_tokens=10
)
print('✅ API connection successful:', response.choices[0].message.content)
"
```

## Next Steps

After successful installation:

1. **Run the setup test:** `python basic_setup_test.py`
2. **Try a simple pattern:** `python patterns/sequential_workflow.py`
3. **Follow the learning plan:** See `PLAN.md` for structured progression
4. **Check logs:** All execution logs are saved in `logs/` directory

## Getting Help

- **GitHub Issues:** Report problems at repository issues page
- **Documentation:** Check `README.md` and `PLAN.md` for guidance
- **Examples:** Run the pattern examples to see working code
- **Logs:** Check `logs/` directory for detailed error information