#!/usr/bin/env python3
"""
Basic setup test to verify AutoGen v0.7.1 and GPT-4o-mini integration
"""

import sys
import asyncio
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config
from utils.logging_utils import setup_logging

async def test_configuration():
    """Test that configuration is properly loaded"""
    logger = setup_logging("setup_test_v071")
    
    logger.info("Testing configuration...")
    
    if not Config.validate_config():
        logger.error("Configuration validation failed")
        return False
    
    logger.info(f"Using model: {Config.OPENAI_MODEL}")
    logger.info(f"Cost tracking enabled: {Config.ENABLE_COST_TRACKING}")
    logger.info(f"Max daily cost: ${Config.MAX_DAILY_COST}")
    
    return True

async def test_autogen_import():
    """Test AutoGen v0.7.1 import and basic functionality"""
    logger = setup_logging("setup_test_v071")
    
    try:
        # Test core imports
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.messages import TextMessage
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        logger.info("✅ AutoGen v0.7.1 imports successful")
        
        # Test model client creation
        model_client = Config.get_model_client()
        logger.info("✅ Model client created successfully")
        
        # Test basic agent creation
        assistant = AssistantAgent(
            name="test_assistant",
            model_client=model_client,
            system_message="You are a helpful AI assistant for testing purposes."
        )
        
        logger.info("✅ Successfully created AutoGen v0.7.1 agents")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import AutoGen v0.7.1: {e}")
        logger.error("Please install: pip install -r requirements-minimal.txt")
        return False
    except Exception as e:
        logger.error(f"Error creating AutoGen v0.7.1 agents: {e}")
        return False

async def test_simple_interaction():
    """Test a simple agent interaction"""
    logger = setup_logging("setup_test_v071")
    
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.messages import TextMessage
        
        model_client = Config.get_model_client()
        
        assistant = AssistantAgent(
            name="math_assistant",
            model_client=model_client,
            system_message="You are a math tutor. Solve problems step by step."
        )
        
        logger.info("Starting simple math interaction...")
        
        # Test interaction using v0.7.1 async pattern
        from autogen_core import CancellationToken
        message = TextMessage(content="What is 15 + 27? Please show your work.", source="user")
        cancellation_token = CancellationToken()
        response = await assistant.on_messages([message], cancellation_token)
        
        logger.info(f"Assistant response: {response.chat_message.content[:100]}...")
        logger.info("✅ Simple interaction completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in simple interaction: {e}")
        return False

async def test_team_functionality():
    """Test AutoGen v0.7.1 team functionality"""
    logger = setup_logging("setup_test_v071")
    
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_agentchat.messages import TextMessage
        from autogen_agentchat.conditions import MaxMessageTermination
        
        model_client = Config.get_model_client()
        
        # Create two simple agents
        agent1 = AssistantAgent(
            name="agent1",
            model_client=model_client,
            system_message="You are a helpful assistant. Keep responses brief."
        )
        
        agent2 = AssistantAgent(
            name="agent2", 
            model_client=model_client,
            system_message="You are a helpful assistant. Keep responses brief."
        )
        
        # Create a simple team
        team = RoundRobinGroupChat(
            participants=[agent1, agent2],
            termination_condition=MaxMessageTermination(max_messages=4)
        )
        
        logger.info("Testing team functionality...")
        
        # Test team interaction
        message = TextMessage(content="Hello team! Please introduce yourselves briefly.", source="user")
        result = await team.run(task=message)
        
        logger.info(f"Team completed with {len(result.messages)} messages")
        logger.info("✅ Team functionality test successful")
        return True
        
    except Exception as e:
        logger.error(f"Error in team functionality test: {e}")
        return False

async def main():
    """Run all setup tests"""
    logger = setup_logging("setup_test_v071")
    
    logger.info("=== AutoGen v0.7.1 Setup Test ===")
    
    tests = [
        ("Configuration", test_configuration),
        ("AutoGen v0.7.1 Import", test_autogen_import),
        ("Simple Interaction", test_simple_interaction),
        ("Team Functionality", test_team_functionality)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            results[test_name] = await test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
        
        if results[test_name]:
            logger.info(f"✅ {test_name} test PASSED")
        else:
            logger.error(f"❌ {test_name} test FAILED")
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    logger.info(f"\n=== Test Summary ===")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! AutoGen v0.7.1 setup is ready.")
        print("\n✅ Setup complete! You can now run the v0.7.1 example workflows.")
        print("\nTry these commands:")
        print("python patterns/sequential_workflow_v071.py")
        print("python patterns/concurrent_agents_v071.py") 
        print("python patterns/group_chat_v071.py")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the configuration.")
        print("\n❌ Setup incomplete. Please check the logs and fix any issues.")
        print("\nQuick fixes:")
        print("1. Install AutoGen v0.7.1: pip install -r requirements-minimal.txt")
        print("2. Configure API key: cp .env.example .env (then edit .env)")
        print("3. Check Python version: Requires Python 3.10+")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)