#!/usr/bin/env python3
"""
Basic setup test to verify AutoGen and GPT-4o-mini integration
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config
from utils.logging_utils import setup_logging

def test_configuration():
    """Test that configuration is properly loaded"""
    logger = setup_logging("setup_test")
    
    logger.info("Testing configuration...")
    
    if not Config.validate_config():
        logger.error("Configuration validation failed")
        return False
    
    logger.info(f"Using model: {Config.OPENAI_MODEL}")
    logger.info(f"Cost tracking enabled: {Config.ENABLE_COST_TRACKING}")
    logger.info(f"Max daily cost: ${Config.MAX_DAILY_COST}")
    
    return True

def test_autogen_import():
    """Test AutoGen import and basic functionality"""
    logger = setup_logging("setup_test")
    
    try:
        import autogen
        logger.info(f"AutoGen version: {autogen.__version__}")
        
        # Test basic agent creation
        llm_config = Config.get_llm_config()
        
        assistant = autogen.AssistantAgent(
            name="test_assistant",
            llm_config=llm_config,
            system_message="You are a helpful AI assistant for testing purposes."
        )
        
        user_proxy = autogen.UserProxyAgent(
            name="test_user",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False
        )
        
        logger.info("Successfully created AutoGen agents")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import AutoGen: {e}")
        return False
    except Exception as e:
        logger.error(f"Error creating AutoGen agents: {e}")
        return False

def test_simple_interaction():
    """Test a simple agent interaction"""
    logger = setup_logging("setup_test")
    
    try:
        import autogen
        
        llm_config = Config.get_llm_config()
        
        assistant = autogen.AssistantAgent(
            name="math_assistant",
            llm_config=llm_config,
            system_message="You are a math tutor. Solve problems step by step."
        )
        
        user_proxy = autogen.UserProxyAgent(
            name="student",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False
        )
        
        logger.info("Starting simple math interaction...")
        
        # Test interaction
        user_proxy.initiate_chat(
            assistant, 
            message="What is 15 + 27? Please show your work."
        )
        
        logger.info("Simple interaction completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in simple interaction: {e}")
        return False

def main():
    """Run all setup tests"""
    logger = setup_logging("setup_test")
    
    logger.info("=== AutoGen Setup Test ===")
    
    tests = [
        ("Configuration", test_configuration),
        ("AutoGen Import", test_autogen_import),
        ("Simple Interaction", test_simple_interaction)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        results[test_name] = test_func()
        
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
        logger.info("🎉 All tests passed! AutoGen setup is ready.")
        print("\n✅ Setup complete! You can now run the example workflows.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the configuration.")
        print("\n❌ Setup incomplete. Please check the logs and fix any issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)