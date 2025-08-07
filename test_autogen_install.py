#!/usr/bin/env python3
"""
Simple test to verify AutoGen installation and imports
"""

print("🧪 Testing AutoGen Installation...")

# Test 1: Basic AutoGen import
try:
    import autogen
    print("✅ AutoGen imported successfully")
    print(f"   Version: {autogen.__version__}")
except ImportError as e:
    print(f"❌ AutoGen import failed: {e}")
    print("   Solution: pip install -r requirements-v02.txt")
    exit(1)

# Test 2: Core agent classes
try:
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    print("✅ Core agent classes imported successfully")
except ImportError as e:
    print(f"❌ Agent classes import failed: {e}")
    exit(1)

# Test 3: OpenAI import
try:
    import openai
    print(f"✅ OpenAI imported successfully (version: {openai.__version__})")
except ImportError as e:
    print(f"❌ OpenAI import failed: {e}")
    print("   Solution: pip install openai>=1.12.0")
    exit(1)

# Test 4: Configuration
try:
    from config import Config
    print("✅ Configuration module imported")
    
    if Config.OPENAI_API_KEY:
        print("✅ OpenAI API key found in configuration")
    else:
        print("⚠️  OpenAI API key not configured")
        print("   Solution: Copy .env.example to .env and add your API key")
except Exception as e:
    print(f"⚠️  Configuration issue: {e}")

# Test 5: Basic agent creation
try:
    # This should work without API calls
    test_config = {
        "config_list": [{"model": "gpt-4o-mini", "api_key": "test"}],
        "timeout": 60
    }
    
    assistant = autogen.AssistantAgent(
        name="test_assistant",
        llm_config=test_config,
        system_message="You are a test assistant."
    )
    
    user_proxy = autogen.UserProxyAgent(
        name="test_user",
        human_input_mode="NEVER",
        code_execution_config=False
    )
    
    print("✅ Basic agents created successfully")
    print("✅ AutoGen is ready to use!")
    
except Exception as e:
    print(f"❌ Agent creation failed: {e}")

print("\n🎉 Installation test completed!")
print("Next steps:")
print("1. Configure your OpenAI API key in .env file")
print("2. Run: python basic_setup_test.py")
print("3. Try: python patterns/sequential_workflow.py")