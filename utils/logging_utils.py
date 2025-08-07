import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from config import Config

def setup_logging(
    name: str = "agentic_workflow",
    log_file: Optional[str] = None,
    level: str = None
) -> logging.Logger:
    """Setup structured logging for agentic workflows"""
    
    level = level or Config.LOG_LEVEL
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if not log_file:
        log_file = f"logs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

class AgentLogger:
    """Specialized logger for agent interactions"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = setup_logging(f"agent_{agent_name}")
        self.conversation_log = []
    
    def log_message(self, sender: str, recipient: str, message: str, message_type: str = "chat"):
        """Log agent-to-agent messages"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "sender": sender,
            "recipient": recipient,
            "message": message,
            "type": message_type
        }
        
        self.conversation_log.append(log_entry)
        self.logger.info(f"{sender} -> {recipient}: {message[:100]}...")
    
    def log_tool_use(self, agent: str, tool: str, input_data: str, output: str):
        """Log tool usage by agents"""
        self.logger.info(f"{agent} used {tool} with input: {input_data[:50]}...")
        self.logger.debug(f"Tool output: {output}")
    
    def log_decision(self, agent: str, decision: str, reasoning: str):
        """Log agent decision-making"""
        self.logger.info(f"{agent} decided: {decision}")
        self.logger.debug(f"Reasoning: {reasoning}")
    
    def save_conversation(self, filename: Optional[str] = None):
        """Save conversation log to file"""
        if not filename:
            filename = f"logs/conversation_{self.agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(filename, 'w') as f:
            json.dump(self.conversation_log, f, indent=2)
        
        self.logger.info(f"Conversation saved to {filename}")