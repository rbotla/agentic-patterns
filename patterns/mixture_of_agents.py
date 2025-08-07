"""
Mixture of Agents (MoA) Pattern Implementation
Leverages the "collaborativeness of LLMs" phenomenon where models generate 
better responses when provided outputs from other models
Uses layered processing architecture
"""

import autogen
from typing import Dict, Any, Optional, List, Tuple
import logging
from utils.logging_utils import setup_logging
import time
import json
from dataclasses import dataclass
from enum import Enum

logger = setup_logging(__name__)


class LayerType(Enum):
    """Types of processing layers"""
    INITIAL_GENERATION = "initial_generation"
    INTERMEDIATE_PROCESSING = "intermediate_processing"
    FINAL_AGGREGATION = "final_aggregation"


@dataclass
class LayerOutput:
    """Output from a processing layer"""
    layer_number: int
    layer_type: LayerType
    agent_outputs: List[Dict[str, str]]
    processing_time: float
    token_count: int
    
    def get_combined_output(self) -> str:
        """Combine all agent outputs from this layer"""
        outputs = []
        for output in self.agent_outputs:
            outputs.append(f"[{output['agent']}]: {output['response']}")
        return "\n\n".join(outputs)


class MixtureOfAgentsWorkflow:
    """
    Implements the MoA pattern with multi-layer proposer-aggregator architecture
    Based on research showing 65.1% on AlpacaEval 2.0 vs GPT-4 Omni's 57.5%
    """
    
    def __init__(self, llm_config: Optional[Dict] = None, use_lite_mode: bool = False):
        self.llm_config = llm_config or {
            "config_list": [
                {
                    "model": "gpt-3.5-turbo",
                    "api_key": "dummy_key_for_demonstration",
                }
            ],
            "temperature": 0.7,
        }
        
        self.use_lite_mode = use_lite_mode
        self.layer_outputs = []
        self.total_tokens_used = 0
        
        if use_lite_mode:
            self._initialize_moa_lite()
        else:
            self._initialize_full_moa()
    
    def _initialize_full_moa(self):
        """Initialize full MoA with multiple diverse models"""
        self.layers = [
            {
                "layer_number": 1,
                "layer_type": LayerType.INITIAL_GENERATION,
                "agents": self._create_proposer_agents([
                    ("qwen_agent", "You are Qwen, focusing on detailed analytical responses."),
                    ("wizard_agent", "You are Wizard, focusing on creative and comprehensive solutions."),
                    ("llama_agent", "You are Llama, focusing on practical and efficient approaches."),
                    ("claude_agent", "You are Claude, focusing on nuanced and thoughtful analysis."),
                    ("gpt_agent", "You are GPT, focusing on versatile and well-rounded responses."),
                    ("gemini_agent", "You are Gemini, focusing on innovative and cutting-edge solutions.")
                ])
            },
            {
                "layer_number": 2,
                "layer_type": LayerType.INTERMEDIATE_PROCESSING,
                "agents": self._create_synthesis_agents([
                    ("synthesis_agent_1", "You synthesize multiple perspectives into coherent insights."),
                    ("synthesis_agent_2", "You identify patterns and extract key themes from multiple inputs."),
                    ("synthesis_agent_3", "You critically evaluate and refine collective outputs.")
                ])
            },
            {
                "layer_number": 3,
                "layer_type": LayerType.FINAL_AGGREGATION,
                "agents": [self._create_aggregator_agent()]
            }
        ]
    
    def _initialize_moa_lite(self):
        """Initialize MoA-Lite for 2x cost efficiency"""
        self.layers = [
            {
                "layer_number": 1,
                "layer_type": LayerType.INITIAL_GENERATION,
                "agents": self._create_proposer_agents([
                    ("proposer_1", "You provide comprehensive and analytical responses."),
                    ("proposer_2", "You provide creative and innovative solutions."),
                    ("proposer_3", "You provide practical and efficient approaches.")
                ])
            },
            {
                "layer_number": 2,
                "layer_type": LayerType.FINAL_AGGREGATION,
                "agents": [self._create_aggregator_agent()]
            }
        ]
    
    def _create_proposer_agents(self, agent_configs: List[Tuple[str, str]]) -> List[autogen.AssistantAgent]:
        """Create proposer agents with diverse perspectives"""
        agents = []
        for name, system_message in agent_configs:
            agent = autogen.AssistantAgent(
                name=name,
                llm_config=self.llm_config,
                system_message=f"""{system_message}
                
Your role as a proposer:
1. Generate comprehensive initial responses
2. Consider multiple perspectives
3. Provide detailed reasoning
4. Include relevant examples when appropriate
5. Structure your response clearly"""
            )
            agents.append(agent)
        return agents
    
    def _create_synthesis_agents(self, agent_configs: List[Tuple[str, str]]) -> List[autogen.AssistantAgent]:
        """Create intermediate synthesis agents"""
        agents = []
        for name, system_message in agent_configs:
            agent = autogen.AssistantAgent(
                name=name,
                llm_config=self.llm_config,
                system_message=f"""{system_message}
                
Your role as a synthesizer:
1. Analyze outputs from previous layer agents
2. Identify strengths in each response
3. Combine complementary insights
4. Resolve contradictions thoughtfully
5. Enhance overall quality and coherence"""
            )
            agents.append(agent)
        return agents
    
    def _create_aggregator_agent(self) -> autogen.AssistantAgent:
        """Create final aggregator agent"""
        return autogen.AssistantAgent(
            name="final_aggregator",
            llm_config=self.llm_config,
            system_message="""You are the final aggregator in a Mixture of Agents system.

Your role:
1. Synthesize all previous layer outputs into a single, high-quality response
2. Leverage the collaborative knowledge from all contributing agents
3. Ensure the final output is:
   - Comprehensive yet concise
   - Well-structured and coherent
   - Factually accurate (based on consensus)
   - Free of redundancy
   - Superior to any individual agent's response
4. Preserve the best insights from each contributor
5. Present a unified, polished final answer"""
        )
    
    def process_through_layers(self, input_prompt: str) -> Dict[str, Any]:
        """Process input through all MoA layers"""
        logger.info(f"Starting MoA processing for: {input_prompt[:100]}...")
        start_time = time.time()
        
        self.layer_outputs = []
        current_input = input_prompt
        
        for layer_config in self.layers:
            layer_output = self._process_layer(
                current_input,
                layer_config,
                previous_outputs=self.layer_outputs
            )
            self.layer_outputs.append(layer_output)
            
            current_input = self._prepare_next_layer_input(
                input_prompt,
                layer_output
            )
            
            logger.info(f"Layer {layer_config['layer_number']} complete - "
                       f"Type: {layer_config['layer_type'].value}")
        
        elapsed_time = time.time() - start_time
        
        final_response = self.layer_outputs[-1].agent_outputs[0]["response"]
        
        result = {
            "input_prompt": input_prompt,
            "final_response": final_response,
            "layer_outputs": [self._layer_output_to_dict(lo) for lo in self.layer_outputs],
            "total_layers": len(self.layers),
            "total_agents_used": sum(len(layer["agents"]) for layer in self.layers),
            "execution_time": elapsed_time,
            "total_tokens": self.total_tokens_used,
            "mode": "lite" if self.use_lite_mode else "full"
        }
        
        logger.info(f"MoA processing complete in {elapsed_time:.2f}s")
        return result
    
    def _process_layer(
        self, 
        input_text: str, 
        layer_config: Dict,
        previous_outputs: List[LayerOutput]
    ) -> LayerOutput:
        """Process a single layer"""
        layer_start = time.time()
        agent_outputs = []
        layer_tokens = 0
        
        for agent in layer_config["agents"]:
            if layer_config["layer_type"] == LayerType.INITIAL_GENERATION:
                message = input_text
            else:
                message = self._construct_layer_message(
                    input_text,
                    previous_outputs
                )
            
            response = agent.generate_reply(
                messages=[{"role": "user", "content": message}]
            )
            
            agent_outputs.append({
                "agent": agent.name,
                "response": response
            })
            
            layer_tokens += len(response.split()) * 1.3
        
        self.total_tokens_used += layer_tokens
        
        return LayerOutput(
            layer_number=layer_config["layer_number"],
            layer_type=layer_config["layer_type"],
            agent_outputs=agent_outputs,
            processing_time=time.time() - layer_start,
            token_count=int(layer_tokens)
        )
    
    def _construct_layer_message(
        self, 
        original_prompt: str,
        previous_outputs: List[LayerOutput]
    ) -> str:
        """Construct message for non-initial layers"""
        message = f"Original task: {original_prompt}\n\n"
        message += "Previous layer outputs to consider:\n\n"
        
        for output in previous_outputs:
            message += f"=== Layer {output.layer_number} ({output.layer_type.value}) ===\n"
            message += output.get_combined_output()
            message += "\n\n"
        
        message += "Based on all previous outputs, provide your response:"
        return message
    
    def _prepare_next_layer_input(
        self, 
        original_prompt: str,
        current_output: LayerOutput
    ) -> str:
        """Prepare input for next layer"""
        return self._construct_layer_message(original_prompt, [current_output])
    
    def _layer_output_to_dict(self, layer_output: LayerOutput) -> Dict:
        """Convert LayerOutput to dictionary"""
        return {
            "layer_number": layer_output.layer_number,
            "layer_type": layer_output.layer_type.value,
            "agent_count": len(layer_output.agent_outputs),
            "processing_time": layer_output.processing_time,
            "token_count": layer_output.token_count,
            "outputs": layer_output.agent_outputs
        }


class AdaptiveMoA(MixtureOfAgentsWorkflow):
    """Adaptive MoA that adjusts layers based on task complexity"""
    
    def __init__(self, llm_config: Optional[Dict] = None):
        super().__init__(llm_config, use_lite_mode=False)
        self.complexity_analyzer = self._create_complexity_analyzer()
    
    def _create_complexity_analyzer(self) -> autogen.AssistantAgent:
        """Create agent to analyze task complexity"""
        return autogen.AssistantAgent(
            name="complexity_analyzer",
            llm_config=self.llm_config,
            system_message="""Analyze task complexity and recommend MoA configuration.

Respond with:
COMPLEXITY: [low/medium/high]
RECOMMENDED_LAYERS: [2/3/4]
RECOMMENDED_AGENTS_PER_LAYER: [2-6]
REASONING: [explanation]"""
        )
    
    def adaptive_process(self, input_prompt: str) -> Dict[str, Any]:
        """Process with adaptive layer configuration"""
        complexity = self._analyze_complexity(input_prompt)
        logger.info(f"Task complexity: {complexity['complexity']}")
        
        self._adjust_layers_for_complexity(complexity)
        
        result = self.process_through_layers(input_prompt)
        result["complexity_analysis"] = complexity
        
        return result
    
    def _analyze_complexity(self, prompt: str) -> Dict[str, str]:
        """Analyze prompt complexity"""
        response = self.complexity_analyzer.generate_reply(
            messages=[{"role": "user", "content": f"Analyze complexity: {prompt}"}]
        )
        
        complexity = {"complexity": "medium", "layers": 3, "agents": 4}
        
        lines = response.split('\n')
        for line in lines:
            if "COMPLEXITY:" in line:
                complexity["complexity"] = line.split(":")[-1].strip().lower()
            elif "RECOMMENDED_LAYERS:" in line:
                try:
                    complexity["layers"] = int(line.split(":")[-1].strip())
                except:
                    pass
            elif "RECOMMENDED_AGENTS_PER_LAYER:" in line:
                try:
                    agents_str = line.split(":")[-1].strip()
                    complexity["agents"] = int(agents_str.split("-")[0])
                except:
                    pass
        
        return complexity
    
    def _adjust_layers_for_complexity(self, complexity: Dict):
        """Adjust layer configuration based on complexity"""
        if complexity["complexity"] == "low":
            self.layers = self.layers[:2]
        elif complexity["complexity"] == "high":
            if len(self.layers) < 4:
                self.layers.insert(2, {
                    "layer_number": 3,
                    "layer_type": LayerType.INTERMEDIATE_PROCESSING,
                    "agents": self._create_synthesis_agents([
                        ("deep_synthesis", "You perform deep synthesis of complex outputs.")
                    ])
                })


def run_moa_examples():
    """Demonstrate Mixture of Agents pattern"""
    logger.info("Starting Mixture of Agents Pattern Examples")
    
    logger.info("\n=== Basic MoA Example ===")
    
    moa_workflow = MixtureOfAgentsWorkflow(use_lite_mode=False)
    
    prompt1 = "Explain the benefits and challenges of renewable energy adoption"
    
    result1 = moa_workflow.process_through_layers(prompt1)
    
    logger.info(f"\nPrompt: {prompt1}")
    logger.info(f"Layers processed: {result1['total_layers']}")
    logger.info(f"Total agents used: {result1['total_agents_used']}")
    logger.info(f"Execution time: {result1['execution_time']:.2f}s")
    logger.info(f"Total tokens: {result1['total_tokens']}")
    
    logger.info("\n=== MoA-Lite Example ===")
    
    moa_lite = MixtureOfAgentsWorkflow(use_lite_mode=True)
    
    prompt2 = "Write a brief analysis of artificial intelligence impact on education"
    
    result2 = moa_lite.process_through_layers(prompt2)
    
    logger.info(f"\nMoA-Lite Prompt: {prompt2}")
    logger.info(f"Layers: {result2['total_layers']} (Lite mode)")
    logger.info(f"Agents: {result2['total_agents_used']}")
    logger.info(f"Time: {result2['execution_time']:.2f}s")
    logger.info(f"Token efficiency: {result2['total_tokens']} tokens")
    
    logger.info("\n=== Adaptive MoA Example ===")
    
    adaptive_moa = AdaptiveMoA()
    
    simple_prompt = "What is 2 + 2?"
    complex_prompt = "Design a distributed system architecture for a global e-commerce platform handling millions of transactions"
    
    simple_result = adaptive_moa.adaptive_process(simple_prompt)
    complex_result = adaptive_moa.adaptive_process(complex_prompt)
    
    logger.info(f"\nSimple task complexity: {simple_result['complexity_analysis']['complexity']}")
    logger.info(f"Layers used: {simple_result['total_layers']}")
    
    logger.info(f"\nComplex task complexity: {complex_result['complexity_analysis']['complexity']}")
    logger.info(f"Layers used: {complex_result['total_layers']}")
    
    logger.info("\n=== Layer Output Analysis ===")
    
    for i, layer in enumerate(result1['layer_outputs']):
        logger.info(f"\nLayer {layer['layer_number']} ({layer['layer_type']}):")
        logger.info(f"  Agents: {layer['agent_count']}")
        logger.info(f"  Processing time: {layer['processing_time']:.2f}s")
        logger.info(f"  Tokens: {layer['token_count']}")
    
    logger.info("\nMixture of Agents Pattern Examples Complete")


if __name__ == "__main__":
    run_moa_examples()