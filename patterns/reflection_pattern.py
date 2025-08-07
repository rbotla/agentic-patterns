"""
Reflection Pattern Implementation (Self-Improvement)
Self-directed improvement through iterative refinement
Generation → Self-Critique → Refinement → Iteration cycle
"""

import autogen
from typing import Dict, Any, Optional, List, Tuple
import logging
from utils.logging_utils import setup_logging
import time
import json

logger = setup_logging(__name__)


class ReflectionWorkflow:
    """
    Implements the Reflection pattern for self-improvement through iterative refinement
    """
    
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
        
        self.reflection_cycle = [
            "generation",      
            "self_critique",   
            "refinement",      
            "evaluation",      
            "iteration"        
        ]
        
        self.iteration_history = []
        self.quality_metrics = []
        
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize agents for the reflection workflow"""
        
        self.generator_agent = autogen.AssistantAgent(
            name="generator",
            llm_config=self.llm_config,
            system_message="""You are a content generator. Create high-quality responses 
            to given tasks. Focus on accuracy, clarity, and completeness."""
        )
        
        self.critic_agent = autogen.AssistantAgent(
            name="critic",
            llm_config=self.llm_config,
            system_message="""You are a critical reviewer. Analyze generated content and:
            1. Identify strengths and weaknesses
            2. Suggest specific improvements
            3. Rate quality on a scale of 0-1
            4. Focus on: accuracy, clarity, completeness, coherence, relevance
            
            Provide structured feedback in this format:
            QUALITY_SCORE: [0.0-1.0]
            STRENGTHS: [list strengths]
            WEAKNESSES: [list weaknesses]
            IMPROVEMENTS: [specific suggestions]"""
        )
        
        self.refiner_agent = autogen.AssistantAgent(
            name="refiner",
            llm_config=self.llm_config,
            system_message="""You are a content refiner. Take original content and critique,
            then produce an improved version that addresses all identified issues.
            Maintain the core message while enhancing quality."""
        )
        
        self.evaluator_agent = autogen.AssistantAgent(
            name="evaluator",
            llm_config=self.llm_config,
            system_message="""You are a quality evaluator. Compare refined content against
            quality thresholds and determine if further iteration is needed.
            Consider diminishing returns and convergence."""
        )
    
    def self_improving_task(
        self, 
        task: str, 
        quality_threshold: float = 0.8,
        max_iterations: int = 4
    ) -> Dict[str, Any]:
        """
        Execute self-improving task with iterative refinement
        
        Args:
            task: The task to complete
            quality_threshold: Quality score needed to stop iteration (0-1)
            max_iterations: Maximum refinement cycles
            
        Returns:
            Dictionary with final result and iteration history
        """
        logger.info(f"Starting reflection workflow for task: {task[:100]}...")
        start_time = time.time()
        
        current_output = None
        current_quality = 0.0
        iteration_count = 0
        
        while iteration_count < max_iterations and current_quality < quality_threshold:
            iteration_count += 1
            logger.info(f"\n=== Iteration {iteration_count} ===")
            
            iteration_data = {
                "iteration": iteration_count,
                "phase_results": {}
            }
            
            if iteration_count == 1:
                current_output = self._generation_phase(task)
            else:
                current_output = self._refinement_phase(
                    current_output, 
                    iteration_data["phase_results"].get("critique", "")
                )
            
            iteration_data["phase_results"]["generation"] = current_output
            
            critique = self._critique_phase(current_output)
            iteration_data["phase_results"]["critique"] = critique
            
            current_quality = self._extract_quality_score(critique)
            iteration_data["quality_score"] = current_quality
            
            logger.info(f"Quality score: {current_quality:.2f}")
            
            should_continue = self._evaluation_phase(
                current_quality, 
                quality_threshold, 
                iteration_count,
                max_iterations
            )
            iteration_data["continue"] = should_continue
            
            self.iteration_history.append(iteration_data)
            self.quality_metrics.append(current_quality)
            
            if not should_continue:
                logger.info(f"Quality threshold met or convergence detected")
                break
        
        elapsed_time = time.time() - start_time
        
        result = {
            "final_output": current_output,
            "final_quality": current_quality,
            "iterations": iteration_count,
            "iteration_history": self.iteration_history,
            "quality_progression": self.quality_metrics,
            "elapsed_time": elapsed_time,
            "converged": current_quality >= quality_threshold
        }
        
        logger.info(f"Reflection workflow complete in {elapsed_time:.2f}s")
        logger.info(f"Final quality: {current_quality:.2f}, Iterations: {iteration_count}")
        
        return result
    
    def _generation_phase(self, task: str) -> str:
        """Initial response generation"""
        logger.info("Generation phase...")
        
        message = f"Generate a response for this task: {task}"
        
        response = self.generator_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        return response
    
    def _critique_phase(self, content: str) -> str:
        """Analyze quality and identify improvements"""
        logger.info("Self-critique phase...")
        
        message = f"Critically analyze this content and provide structured feedback:\n\n{content}"
        
        critique = self.critic_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        return critique
    
    def _refinement_phase(self, original_content: str, critique: str) -> str:
        """Apply improvements based on critique"""
        logger.info("Refinement phase...")
        
        message = f"""Based on this critique:
{critique}

Improve this content:
{original_content}

Generate an enhanced version that addresses all identified issues."""
        
        refined = self.refiner_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        return refined
    
    def _evaluation_phase(
        self, 
        current_quality: float, 
        threshold: float,
        iteration: int,
        max_iterations: int
    ) -> bool:
        """Determine if further iteration is needed"""
        logger.info("Evaluation phase...")
        
        if current_quality >= threshold:
            return False
        
        if iteration >= max_iterations:
            logger.info("Maximum iterations reached")
            return False
        
        if len(self.quality_metrics) >= 2:
            improvement = current_quality - self.quality_metrics[-2]
            if improvement < 0.05:
                logger.info("Quality plateau detected (improvement < 0.05)")
                return False
        
        return True
    
    def _extract_quality_score(self, critique: str) -> float:
        """Extract quality score from critique"""
        try:
            if "QUALITY_SCORE:" in critique:
                score_line = [line for line in critique.split('\n') 
                            if "QUALITY_SCORE:" in line][0]
                score = float(score_line.split(":")[-1].strip())
                return min(max(score, 0.0), 1.0)
        except:
            pass
        
        return 0.5


class ToolInteractiveReflection(ReflectionWorkflow):
    """
    Advanced reflection with external tool validation
    """
    
    def __init__(self, llm_config: Optional[Dict] = None):
        super().__init__(llm_config)
        self.external_validators = {}
    
    def register_validator(self, name: str, validator_func: callable):
        """Register external validation tool"""
        self.external_validators[name] = validator_func
        logger.info(f"Registered validator: {name}")
    
    def _critique_phase(self, content: str) -> str:
        """Enhanced critique with external validation"""
        base_critique = super()._critique_phase(content)
        
        validation_results = []
        for name, validator in self.external_validators.items():
            try:
                result = validator(content)
                validation_results.append(f"{name}: {result}")
            except Exception as e:
                logger.error(f"Validator {name} failed: {e}")
        
        if validation_results:
            enhanced_critique = f"{base_critique}\n\nEXTERNAL VALIDATION:\n"
            enhanced_critique += "\n".join(validation_results)
            return enhanced_critique
        
        return base_critique


class MetacognitiveReflection(ReflectionWorkflow):
    """
    Reflection with strategy adjustment based on performance patterns
    """
    
    def __init__(self, llm_config: Optional[Dict] = None):
        super().__init__(llm_config)
        self.strategy_adjustments = []
        self.performance_patterns = {}
    
    def _analyze_performance_patterns(self):
        """Analyze iteration history for patterns"""
        if len(self.iteration_history) < 2:
            return
        
        quality_deltas = []
        for i in range(1, len(self.quality_metrics)):
            quality_deltas.append(self.quality_metrics[i] - self.quality_metrics[i-1])
        
        avg_improvement = sum(quality_deltas) / len(quality_deltas) if quality_deltas else 0
        
        self.performance_patterns = {
            "average_improvement": avg_improvement,
            "diminishing_returns": avg_improvement < 0.1,
            "stuck_pattern": max(quality_deltas) < 0.05 if quality_deltas else False,
            "rapid_convergence": self.quality_metrics[-1] > 0.8 and len(self.iteration_history) <= 2
        }
    
    def _adjust_strategy(self):
        """Adjust refinement strategy based on patterns"""
        self._analyze_performance_patterns()
        
        adjustments = []
        
        if self.performance_patterns.get("stuck_pattern"):
            adjustments.append("Switch to more creative refinement approach")
            self.llm_config["temperature"] = min(self.llm_config["temperature"] + 0.2, 1.0)
        
        if self.performance_patterns.get("rapid_convergence"):
            adjustments.append("Maintain current strategy")
        
        if self.performance_patterns.get("diminishing_returns"):
            adjustments.append("Consider terminating iterations early")
        
        self.strategy_adjustments.extend(adjustments)
        
        if adjustments:
            logger.info(f"Strategy adjustments: {', '.join(adjustments)}")


def example_code_validator(content: str) -> str:
    """Example validator for code content"""
    if "```python" in content or "def " in content or "class " in content:
        return "Code structure detected - consider syntax validation"
    return "No code detected"


def example_length_validator(content: str) -> str:
    """Example validator for content length"""
    word_count = len(content.split())
    if word_count < 50:
        return f"Content too short ({word_count} words)"
    elif word_count > 1000:
        return f"Content too long ({word_count} words)"
    return f"Content length appropriate ({word_count} words)"


def run_reflection_examples():
    """Demonstrate the Reflection pattern"""
    logger.info("Starting Reflection Pattern Examples")
    
    logger.info("\n=== Basic Reflection Example ===")
    
    basic_workflow = ReflectionWorkflow()
    
    task = "Write a brief explanation of how machine learning models learn from data"
    
    result = basic_workflow.self_improving_task(
        task=task,
        quality_threshold=0.75,
        max_iterations=3
    )
    
    logger.info(f"\nFinal output quality: {result['final_quality']:.2f}")
    logger.info(f"Iterations completed: {result['iterations']}")
    logger.info(f"Quality progression: {[f'{q:.2f}' for q in result['quality_progression']]}")
    
    logger.info("\n=== Tool-Interactive Reflection Example ===")
    
    interactive_workflow = ToolInteractiveReflection()
    
    interactive_workflow.register_validator("code_check", example_code_validator)
    interactive_workflow.register_validator("length_check", example_length_validator)
    
    code_task = "Write a Python function that sorts a list of numbers"
    
    interactive_result = interactive_workflow.self_improving_task(
        task=code_task,
        quality_threshold=0.8,
        max_iterations=3
    )
    
    logger.info(f"\nInteractive result quality: {interactive_result['final_quality']:.2f}")
    
    logger.info("\n=== Metacognitive Reflection Example ===")
    
    meta_workflow = MetacognitiveReflection()
    
    complex_task = "Explain quantum computing to a high school student"
    
    meta_result = meta_workflow.self_improving_task(
        task=complex_task,
        quality_threshold=0.85,
        max_iterations=4
    )
    
    logger.info(f"\nMetacognitive result quality: {meta_result['final_quality']:.2f}")
    logger.info(f"Performance patterns: {meta_workflow.performance_patterns}")
    logger.info(f"Strategy adjustments: {meta_workflow.strategy_adjustments}")
    
    logger.info("\nReflection Pattern Examples Complete")


if __name__ == "__main__":
    run_reflection_examples()