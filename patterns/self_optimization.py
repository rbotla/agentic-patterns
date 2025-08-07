"""
Self-Optimization Pattern Implementation
Agents that continuously improve their performance through learning,
adaptation, and automatic hyperparameter tuning
"""

import autogen
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging
from utils.logging_utils import setup_logging
import time
import json
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
import pickle
from abc import ABC, abstractmethod

logger = setup_logging(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategies"""
    BAYESIAN = "bayesian"          # Bayesian optimization
    GENETIC = "genetic"            # Genetic algorithm
    GRADIENT_BASED = "gradient"    # Gradient-based optimization
    RANDOM_SEARCH = "random"       # Random search
    REINFORCEMENT = "reinforcement" # Reinforcement learning


class PerformanceMetric(Enum):
    """Performance metrics to optimize"""
    RESPONSE_TIME = "response_time"
    ACCURACY = "accuracy"
    COST = "cost"
    THROUGHPUT = "throughput"
    USER_SATISFACTION = "user_satisfaction"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"


@dataclass
class OptimizationConfig:
    """Configuration for optimization"""
    target_metric: PerformanceMetric
    optimization_goal: str  # "minimize" or "maximize"
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    memory_size: int = 1000
    update_frequency: int = 10
    convergence_threshold: float = 0.001
    max_iterations: int = 1000


@dataclass
class PerformanceObservation:
    """Performance observation data point"""
    timestamp: float
    configuration: Dict[str, Any]
    metrics: Dict[str, float]
    context: Dict[str, Any] = field(default_factory=dict)
    
    def get_score(self, target_metric: PerformanceMetric, goal: str) -> float:
        """Get normalized score for target metric"""
        value = self.metrics.get(target_metric.value, 0.0)
        
        # Simple normalization (in practice, you'd use domain knowledge)
        if goal == "minimize":
            return 1.0 / (1.0 + value)
        else:  # maximize
            return value


class OptimizationHistory:
    """Maintains history of optimization attempts"""
    
    def __init__(self, max_size: int = 10000):
        self.observations: deque = deque(maxlen=max_size)
        self.best_configuration = None
        self.best_score = float('-inf')
        self.convergence_history = []
    
    def add_observation(self, observation: PerformanceObservation, target_metric: PerformanceMetric, goal: str):
        """Add a new observation"""
        self.observations.append(observation)
        
        score = observation.get_score(target_metric, goal)
        if score > self.best_score:
            self.best_score = score
            self.best_configuration = observation.configuration.copy()
        
        # Track convergence
        recent_scores = [obs.get_score(target_metric, goal) 
                        for obs in list(self.observations)[-10:]]
        convergence = np.std(recent_scores) if len(recent_scores) > 1 else float('inf')
        self.convergence_history.append(convergence)
    
    def get_recent_observations(self, n: int = 100) -> List[PerformanceObservation]:
        """Get recent observations"""
        return list(self.observations)[-n:]
    
    def has_converged(self, threshold: float = 0.001) -> bool:
        """Check if optimization has converged"""
        if len(self.convergence_history) < 10:
            return False
        
        return np.mean(self.convergence_history[-10:]) < threshold


class BayesianOptimizer:
    """Bayesian optimization for hyperparameter tuning"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.history = OptimizationHistory()
        
        # Simple Gaussian Process approximation
        self.parameter_bounds = {}
        self.acquisition_function = "expected_improvement"
    
    def suggest_configuration(self, parameter_space: Dict[str, Tuple]) -> Dict[str, Any]:
        """Suggest next configuration to try"""
        if len(self.history.observations) < 5:
            # Random exploration for first few iterations
            return self._random_configuration(parameter_space)
        
        # Use acquisition function to balance exploration/exploitation
        return self._acquisition_based_suggestion(parameter_space)
    
    def update(self, configuration: Dict[str, Any], metrics: Dict[str, float], context: Dict = None):
        """Update optimizer with new observation"""
        observation = PerformanceObservation(
            timestamp=time.time(),
            configuration=configuration,
            metrics=metrics,
            context=context or {}
        )
        
        self.history.add_observation(
            observation,
            self.config.target_metric,
            self.config.optimization_goal
        )
        
        logger.debug(f"Updated optimizer with score: {observation.get_score(self.config.target_metric, self.config.optimization_goal):.4f}")
    
    def _random_configuration(self, parameter_space: Dict[str, Tuple]) -> Dict[str, Any]:
        """Generate random configuration"""
        config = {}
        for param, bounds in parameter_space.items():
            if isinstance(bounds[0], (int, float)):
                config[param] = np.random.uniform(bounds[0], bounds[1])
            else:
                config[param] = np.random.choice(bounds)
        return config
    
    def _acquisition_based_suggestion(self, parameter_space: Dict[str, Tuple]) -> Dict[str, Any]:
        """Suggest configuration using acquisition function"""
        # Simplified acquisition function - in practice use GP
        
        # Get recent successful configurations
        recent_obs = self.history.get_recent_observations(50)
        successful_configs = [
            obs.configuration for obs in recent_obs 
            if obs.get_score(self.config.target_metric, self.config.optimization_goal) > 0.5
        ]
        
        if not successful_configs:
            return self._random_configuration(parameter_space)
        
        # Perturb best configuration with some noise
        base_config = self.history.best_configuration or successful_configs[-1]
        new_config = {}
        
        for param, bounds in parameter_space.items():
            base_value = base_config.get(param, (bounds[0] + bounds[1]) / 2)
            
            if isinstance(bounds[0], (int, float)):
                # Add gaussian noise
                noise = np.random.normal(0, (bounds[1] - bounds[0]) * 0.1)
                new_value = np.clip(base_value + noise, bounds[0], bounds[1])
                new_config[param] = new_value
            else:
                # For categorical, occasionally explore
                if np.random.random() < self.config.exploration_rate:
                    new_config[param] = np.random.choice(bounds)
                else:
                    new_config[param] = base_value
        
        return new_config


class ReinforcementLearningOptimizer:
    """RL-based optimizer using Q-learning"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_discretizer = {}
        self.action_space = []
        self.epsilon = config.exploration_rate
        self.learning_rate = config.learning_rate
        self.history = OptimizationHistory()
    
    def set_action_space(self, actions: List[Dict[str, Any]]):
        """Set the action space for RL"""
        self.action_space = actions
    
    def suggest_configuration(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest configuration using epsilon-greedy policy"""
        state_key = self._discretize_state(current_state)
        
        if np.random.random() < self.epsilon or state_key not in self.q_table:
            # Explore: random action
            return np.random.choice(self.action_space) if self.action_space else {}
        
        # Exploit: best action
        best_action_idx = max(
            enumerate(self.action_space),
            key=lambda x: self.q_table[state_key].get(x[0], 0)
        )[0]
        
        return self.action_space[best_action_idx]
    
    def update(self, state: Dict[str, Any], action: Dict[str, Any], reward: float, next_state: Dict[str, Any]):
        """Update Q-table with experience"""
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Find action index
        action_idx = self._find_action_index(action)
        if action_idx is None:
            return
        
        # Q-learning update
        current_q = self.q_table[state_key].get(action_idx, 0)
        
        if next_state_key in self.q_table:
            max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0
        else:
            max_next_q = 0
        
        new_q = current_q + self.learning_rate * (reward + 0.9 * max_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q
        
        # Decay exploration
        self.epsilon *= 0.999
    
    def _discretize_state(self, state: Dict[str, Any]) -> str:
        """Convert continuous state to discrete key"""
        # Simple discretization - in practice, use better methods
        discrete_values = []
        for key, value in sorted(state.items()):
            if isinstance(value, (int, float)):
                discrete_values.append(f"{key}:{int(value * 10)}")
            else:
                discrete_values.append(f"{key}:{value}")
        return "|".join(discrete_values)
    
    def _find_action_index(self, action: Dict[str, Any]) -> Optional[int]:
        """Find index of action in action space"""
        for i, space_action in enumerate(self.action_space):
            if space_action == action:
                return i
        return None


class SelfOptimizingAgent:
    """Agent that optimizes its own performance"""
    
    def __init__(
        self,
        name: str,
        base_agent: autogen.ConversableAgent,
        optimization_config: OptimizationConfig,
        strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN
    ):
        self.name = name
        self.base_agent = base_agent
        self.config = optimization_config
        self.strategy = strategy
        
        # Initialize optimizer
        if strategy == OptimizationStrategy.BAYESIAN:
            self.optimizer = BayesianOptimizer(optimization_config)
        elif strategy == OptimizationStrategy.REINFORCEMENT:
            self.optimizer = ReinforcementLearningOptimizer(optimization_config)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        
        # Performance tracking
        self.performance_history = []
        self.current_configuration = self._get_default_configuration()
        self.iterations_since_update = 0
        
        # Parameter space definition
        self.parameter_space = self._define_parameter_space()
        
        # Wrap agent methods for monitoring
        self._wrap_agent_methods()
    
    def _define_parameter_space(self) -> Dict[str, Tuple]:
        """Define the parameter space to optimize"""
        return {
            "temperature": (0.1, 2.0),
            "max_tokens": (50, 500),
            "top_p": (0.1, 1.0),
            "frequency_penalty": (0.0, 2.0),
            "presence_penalty": (0.0, 2.0)
        }
    
    def _get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    
    def _wrap_agent_methods(self):
        """Wrap agent methods to collect performance data"""
        original_generate_reply = self.base_agent.generate_reply
        
        def optimized_generate_reply(*args, **kwargs):
            start_time = time.time()
            
            # Apply current configuration to LLM config
            if hasattr(self.base_agent, 'llm_config') and self.base_agent.llm_config:
                for key, value in self.current_configuration.items():
                    if key in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
                        self.base_agent.llm_config[key] = value
            
            try:
                result = original_generate_reply(*args, **kwargs)
                
                # Measure performance
                response_time = time.time() - start_time
                performance_metrics = self._calculate_performance_metrics(result, response_time, *args, **kwargs)
                
                # Record observation
                self._record_performance(performance_metrics)
                
                # Trigger optimization if needed
                self.iterations_since_update += 1
                if self.iterations_since_update >= self.config.update_frequency:
                    self._optimize_configuration()
                    self.iterations_since_update = 0
                
                return result
                
            except Exception as e:
                # Record failure
                error_metrics = {
                    "response_time": time.time() - start_time,
                    "error_rate": 1.0,
                    "accuracy": 0.0
                }
                self._record_performance(error_metrics)
                raise
        
        self.base_agent.generate_reply = optimized_generate_reply
    
    def _calculate_performance_metrics(self, result: str, response_time: float, *args, **kwargs) -> Dict[str, float]:
        """Calculate performance metrics from result"""
        # Simple heuristic-based metrics (in practice, use domain-specific evaluation)
        
        metrics = {
            "response_time": response_time,
            "error_rate": 0.0,  # No error if we got here
        }
        
        # Response quality heuristics
        if result:
            # Length-based quality (prefer moderate length)
            optimal_length = 100
            length_quality = 1.0 - min(abs(len(result) - optimal_length) / optimal_length, 1.0)
            metrics["quality"] = length_quality
            
            # Coherence heuristic (simple word repetition check)
            words = result.lower().split()
            unique_words = len(set(words))
            repetition_score = unique_words / max(len(words), 1)
            metrics["coherence"] = repetition_score
            
            # Combined accuracy score
            metrics["accuracy"] = (length_quality + repetition_score) / 2
        else:
            metrics["quality"] = 0.0
            metrics["coherence"] = 0.0
            metrics["accuracy"] = 0.0
        
        # Cost heuristic (based on response time and length)
        metrics["cost"] = response_time * max(len(result) / 100, 1.0) if result else 1.0
        
        # User satisfaction heuristic
        metrics["user_satisfaction"] = metrics["accuracy"] * (1.0 - min(response_time / 2.0, 0.5))
        
        return metrics
    
    def _record_performance(self, metrics: Dict[str, float]):
        """Record performance observation"""
        self.performance_history.append({
            "timestamp": time.time(),
            "configuration": self.current_configuration.copy(),
            "metrics": metrics
        })
        
        # Update optimizer
        if isinstance(self.optimizer, BayesianOptimizer):
            self.optimizer.update(self.current_configuration, metrics)
        elif isinstance(self.optimizer, ReinforcementLearningOptimizer):
            # For RL, we need state transitions - simplified here
            state = {"recent_performance": np.mean([h["metrics"]["accuracy"] for h in self.performance_history[-5:]])}
            reward = self._calculate_reward(metrics)
            next_state = state  # Simplified
            self.optimizer.update(state, self.current_configuration, reward, next_state)
    
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """Calculate reward for RL optimization"""
        target_value = metrics.get(self.config.target_metric.value, 0.0)
        
        if self.config.optimization_goal == "maximize":
            return target_value
        else:  # minimize
            return -target_value
    
    def _optimize_configuration(self):
        """Optimize the configuration based on collected data"""
        if len(self.performance_history) < 5:
            return
        
        logger.info(f"Optimizing configuration for {self.name}")
        
        # Get suggestion from optimizer
        if isinstance(self.optimizer, BayesianOptimizer):
            suggested_config = self.optimizer.suggest_configuration(self.parameter_space)
        elif isinstance(self.optimizer, ReinforcementLearningOptimizer):
            current_state = {"recent_performance": np.mean([h["metrics"]["accuracy"] for h in self.performance_history[-5:]])}
            suggested_config = self.optimizer.suggest_configuration(current_state)
        else:
            return
        
        # Apply configuration
        old_config = self.current_configuration.copy()
        self.current_configuration = suggested_config
        
        logger.info(f"Configuration updated from {old_config} to {suggested_config}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        if not self.performance_history:
            return {"status": "no_data"}
        
        recent_performance = self.performance_history[-10:]
        recent_scores = [h["metrics"].get(self.config.target_metric.value, 0) for h in recent_performance]
        
        return {
            "current_configuration": self.current_configuration,
            "performance_history_length": len(self.performance_history),
            "recent_average_performance": np.mean(recent_scores),
            "performance_trend": np.polyfit(range(len(recent_scores)), recent_scores, 1)[0] if len(recent_scores) > 1 else 0,
            "best_configuration": getattr(self.optimizer, 'history', None) and self.optimizer.history.best_configuration,
            "best_score": getattr(self.optimizer, 'history', None) and self.optimizer.history.best_score,
            "iterations_since_update": self.iterations_since_update
        }


class AdaptiveWorkflow:
    """Workflow that adapts based on performance feedback"""
    
    def __init__(self):
        self.workflow_variants = {}
        self.performance_tracker = defaultdict(list)
        self.current_variant = "default"
        self.adaptation_threshold = 10  # adaptations per threshold observations
        self.observations = 0
    
    def register_variant(self, name: str, workflow_func: Callable):
        """Register a workflow variant"""
        self.workflow_variants[name] = workflow_func
    
    def execute_adaptive_workflow(self, task: str, context: Dict = None) -> Dict[str, Any]:
        """Execute workflow with adaptation"""
        if not self.workflow_variants:
            raise ValueError("No workflow variants registered")
        
        # Select current variant
        variant_func = self.workflow_variants.get(self.current_variant)
        if not variant_func:
            variant_func = list(self.workflow_variants.values())[0]
        
        # Execute workflow
        start_time = time.time()
        result = variant_func(task, context or {})
        execution_time = time.time() - start_time
        
        # Calculate performance score
        score = self._evaluate_result(result, execution_time)
        
        # Track performance
        self.performance_tracker[self.current_variant].append(score)
        self.observations += 1
        
        # Adapt if needed
        if self.observations % self.adaptation_threshold == 0:
            self._adapt_workflow()
        
        return {
            "result": result,
            "execution_time": execution_time,
            "performance_score": score,
            "variant_used": self.current_variant
        }
    
    def _evaluate_result(self, result: Any, execution_time: float) -> float:
        """Evaluate result quality (placeholder implementation)"""
        # Simple heuristic - in practice, use domain-specific evaluation
        quality_score = 0.8  # Assume decent quality
        time_penalty = min(execution_time / 5.0, 0.5)  # Penalize slow execution
        return quality_score - time_penalty
    
    def _adapt_workflow(self):
        """Adapt workflow based on performance"""
        if len(self.performance_tracker) < 2:
            return
        
        # Find best performing variant
        variant_averages = {}
        for variant, scores in self.performance_tracker.items():
            if scores:
                variant_averages[variant] = np.mean(scores[-10:])  # Recent average
        
        if not variant_averages:
            return
        
        best_variant = max(variant_averages, key=variant_averages.get)
        
        if best_variant != self.current_variant:
            logger.info(f"Adapting workflow from {self.current_variant} to {best_variant}")
            logger.info(f"Performance improvement: {variant_averages[best_variant] - variant_averages.get(self.current_variant, 0):.3f}")
            self.current_variant = best_variant


def run_optimization_examples():
    """Demonstrate self-optimization patterns"""
    logger.info("Starting Self-Optimization Pattern Examples")
    
    logger.info("\n=== Bayesian Optimizer Example ===")
    
    # Test Bayesian optimization
    config = OptimizationConfig(
        target_metric=PerformanceMetric.ACCURACY,
        optimization_goal="maximize",
        learning_rate=0.01,
        exploration_rate=0.1,
        update_frequency=5
    )
    
    bayesian_opt = BayesianOptimizer(config)
    parameter_space = {
        "temperature": (0.1, 2.0),
        "max_tokens": (50, 300)
    }
    
    # Simulate optimization iterations
    for i in range(20):
        config_suggestion = bayesian_opt.suggest_configuration(parameter_space)
        
        # Simulate performance (higher accuracy with temperature around 0.7)
        temp = config_suggestion["temperature"]
        simulated_accuracy = 0.8 + 0.2 * np.exp(-((temp - 0.7) ** 2) / 0.1) + np.random.normal(0, 0.05)
        simulated_accuracy = np.clip(simulated_accuracy, 0, 1)
        
        metrics = {
            "accuracy": simulated_accuracy,
            "response_time": np.random.uniform(0.5, 2.0)
        }
        
        bayesian_opt.update(config_suggestion, metrics)
    
    logger.info(f"Best configuration found: {bayesian_opt.history.best_configuration}")
    logger.info(f"Best score: {bayesian_opt.history.best_score:.4f}")
    
    logger.info("\n=== Self-Optimizing Agent Example ===")
    
    # Create base agent
    base_agent = autogen.ConversableAgent(
        name="base_agent",
        llm_config=False,  # Simplified for demo
        human_input_mode="NEVER"
    )
    
    # Simple response function
    def simple_response(**kwargs):
        messages = kwargs.get("messages", [])
        if messages:
            return True, f"Response to: {messages[-1].get('content', 'N/A')}"
        return True, "Default response"
    
    base_agent.register_reply(
        trigger=lambda sender: True,
        reply_func=simple_response,
        position=0
    )
    
    # Create self-optimizing wrapper
    opt_config = OptimizationConfig(
        target_metric=PerformanceMetric.ACCURACY,
        optimization_goal="maximize",
        update_frequency=3
    )
    
    self_opt_agent = SelfOptimizingAgent(
        "optimizing_agent",
        base_agent,
        opt_config,
        OptimizationStrategy.BAYESIAN
    )
    
    # Simulate agent interactions
    test_messages = [
        "Hello, how are you?",
        "Can you help me with Python?",
        "What is machine learning?",
        "Explain quantum computing",
        "How do I write good code?"
    ]
    
    for i, message in enumerate(test_messages * 2):  # Run twice for optimization
        response = base_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        logger.debug(f"Iteration {i+1}: {message[:30]}... -> {response[:30]}...")
    
    # Get optimization status
    opt_status = self_opt_agent.get_optimization_status()
    logger.info(f"Optimization status: {json.dumps(opt_status, indent=2, default=str)}")
    
    logger.info("\n=== Adaptive Workflow Example ===")
    
    # Create adaptive workflow
    adaptive_wf = AdaptiveWorkflow()
    
    # Register workflow variants
    def fast_workflow(task: str, context: Dict) -> str:
        time.sleep(0.1)  # Fast execution
        return f"Fast result for: {task}"
    
    def thorough_workflow(task: str, context: Dict) -> str:
        time.sleep(0.5)  # Slower but more thorough
        return f"Thorough analysis of: {task}"
    
    def balanced_workflow(task: str, context: Dict) -> str:
        time.sleep(0.2)  # Balanced approach
        return f"Balanced processing of: {task}"
    
    adaptive_wf.register_variant("fast", fast_workflow)
    adaptive_wf.register_variant("thorough", thorough_workflow)
    adaptive_wf.register_variant("balanced", balanced_workflow)
    
    # Execute adaptive workflow
    tasks = [
        "Analyze customer feedback",
        "Process payment data", 
        "Generate report",
        "Validate inputs",
        "Optimize performance"
    ]
    
    results = []
    for task in tasks * 5:  # Multiple iterations for adaptation
        result = adaptive_wf.execute_adaptive_workflow(task)
        results.append(result)
    
    # Analyze adaptation
    variant_usage = defaultdict(int)
    for result in results:
        variant_usage[result["variant_used"]] += 1
    
    logger.info(f"Workflow variant usage: {dict(variant_usage)}")
    logger.info(f"Final selected variant: {adaptive_wf.current_variant}")
    
    # Performance by variant
    for variant in variant_usage:
        variant_results = [r for r in results if r["variant_used"] == variant]
        avg_score = np.mean([r["performance_score"] for r in variant_results])
        avg_time = np.mean([r["execution_time"] for r in variant_results])
        logger.info(f"Variant '{variant}': avg_score={avg_score:.3f}, avg_time={avg_time:.3f}s")
    
    logger.info("\n=== Performance Evolution Analysis ===")
    
    # Analyze performance evolution
    if self_opt_agent.performance_history:
        # Extract accuracy over time
        accuracy_history = [h["metrics"]["accuracy"] for h in self_opt_agent.performance_history]
        
        # Calculate improvement
        if len(accuracy_history) > 5:
            early_performance = np.mean(accuracy_history[:5])
            recent_performance = np.mean(accuracy_history[-5:])
            improvement = recent_performance - early_performance
            
            logger.info(f"Performance evolution:")
            logger.info(f"  Early performance: {early_performance:.4f}")
            logger.info(f"  Recent performance: {recent_performance:.4f}")
            logger.info(f"  Improvement: {improvement:.4f} ({improvement/early_performance*100:.1f}%)")
        
        # Configuration evolution
        if len(self_opt_agent.performance_history) > 1:
            first_config = self_opt_agent.performance_history[0]["configuration"]
            latest_config = self_opt_agent.performance_history[-1]["configuration"]
            
            logger.info(f"Configuration evolution:")
            for param in first_config:
                old_val = first_config[param]
                new_val = latest_config[param]
                change = ((new_val - old_val) / old_val * 100) if old_val != 0 else 0
                logger.info(f"  {param}: {old_val:.3f} -> {new_val:.3f} ({change:+.1f}%)")
    
    logger.info("\nSelf-Optimization Pattern Examples Complete")


if __name__ == "__main__":
    run_optimization_examples()