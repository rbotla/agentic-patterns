"""
Multi-Agent Debate Pattern Implementation
Structured argumentation improves decision quality through adversarial reasoning
Using the Agent4Debate framework with specialized roles and ASPIC+ argumentation
"""

import autogen
from typing import Dict, Any, Optional, List, Tuple, Set
import logging
from utils.logging_utils import setup_logging
import time
import json
from dataclasses import dataclass, field
from enum import Enum

logger = setup_logging(__name__)


class ArgumentType(Enum):
    """Types of arguments in ASPIC+ framework"""
    STRICT = "strict"  # Deductive inference
    DEFEASIBLE = "defeasible"  # Presumptive inference
    PREMISE = "premise"  # Base assertion
    REBUTTAL = "rebuttal"  # Attack on conclusion
    UNDERCUT = "undercut"  # Attack on inference


class DebateRole(Enum):
    """Roles in the debate system"""
    SEARCHER = "searcher"
    ANALYZER = "analyzer"
    WRITER = "writer"
    REVIEWER = "reviewer"
    MODERATOR = "moderator"


@dataclass
class Argument:
    """Represents an argument in the debate"""
    id: str
    author: str
    claim: str
    evidence: List[str]
    reasoning: str
    argument_type: ArgumentType
    strength: float  # 0.0 to 1.0
    attacks: List[str] = field(default_factory=list)
    supports: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "author": self.author,
            "claim": self.claim,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "type": self.argument_type.value,
            "strength": self.strength,
            "attacks": self.attacks,
            "supports": self.supports
        }


@dataclass
class DebateState:
    """Current state of the debate"""
    proposition: str
    arguments: Dict[str, Argument]
    current_round: int
    consensus_level: float
    quality_scores: List[float]
    convergence_history: List[float]
    
    def add_argument(self, argument: Argument):
        """Add an argument to the debate"""
        self.arguments[argument.id] = argument
    
    def get_winning_position(self) -> Optional[str]:
        """Determine winning position based on arguments"""
        pro_strength = sum(arg.strength for arg in self.arguments.values() 
                          if "support" in arg.claim.lower() or "agree" in arg.claim.lower())
        con_strength = sum(arg.strength for arg in self.arguments.values() 
                          if "oppose" in arg.claim.lower() or "disagree" in arg.claim.lower())
        
        if pro_strength > con_strength:
            return "PRO"
        elif con_strength > pro_strength:
            return "CON"
        return "UNDECIDED"


class MultiAgentDebateWorkflow:
    """
    Implements structured multi-agent debate with ASPIC+ argumentation
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
        
        self.debate_agents = {}
        self.argumentation_framework = "ASPIC+"
        self.debate_history = []
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize debate agents with specialized roles"""
        
        self.debate_agents[DebateRole.SEARCHER] = autogen.AssistantAgent(
            name="searcher",
            llm_config=self.llm_config,
            system_message="""You are a research specialist in debates.

Your responsibilities:
1. Gather relevant information and evidence
2. Find supporting facts and statistics
3. Identify credible sources
4. Discover counter-examples and edge cases
5. Provide comprehensive background information

Format evidence as:
EVIDENCE: [fact or statistic]
SOURCE: [credibility level 1-10]
RELEVANCE: [how it relates to the proposition]"""
        )
        
        self.debate_agents[DebateRole.ANALYZER] = autogen.AssistantAgent(
            name="analyzer",
            llm_config=self.llm_config,
            system_message="""You are a logical analysis specialist.

Your responsibilities:
1. Evaluate evidence strength and reliability
2. Identify logical fallacies
3. Assess argument consistency
4. Find weaknesses in reasoning
5. Rate argument quality (0.0-1.0)

Analyze using:
LOGIC_CHECK: [valid/invalid]
FALLACIES: [list any found]
STRENGTH: [0.0-1.0]
WEAKNESSES: [identified issues]"""
        )
        
        self.debate_agents[DebateRole.WRITER] = autogen.AssistantAgent(
            name="writer",
            llm_config=self.llm_config,
            system_message="""You are an argument construction specialist.

Your responsibilities:
1. Formulate structured arguments with clear claims
2. Support claims with evidence and citations
3. Build logical reasoning chains
4. Create rebuttals and counter-arguments
5. Ensure clarity and persuasiveness

Structure arguments as:
CLAIM: [main assertion]
EVIDENCE: [supporting facts]
REASONING: [logical chain]
TYPE: [strict/defeasible/rebuttal/undercut]
STRENGTH: [self-assessed 0.0-1.0]"""
        )
        
        self.debate_agents[DebateRole.REVIEWER] = autogen.AssistantAgent(
            name="reviewer",
            llm_config=self.llm_config,
            system_message="""You are a quality assessment specialist.

Your responsibilities:
1. Assess overall argument quality
2. Identify gaps in reasoning
3. Evaluate evidence sufficiency
4. Check for bias and balance
5. Recommend improvements

Provide review as:
QUALITY: [0.0-1.0]
GAPS: [missing elements]
IMPROVEMENTS: [suggestions]
BIAS_CHECK: [detected biases]"""
        )
        
        self.debate_agents[DebateRole.MODERATOR] = autogen.AssistantAgent(
            name="moderator",
            llm_config=self.llm_config,
            system_message="""You are the debate moderator.

Your responsibilities:
1. Manage debate flow and rounds
2. Ensure fair participation
3. Track consensus and convergence
4. Identify when conclusion is reached
5. Summarize key points and outcomes

Monitor for:
CONSENSUS_LEVEL: [0.0-1.0]
CONVERGENCE: [improving/stagnant/diverging]
NEXT_ACTION: [continue/conclude/redirect]
SUMMARY: [current state]"""
        )
    
    def conduct_structured_debate(
        self,
        proposition: str,
        rounds: int = 5,
        consensus_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Conduct a structured multi-agent debate"""
        logger.info(f"Starting debate on: {proposition[:100]}...")
        start_time = time.time()
        
        debate_state = DebateState(
            proposition=proposition,
            arguments={},
            current_round=0,
            consensus_level=0.0,
            quality_scores=[],
            convergence_history=[]
        )
        
        for round_num in range(1, rounds + 1):
            logger.info(f"\n=== Debate Round {round_num} ===")
            debate_state.current_round = round_num
            
            round_arguments = self._conduct_round(debate_state)
            
            for arg in round_arguments:
                debate_state.add_argument(arg)
            
            consensus = self._evaluate_consensus(debate_state)
            debate_state.consensus_level = consensus
            debate_state.convergence_history.append(consensus)
            
            logger.info(f"Round {round_num} consensus: {consensus:.2f}")
            
            if self._check_convergence(debate_state, consensus_threshold):
                logger.info(f"Convergence reached at round {round_num}")
                break
        
        final_position = debate_state.get_winning_position()
        elapsed_time = time.time() - start_time
        
        result = {
            "proposition": proposition,
            "rounds_completed": debate_state.current_round,
            "final_position": final_position,
            "consensus_level": debate_state.consensus_level,
            "arguments": [arg.to_dict() for arg in debate_state.arguments.values()],
            "convergence_history": debate_state.convergence_history,
            "execution_time": elapsed_time,
            "quality_average": sum(debate_state.quality_scores) / len(debate_state.quality_scores) if debate_state.quality_scores else 0
        }
        
        self.debate_history.append(result)
        return result
    
    def _conduct_round(self, debate_state: DebateState) -> List[Argument]:
        """Conduct a single debate round"""
        round_arguments = []
        
        evidence = self._gather_evidence(debate_state)
        
        analysis = self._analyze_arguments(debate_state, evidence)
        
        new_arguments = self._construct_arguments(debate_state, evidence, analysis)
        round_arguments.extend(new_arguments)
        
        reviews = self._review_arguments(new_arguments)
        
        for arg, review in zip(new_arguments, reviews):
            quality_score = self._extract_quality_score(review)
            arg.strength = quality_score
            debate_state.quality_scores.append(quality_score)
        
        moderation = self._moderate_round(debate_state, round_arguments)
        
        return round_arguments
    
    def _gather_evidence(self, debate_state: DebateState) -> List[Dict]:
        """Searcher gathers evidence"""
        message = f"""Gather evidence for debate on: {debate_state.proposition}
        
Current round: {debate_state.current_round}
Existing arguments: {len(debate_state.arguments)}

Find new evidence to support or refute the proposition."""
        
        response = self.debate_agents[DebateRole.SEARCHER].generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        evidence = self._parse_evidence(response)
        return evidence
    
    def _analyze_arguments(self, debate_state: DebateState, evidence: List[Dict]) -> Dict:
        """Analyzer evaluates arguments"""
        existing_args = [arg.to_dict() for arg in debate_state.arguments.values()]
        
        message = f"""Analyze arguments for: {debate_state.proposition}

Existing arguments: {json.dumps(existing_args, indent=2)}
New evidence: {json.dumps(evidence, indent=2)}

Evaluate logical consistency and identify weaknesses."""
        
        response = self.debate_agents[DebateRole.ANALYZER].generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        return {"analysis": response}
    
    def _construct_arguments(
        self,
        debate_state: DebateState,
        evidence: List[Dict],
        analysis: Dict
    ) -> List[Argument]:
        """Writer constructs new arguments"""
        message = f"""Construct arguments for: {debate_state.proposition}

Round: {debate_state.current_round}
Evidence: {json.dumps(evidence, indent=2)}
Analysis: {analysis['analysis']}

Create structured arguments (both supporting and opposing)."""
        
        response = self.debate_agents[DebateRole.WRITER].generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        arguments = self._parse_arguments(response, debate_state.current_round)
        return arguments
    
    def _review_arguments(self, arguments: List[Argument]) -> List[str]:
        """Reviewer assesses argument quality"""
        reviews = []
        
        for arg in arguments:
            message = f"""Review this argument:

Claim: {arg.claim}
Evidence: {arg.evidence}
Reasoning: {arg.reasoning}
Type: {arg.argument_type.value}

Assess quality and identify improvements."""
            
            response = self.debate_agents[DebateRole.REVIEWER].generate_reply(
                messages=[{"role": "user", "content": message}]
            )
            reviews.append(response)
        
        return reviews
    
    def _moderate_round(
        self,
        debate_state: DebateState,
        new_arguments: List[Argument]
    ) -> Dict:
        """Moderator manages debate flow"""
        message = f"""Moderate debate round {debate_state.current_round}:

Proposition: {debate_state.proposition}
New arguments this round: {len(new_arguments)}
Total arguments: {len(debate_state.arguments) + len(new_arguments)}
Current consensus: {debate_state.consensus_level:.2f}

Assess progress and determine next steps."""
        
        response = self.debate_agents[DebateRole.MODERATOR].generate_reply(
            messages=[{"role": "user", "content": message}]
        )
        
        return {"moderation": response}
    
    def _parse_evidence(self, response: str) -> List[Dict]:
        """Parse evidence from searcher response"""
        evidence = []
        lines = response.split('\n')
        
        current_evidence = {}
        for line in lines:
            if "EVIDENCE:" in line:
                if current_evidence:
                    evidence.append(current_evidence)
                current_evidence = {"fact": line.split(":")[-1].strip()}
            elif "SOURCE:" in line and current_evidence:
                current_evidence["source"] = line.split(":")[-1].strip()
            elif "RELEVANCE:" in line and current_evidence:
                current_evidence["relevance"] = line.split(":")[-1].strip()
        
        if current_evidence:
            evidence.append(current_evidence)
        
        return evidence if evidence else [{"fact": "General observation", "source": "5", "relevance": "Related"}]
    
    def _parse_arguments(self, response: str, round_num: int) -> List[Argument]:
        """Parse arguments from writer response"""
        arguments = []
        
        arg_id = f"arg_{round_num}_{len(arguments) + 1}"
        
        claim = "Position on the proposition"
        evidence = []
        reasoning = "Logical reasoning"
        arg_type = ArgumentType.DEFEASIBLE
        strength = 0.5
        
        lines = response.split('\n')
        for line in lines:
            if "CLAIM:" in line:
                claim = line.split(":")[-1].strip()
            elif "EVIDENCE:" in line:
                evidence.append(line.split(":")[-1].strip())
            elif "REASONING:" in line:
                reasoning = line.split(":")[-1].strip()
            elif "TYPE:" in line:
                type_str = line.split(":")[-1].strip().lower()
                if "strict" in type_str:
                    arg_type = ArgumentType.STRICT
                elif "rebuttal" in type_str:
                    arg_type = ArgumentType.REBUTTAL
                elif "undercut" in type_str:
                    arg_type = ArgumentType.UNDERCUT
            elif "STRENGTH:" in line:
                try:
                    strength = float(line.split(":")[-1].strip())
                except:
                    pass
        
        argument = Argument(
            id=arg_id,
            author="writer",
            claim=claim,
            evidence=evidence if evidence else ["Supporting evidence"],
            reasoning=reasoning,
            argument_type=arg_type,
            strength=strength
        )
        
        arguments.append(argument)
        
        if round_num > 1:
            counter_arg = Argument(
                id=f"arg_{round_num}_2",
                author="writer",
                claim=f"Counter to: {claim}",
                evidence=["Counter evidence"],
                reasoning="Alternative perspective",
                argument_type=ArgumentType.REBUTTAL,
                strength=0.6,
                attacks=[arg_id]
            )
            arguments.append(counter_arg)
        
        return arguments
    
    def _extract_quality_score(self, review: str) -> float:
        """Extract quality score from review"""
        lines = review.split('\n')
        for line in lines:
            if "QUALITY:" in line:
                try:
                    return float(line.split(":")[-1].strip())
                except:
                    pass
        return 0.5
    
    def _evaluate_consensus(self, debate_state: DebateState) -> float:
        """Evaluate current consensus level"""
        if not debate_state.arguments:
            return 0.0
        
        position_counts = {"pro": 0, "con": 0, "neutral": 0}
        
        for arg in debate_state.arguments.values():
            if "support" in arg.claim.lower() or "agree" in arg.claim.lower():
                position_counts["pro"] += arg.strength
            elif "oppose" in arg.claim.lower() or "disagree" in arg.claim.lower():
                position_counts["con"] += arg.strength
            else:
                position_counts["neutral"] += arg.strength
        
        total = sum(position_counts.values())
        if total == 0:
            return 0.0
        
        max_position = max(position_counts.values())
        consensus = max_position / total
        
        return consensus
    
    def _check_convergence(
        self,
        debate_state: DebateState,
        threshold: float
    ) -> bool:
        """Check if debate has converged"""
        if debate_state.consensus_level >= threshold:
            return True
        
        if len(debate_state.convergence_history) >= 3:
            recent = debate_state.convergence_history[-3:]
            if max(recent) - min(recent) < 0.05:
                logger.info("Quality plateau detected")
                return True
        
        return False


class AdvancedDebate(MultiAgentDebateWorkflow):
    """Advanced debate with preference-based conflict resolution"""
    
    def __init__(self, llm_config: Optional[Dict] = None):
        super().__init__(llm_config)
        self.preference_ordering = {}
    
    def set_preferences(self, preferences: Dict[str, float]):
        """Set preference weights for different argument types"""
        self.preference_ordering = preferences
    
    def resolve_conflicts(self, arguments: List[Argument]) -> Argument:
        """Resolve conflicts using preference-based ordering"""
        if not arguments:
            return None
        
        scored_args = []
        for arg in arguments:
            base_score = arg.strength
            
            pref_weight = self.preference_ordering.get(
                arg.argument_type.value, 1.0
            )
            final_score = base_score * pref_weight
            
            scored_args.append((arg, final_score))
        
        scored_args.sort(key=lambda x: x[1], reverse=True)
        
        winner = scored_args[0][0]
        logger.info(f"Conflict resolved in favor of: {winner.claim[:50]}...")
        
        return winner


def run_debate_examples():
    """Demonstrate Multi-Agent Debate pattern"""
    logger.info("Starting Multi-Agent Debate Pattern Examples")
    
    logger.info("\n=== Basic Debate Example ===")
    
    debate_workflow = MultiAgentDebateWorkflow()
    
    proposition1 = "Artificial Intelligence will have a net positive impact on employment"
    
    result1 = debate_workflow.conduct_structured_debate(
        proposition1,
        rounds=3,
        consensus_threshold=0.7
    )
    
    logger.info(f"\nProposition: {proposition1}")
    logger.info(f"Rounds completed: {result1['rounds_completed']}")
    logger.info(f"Final position: {result1['final_position']}")
    logger.info(f"Consensus level: {result1['consensus_level']:.2f}")
    logger.info(f"Total arguments: {len(result1['arguments'])}")
    
    logger.info("\n=== Extended Debate Example ===")
    
    proposition2 = "Remote work should be the default option for knowledge workers"
    
    result2 = debate_workflow.conduct_structured_debate(
        proposition2,
        rounds=5,
        consensus_threshold=0.8
    )
    
    logger.info(f"\nExtended debate on: {proposition2}")
    logger.info(f"Convergence history: {[f'{c:.2f}' for c in result2['convergence_history']]}")
    logger.info(f"Quality average: {result2['quality_average']:.2f}")
    
    logger.info("\n=== Advanced Debate with Preferences ===")
    
    advanced_debate = AdvancedDebate()
    
    advanced_debate.set_preferences({
        "strict": 1.5,
        "defeasible": 1.0,
        "rebuttal": 0.8,
        "undercut": 0.7
    })
    
    proposition3 = "Universal Basic Income is necessary for future economic stability"
    
    result3 = advanced_debate.conduct_structured_debate(
        proposition3,
        rounds=4,
        consensus_threshold=0.75
    )
    
    logger.info(f"\nAdvanced debate with preferences")
    logger.info(f"Proposition: {proposition3}")
    logger.info(f"Final position: {result3['final_position']}")
    
    logger.info("\n=== Argument Analysis ===")
    
    if result1['arguments']:
        logger.info("\nSample arguments from first debate:")
        for arg in result1['arguments'][:3]:
            logger.info(f"\n- Claim: {arg['claim'][:100]}...")
            logger.info(f"  Type: {arg['type']}")
            logger.info(f"  Strength: {arg['strength']:.2f}")
            if arg['attacks']:
                logger.info(f"  Attacks: {arg['attacks']}")
            if arg['supports']:
                logger.info(f"  Supports: {arg['supports']}")
    
    logger.info("\nMulti-Agent Debate Pattern Examples Complete")


if __name__ == "__main__":
    run_debate_examples()