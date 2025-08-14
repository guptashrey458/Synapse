"""
Core interfaces for the autonomous delivery coordinator agent.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any


class UrgencyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScenarioType(Enum):
    TRAFFIC = "traffic"
    MERCHANT = "merchant"
    ADDRESS = "address"
    MULTI_FACTOR = "multi_factor"
    OTHER = "other"


class EntityType(Enum):
    ADDRESS = "address"
    MERCHANT = "merchant"
    DELIVERY_ID = "delivery_id"
    PERSON = "person"
    PHONE_NUMBER = "phone"
    TIME = "time"


@dataclass
class Entity:
    text: str
    entity_type: EntityType
    confidence: float
    normalized_value: Optional[str] = None


@dataclass
class DisruptionScenario:
    description: str
    entities: List[Entity]
    scenario_type: ScenarioType
    urgency_level: UrgencyLevel


@dataclass
class ReasoningStep:
    step_number: int
    thought: str
    action: Optional[str]
    observation: Optional[str]
    timestamp: datetime


@dataclass
class ReasoningTrace:
    steps: List[ReasoningStep]
    scenario: DisruptionScenario
    start_time: datetime
    end_time: Optional[datetime] = None


@dataclass
class PlanStep:
    sequence: int
    action: str
    responsible_party: str
    estimated_time: timedelta
    dependencies: List[int]
    success_criteria: str


@dataclass
class ResolutionPlan:
    steps: List[PlanStep]
    estimated_duration: timedelta
    success_probability: float
    alternatives: List[str]
    stakeholders: List[str]


@dataclass
class ResolutionResult:
    scenario: DisruptionScenario
    reasoning_trace: ReasoningTrace
    resolution_plan: ResolutionPlan
    success: bool
    error_message: Optional[str] = None


class Agent(ABC):
    """Base interface for autonomous agents."""
    
    @abstractmethod
    def process_scenario(self, scenario: str) -> ResolutionResult:
        """
        Main entry point for processing disruption scenarios.
        
        Args:
            scenario: Natural language description of the disruption
            
        Returns:
            ResolutionResult containing the reasoning trace and resolution plan
        """
        pass