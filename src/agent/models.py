"""
Enhanced data models with validation for the autonomous delivery coordinator.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from .interfaces import Entity, EntityType, DisruptionScenario, ScenarioType, UrgencyLevel


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


@dataclass
class ValidatedEntity(Entity):
    """Enhanced Entity with validation methods."""
    
    def __post_init__(self):
        """Validate entity data after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate entity data."""
        if not self.text or not self.text.strip():
            raise ValidationError("Entity text cannot be empty")
        
        if not isinstance(self.entity_type, EntityType):
            raise ValidationError(f"Invalid entity type: {self.entity_type}")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        # Type-specific validation
        if self.entity_type == EntityType.DELIVERY_ID:
            self._validate_delivery_id()
        elif self.entity_type == EntityType.PHONE_NUMBER:
            self._validate_phone_number()
        elif self.entity_type == EntityType.ADDRESS:
            self._validate_address()
    
    def _validate_delivery_id(self) -> None:
        """Validate delivery ID format."""
        if self.normalized_value:
            # Expect format like DEL123456 or similar
            if not re.match(r'^[A-Z]{2,4}\d{3,8}$', self.normalized_value):
                raise ValidationError(f"Invalid delivery ID format: {self.normalized_value}")
    
    def _validate_phone_number(self) -> None:
        """Validate phone number format."""
        if self.normalized_value:
            # Basic phone number validation (digits, spaces, dashes, parentheses)
            if not re.match(r'^[\d\s\-\(\)\+]{10,15}$', self.normalized_value):
                raise ValidationError(f"Invalid phone number format: {self.normalized_value}")
    
    def _validate_address(self) -> None:
        """Validate address format."""
        if self.normalized_value:
            # Basic address validation - should contain some numbers and letters
            if len(self.normalized_value.strip()) < 5:
                raise ValidationError(f"Address too short: {self.normalized_value}")


@dataclass
class ValidatedDisruptionScenario(DisruptionScenario):
    """Enhanced DisruptionScenario with validation methods."""
    
    def __post_init__(self):
        """Validate scenario data after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate disruption scenario data."""
        if not self.description or not self.description.strip():
            raise ValidationError("Scenario description cannot be empty")
        
        if len(self.description.strip()) < 10:
            raise ValidationError("Scenario description too short (minimum 10 characters)")
        
        if not isinstance(self.scenario_type, ScenarioType):
            raise ValidationError(f"Invalid scenario type: {self.scenario_type}")
        
        if not isinstance(self.urgency_level, UrgencyLevel):
            raise ValidationError(f"Invalid urgency level: {self.urgency_level}")
        
        # Validate entities
        for entity in self.entities:
            if not isinstance(entity, (Entity, ValidatedEntity)):
                raise ValidationError(f"Invalid entity type: {type(entity)}")
            
            # If it's a regular Entity, validate it manually
            if isinstance(entity, Entity) and not isinstance(entity, ValidatedEntity):
                if not entity.text or not entity.text.strip():
                    raise ValidationError("Entity text cannot be empty")
                if not 0.0 <= entity.confidence <= 1.0:
                    raise ValidationError(f"Entity confidence must be between 0.0 and 1.0, got {entity.confidence}")
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type."""
        return [entity for entity in self.entities if entity.entity_type == entity_type]
    
    def has_entity_type(self, entity_type: EntityType) -> bool:
        """Check if scenario contains entities of a specific type."""
        return any(entity.entity_type == entity_type for entity in self.entities)
    
    def get_primary_address(self) -> Optional[Entity]:
        """Get the primary address entity (highest confidence)."""
        addresses = self.get_entities_by_type(EntityType.ADDRESS)
        if not addresses:
            return None
        return max(addresses, key=lambda e: e.confidence)
    
    def get_primary_merchant(self) -> Optional[Entity]:
        """Get the primary merchant entity (highest confidence)."""
        merchants = self.get_entities_by_type(EntityType.MERCHANT)
        if not merchants:
            return None
        return max(merchants, key=lambda e: e.confidence)
    
    def get_delivery_ids(self) -> List[str]:
        """Get all delivery IDs from entities."""
        delivery_entities = self.get_entities_by_type(EntityType.DELIVERY_ID)
        return [entity.normalized_value or entity.text for entity in delivery_entities]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'description': self.description,
            'entities': [
                {
                    'text': entity.text,
                    'entity_type': entity.entity_type.value,
                    'confidence': entity.confidence,
                    'normalized_value': entity.normalized_value
                }
                for entity in self.entities
            ],
            'scenario_type': self.scenario_type.value,
            'urgency_level': self.urgency_level.value
        }


class EntityExtractor:
    """Utility class for extracting entities from text."""
    
    # Regex patterns for entity extraction
    DELIVERY_ID_PATTERN = re.compile(r'\b([A-Z]{2,4}\d{3,8})\b')
    PHONE_PATTERN = re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b')
    ADDRESS_PATTERN = re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct)\b', re.IGNORECASE)
    
    # Common merchant keywords
    MERCHANT_KEYWORDS = [
        'restaurant', 'cafe', 'pizza', 'burger', 'chinese', 'italian', 'mexican',
        'thai', 'indian', 'sushi', 'deli', 'bakery', 'grill', 'kitchen', 'bistro',
        'mcdonalds', 'subway', 'starbucks', 'kfc', 'taco bell', 'dominos', 'papa johns'
    ]
    
    def extract_entities(self, text: str) -> List[ValidatedEntity]:
        """Extract entities from text."""
        entities = []
        text_lower = text.lower()
        
        # Extract delivery IDs
        for match in self.DELIVERY_ID_PATTERN.finditer(text):
            entities.append(ValidatedEntity(
                text=match.group(0),
                entity_type=EntityType.DELIVERY_ID,
                confidence=0.9,
                normalized_value=match.group(1)
            ))
        
        # Extract phone numbers
        for match in self.PHONE_PATTERN.finditer(text):
            phone_text = match.group(0)
            normalized = f"({match.group(1)}) {match.group(2)}-{match.group(3)}"
            entities.append(ValidatedEntity(
                text=phone_text,
                entity_type=EntityType.PHONE_NUMBER,
                confidence=0.8,
                normalized_value=normalized
            ))
        
        # Extract addresses
        for match in self.ADDRESS_PATTERN.finditer(text):
            entities.append(ValidatedEntity(
                text=match.group(0),
                entity_type=EntityType.ADDRESS,
                confidence=0.7,
                normalized_value=match.group(0).strip()
            ))
        
        # Extract merchants (keyword-based)
        found_merchants = set()  # Track found merchants to avoid duplicates
        for keyword in self.MERCHANT_KEYWORDS:
            if keyword in text_lower:
                # Find the actual occurrence in the original text
                words = text.split()
                for i, word in enumerate(words):
                    if keyword in word.lower() and word.lower() not in found_merchants:
                        # Take the word and potentially the next word if it looks like a name
                        merchant_text = word
                        if i + 1 < len(words) and words[i + 1][0].isupper() and not words[i + 1].lower() in ['street', 'avenue', 'road', 'drive']:
                            merchant_text += " " + words[i + 1]
                        
                        found_merchants.add(merchant_text.lower())
                        entities.append(ValidatedEntity(
                            text=merchant_text,
                            entity_type=EntityType.MERCHANT,
                            confidence=0.6,
                            normalized_value=merchant_text.strip()
                        ))
                        break
        
        # Extract person names (improved heuristic)
        words = text.split()
        found_names = set()
        exclude_words = {
            'the', 'and', 'or', 'but', 'street', 'avenue', 'road', 'drive', 'palace', 'main', 
            'delivery', 'driver', 'urgent', 'emergency', 'critical', 'high', 'medium', 'low',
            'customer', 'merchant', 'restaurant', 'traffic', 'route', 'order', 'food'
        }
        
        # Common first names for better detection
        common_names = {
            'mike', 'sarah', 'john', 'jane', 'tom', 'mary', 'david', 'lisa', 'chris', 'anna',
            'james', 'jennifer', 'robert', 'linda', 'michael', 'elizabeth', 'william', 'barbara',
            'richard', 'susan', 'joseph', 'jessica', 'thomas', 'karen', 'charles', 'nancy'
        }
        
        for i, word in enumerate(words):
            # Clean the word of punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            if (len(clean_word) > 1 and clean_word[0].isupper() and 
                clean_word.lower() not in exclude_words):
                
                # Check if it's already part of another entity
                word_already_used = any(clean_word.lower() in entity.text.lower() for entity in entities)
                if word_already_used or clean_word.lower() in found_names:
                    continue
                
                # Higher confidence if it's a common name
                confidence = 0.7 if clean_word.lower() in common_names else 0.4
                
                # Look for full name pattern (First Last)
                if (i + 1 < len(words) and len(words) > i + 1):
                    next_word = re.sub(r'[^\w]', '', words[i + 1])
                    if (len(next_word) > 1 and next_word[0].isupper() and 
                        next_word.lower() not in exclude_words and
                        not any(next_word.lower() in entity.text.lower() for entity in entities)):
                        
                        full_name = f"{clean_word} {next_word}"
                        found_names.add(full_name.lower())
                        entities.append(ValidatedEntity(
                            text=full_name,
                            entity_type=EntityType.PERSON,
                            confidence=confidence + 0.1,
                            normalized_value=full_name
                        ))
                        continue
                
                # Single name if it looks like a person name
                if confidence > 0.5:  # Only add single names if high confidence
                    found_names.add(clean_word.lower())
                    entities.append(ValidatedEntity(
                        text=clean_word,
                        entity_type=EntityType.PERSON,
                        confidence=confidence,
                        normalized_value=clean_word
                    ))
        
        return entities
    
    def classify_scenario_type(self, text: str, entities: List[Entity]) -> ScenarioType:
        """Classify the scenario type based on text and entities."""
        text_lower = text.lower()
        
        # Count indicators for each scenario type
        traffic_indicators = ['traffic', 'road', 'closed', 'construction', 'accident', 'jam', 'route', 'detour']
        merchant_indicators = ['restaurant', 'closed', 'busy', 'overloaded', 'kitchen', 'prep time', 'unavailable']
        address_indicators = ['address', 'wrong', 'incorrect', 'missing', 'apartment', 'building', 'location']
        
        traffic_score = sum(1 for indicator in traffic_indicators if indicator in text_lower)
        merchant_score = sum(1 for indicator in merchant_indicators if indicator in text_lower)
        address_score = sum(1 for indicator in address_indicators if indicator in text_lower)
        
        # Boost scores based on entity types present
        if any(e.entity_type == EntityType.MERCHANT for e in entities):
            merchant_score += 2
        if any(e.entity_type == EntityType.ADDRESS for e in entities):
            address_score += 2
        
        # Determine scenario type
        scores = {
            ScenarioType.TRAFFIC: traffic_score,
            ScenarioType.MERCHANT: merchant_score,
            ScenarioType.ADDRESS: address_score
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            return ScenarioType.OTHER
        
        # Check for multi-factor scenarios
        high_scores = [scenario_type for scenario_type, score in scores.items() if score >= max_score - 1 and score > 0]
        if len(high_scores) > 1:
            return ScenarioType.MULTI_FACTOR
        
        return max(scores, key=scores.get)
    
    def determine_urgency_level(self, text: str) -> UrgencyLevel:
        """Determine urgency level based on text content."""
        text_lower = text.lower()
        
        critical_indicators = ['emergency', 'urgent', 'asap', 'immediately', 'critical', 'now']
        high_indicators = ['important', 'priority', 'soon', 'quickly', 'fast']
        low_indicators = ['whenever', 'no rush', 'eventually', 'later']
        
        if any(indicator in text_lower for indicator in critical_indicators):
            return UrgencyLevel.CRITICAL
        elif any(indicator in text_lower for indicator in high_indicators):
            return UrgencyLevel.HIGH
        elif any(indicator in text_lower for indicator in low_indicators):
            return UrgencyLevel.LOW
        else:
            return UrgencyLevel.MEDIUM
    
    def create_scenario_from_text(self, description: str) -> ValidatedDisruptionScenario:
        """Create a complete validated scenario from text description."""
        entities = self.extract_entities(description)
        scenario_type = self.classify_scenario_type(description, entities)
        urgency_level = self.determine_urgency_level(description)
        
        return ValidatedDisruptionScenario(
            description=description,
            entities=entities,
            scenario_type=scenario_type,
            urgency_level=urgency_level
        )

# Enhanced reasoning and plan data structures

@dataclass
class ToolAction:
    """Represents a tool action within a reasoning step."""
    tool_name: str
    parameters: Dict[str, Any]
    
    def __post_init__(self):
        """Validate tool action data."""
        if not self.tool_name or not self.tool_name.strip():
            raise ValidationError("Tool name cannot be empty")
        
        if not isinstance(self.parameters, dict):
            raise ValidationError("Tool parameters must be a dictionary")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'tool_name': self.tool_name,
            'parameters': self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolAction':
        """Create from dictionary."""
        return cls(
            tool_name=data['tool_name'],
            parameters=data['parameters']
        )


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    tool_name: str
    success: bool
    data: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate and set defaults."""
        if not self.tool_name or not self.tool_name.strip():
            raise ValidationError("Tool name cannot be empty")
        
        if self.execution_time < 0:
            raise ValidationError("Execution time cannot be negative")
        
        if not isinstance(self.data, dict):
            raise ValidationError("Tool result data must be a dictionary")
        
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'tool_name': self.tool_name,
            'success': self.success,
            'data': self.data,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolResult':
        """Create from dictionary."""
        timestamp = None
        if data.get('timestamp'):
            timestamp = datetime.fromisoformat(data['timestamp'])
        
        return cls(
            tool_name=data['tool_name'],
            success=data['success'],
            data=data['data'],
            execution_time=data['execution_time'],
            error_message=data.get('error_message'),
            timestamp=timestamp
        )


@dataclass
class ValidatedReasoningStep:
    """Enhanced ReasoningStep with validation and serialization."""
    step_number: int
    thought: str
    action: Optional[ToolAction] = None
    observation: Optional[str] = None
    timestamp: Optional[datetime] = None
    tool_results: List[ToolResult] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate reasoning step data."""
        if self.step_number < 1:
            raise ValidationError("Step number must be positive")
        
        if not self.thought or not self.thought.strip():
            raise ValidationError("Thought cannot be empty")
        
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def add_tool_result(self, result: ToolResult) -> None:
        """Add a tool result to this reasoning step."""
        if not isinstance(result, ToolResult):
            raise ValidationError("Must provide a ToolResult instance")
        self.tool_results.append(result)
    
    def get_successful_results(self) -> List[ToolResult]:
        """Get only successful tool results."""
        return [result for result in self.tool_results if result.success]
    
    def get_failed_results(self) -> List[ToolResult]:
        """Get only failed tool results."""
        return [result for result in self.tool_results if not result.success]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'step_number': self.step_number,
            'thought': self.thought,
            'action': self.action.to_dict() if self.action else None,
            'observation': self.observation,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'tool_results': [result.to_dict() for result in self.tool_results]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidatedReasoningStep':
        """Create from dictionary."""
        timestamp = None
        if data.get('timestamp'):
            timestamp = datetime.fromisoformat(data['timestamp'])
        
        action = None
        if data.get('action'):
            action = ToolAction.from_dict(data['action'])
        
        tool_results = []
        if data.get('tool_results'):
            tool_results = [ToolResult.from_dict(result) for result in data['tool_results']]
        
        return cls(
            step_number=data['step_number'],
            thought=data['thought'],
            action=action,
            observation=data.get('observation'),
            timestamp=timestamp,
            tool_results=tool_results
        )


@dataclass
class ValidatedReasoningTrace:
    """Enhanced ReasoningTrace with validation and serialization."""
    steps: List[ValidatedReasoningStep]
    scenario: ValidatedDisruptionScenario
    start_time: datetime
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate reasoning trace data."""
        if not isinstance(self.steps, list):
            raise ValidationError("Steps must be a list")
        
        if not isinstance(self.scenario, (DisruptionScenario, ValidatedDisruptionScenario)):
            raise ValidationError("Scenario must be a DisruptionScenario instance")
        
        # Validate step numbering
        expected_step = 1
        for step in self.steps:
            if not isinstance(step, (ValidatedReasoningStep)):
                raise ValidationError(f"All steps must be ValidatedReasoningStep instances")
            if step.step_number != expected_step:
                raise ValidationError(f"Step numbering error: expected {expected_step}, got {step.step_number}")
            expected_step += 1
    
    def add_step(self, step: ValidatedReasoningStep) -> None:
        """Add a reasoning step to the trace."""
        if not isinstance(step, ValidatedReasoningStep):
            raise ValidationError("Must provide a ValidatedReasoningStep instance")
        
        expected_step_number = len(self.steps) + 1
        if step.step_number != expected_step_number:
            step.step_number = expected_step_number
        
        self.steps.append(step)
    
    def complete_trace(self) -> None:
        """Mark the reasoning trace as complete."""
        self.end_time = datetime.now()
    
    def get_duration(self) -> Optional[timedelta]:
        """Get the total duration of the reasoning process."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def get_tool_usage_summary(self) -> Dict[str, int]:
        """Get a summary of tool usage across all steps."""
        tool_counts = {}
        for step in self.steps:
            for result in step.tool_results:
                tool_counts[result.tool_name] = tool_counts.get(result.tool_name, 0) + 1
        return tool_counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'steps': [step.to_dict() for step in self.steps],
            'scenario': self.scenario.to_dict() if hasattr(self.scenario, 'to_dict') else {
                'description': self.scenario.description,
                'scenario_type': self.scenario.scenario_type.value,
                'urgency_level': self.scenario.urgency_level.value
            },
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidatedReasoningTrace':
        """Create from dictionary."""
        steps = [ValidatedReasoningStep.from_dict(step_data) for step_data in data['steps']]
        
        # Reconstruct scenario (simplified)
        scenario_data = data['scenario']
        scenario = ValidatedDisruptionScenario(
            description=scenario_data['description'],
            entities=[],  # Would need more complex reconstruction
            scenario_type=ScenarioType(scenario_data['scenario_type']),
            urgency_level=UrgencyLevel(scenario_data['urgency_level'])
        )
        
        start_time = datetime.fromisoformat(data['start_time'])
        end_time = None
        if data.get('end_time'):
            end_time = datetime.fromisoformat(data['end_time'])
        
        return cls(
            steps=steps,
            scenario=scenario,
            start_time=start_time,
            end_time=end_time
        )


@dataclass
class ValidatedPlanStep:
    """Enhanced PlanStep with validation and serialization."""
    sequence: int
    action: str
    responsible_party: str
    estimated_time: timedelta
    dependencies: List[int]
    success_criteria: str
    status: str = "pending"  # pending, in_progress, completed, failed
    actual_duration: Optional[timedelta] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Validate plan step data."""
        if self.sequence < 1:
            raise ValidationError("Sequence number must be positive")
        
        if not self.action or not self.action.strip():
            raise ValidationError("Action cannot be empty")
        
        if not self.responsible_party or not self.responsible_party.strip():
            raise ValidationError("Responsible party cannot be empty")
        
        if self.estimated_time.total_seconds() <= 0:
            raise ValidationError("Estimated time must be positive")
        
        if not self.success_criteria or not self.success_criteria.strip():
            raise ValidationError("Success criteria cannot be empty")
        
        valid_statuses = ["pending", "in_progress", "completed", "failed"]
        if self.status not in valid_statuses:
            raise ValidationError(f"Status must be one of: {valid_statuses}")
        
        # Validate dependencies
        for dep in self.dependencies:
            if not isinstance(dep, int) or dep < 1:
                raise ValidationError("Dependencies must be positive integers")
            if dep >= self.sequence:
                raise ValidationError("Dependencies must reference earlier steps")
    
    def mark_in_progress(self) -> None:
        """Mark the step as in progress."""
        self.status = "in_progress"
    
    def mark_completed(self, actual_duration: Optional[timedelta] = None, notes: Optional[str] = None) -> None:
        """Mark the step as completed."""
        self.status = "completed"
        if actual_duration:
            self.actual_duration = actual_duration
        if notes:
            self.notes = notes
    
    def mark_failed(self, notes: Optional[str] = None) -> None:
        """Mark the step as failed."""
        self.status = "failed"
        if notes:
            self.notes = notes
    
    def is_ready_to_execute(self, completed_steps: List[int]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_steps for dep in self.dependencies)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sequence': self.sequence,
            'action': self.action,
            'responsible_party': self.responsible_party,
            'estimated_time': self.estimated_time.total_seconds(),
            'dependencies': self.dependencies,
            'success_criteria': self.success_criteria,
            'status': self.status,
            'actual_duration': self.actual_duration.total_seconds() if self.actual_duration else None,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidatedPlanStep':
        """Create from dictionary."""
        estimated_time = timedelta(seconds=data['estimated_time'])
        actual_duration = None
        if data.get('actual_duration'):
            actual_duration = timedelta(seconds=data['actual_duration'])
        
        return cls(
            sequence=data['sequence'],
            action=data['action'],
            responsible_party=data['responsible_party'],
            estimated_time=estimated_time,
            dependencies=data['dependencies'],
            success_criteria=data['success_criteria'],
            status=data.get('status', 'pending'),
            actual_duration=actual_duration,
            notes=data.get('notes')
        )


@dataclass
class AlternativePlan:
    """Represents an alternative resolution plan."""
    name: str
    description: str
    steps: List[ValidatedPlanStep]
    estimated_duration: timedelta
    success_probability: float
    trade_offs: List[str]
    
    def __post_init__(self):
        """Validate alternative plan data."""
        if not self.name or not self.name.strip():
            raise ValidationError("Alternative plan name cannot be empty")
        
        if not self.description or not self.description.strip():
            raise ValidationError("Alternative plan description cannot be empty")
        
        if not 0.0 <= self.success_probability <= 1.0:
            raise ValidationError("Success probability must be between 0.0 and 1.0")
        
        if self.estimated_duration.total_seconds() <= 0:
            raise ValidationError("Estimated duration must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'steps': [step.to_dict() for step in self.steps],
            'estimated_duration': self.estimated_duration.total_seconds(),
            'success_probability': self.success_probability,
            'trade_offs': self.trade_offs
        }


@dataclass
class ValidatedResolutionPlan:
    """Enhanced ResolutionPlan with validation and serialization."""
    steps: List[ValidatedPlanStep]
    estimated_duration: timedelta
    success_probability: float
    alternatives: List[AlternativePlan]
    stakeholders: List[str]
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate resolution plan data."""
        if not isinstance(self.steps, list) or len(self.steps) == 0:
            raise ValidationError("Plan must have at least one step")
        
        if not 0.0 <= self.success_probability <= 1.0:
            raise ValidationError("Success probability must be between 0.0 and 1.0")
        
        if self.estimated_duration.total_seconds() <= 0:
            raise ValidationError("Estimated duration must be positive")
        
        # Validate step sequencing
        expected_sequence = 1
        for step in self.steps:
            if not isinstance(step, ValidatedPlanStep):
                raise ValidationError("All steps must be ValidatedPlanStep instances")
            if step.sequence != expected_sequence:
                raise ValidationError(f"Step sequence error: expected {expected_sequence}, got {step.sequence}")
            expected_sequence += 1
        
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def get_next_steps(self) -> List[ValidatedPlanStep]:
        """Get steps that are ready to execute."""
        completed_steps = [step.sequence for step in self.steps if step.status == "completed"]
        return [step for step in self.steps if step.status == "pending" and step.is_ready_to_execute(completed_steps)]
    
    def get_critical_path(self) -> List[ValidatedPlanStep]:
        """Get the critical path through the plan."""
        # Simple implementation - steps with no dependencies or longest dependency chain
        critical_steps = []
        for step in self.steps:
            if not step.dependencies or len(step.dependencies) == max(len(s.dependencies) for s in self.steps):
                critical_steps.append(step)
        return critical_steps
    
    def get_completion_percentage(self) -> float:
        """Get the percentage of completed steps."""
        if not self.steps:
            return 0.0
        completed = len([step for step in self.steps if step.status == "completed"])
        return (completed / len(self.steps)) * 100.0
    
    def add_alternative(self, alternative: AlternativePlan) -> None:
        """Add an alternative plan."""
        if not isinstance(alternative, AlternativePlan):
            raise ValidationError("Must provide an AlternativePlan instance")
        self.alternatives.append(alternative)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'steps': [step.to_dict() for step in self.steps],
            'estimated_duration': self.estimated_duration.total_seconds(),
            'success_probability': self.success_probability,
            'alternatives': [alt.to_dict() for alt in self.alternatives],
            'stakeholders': self.stakeholders,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidatedResolutionPlan':
        """Create from dictionary."""
        steps = [ValidatedPlanStep.from_dict(step_data) for step_data in data['steps']]
        estimated_duration = timedelta(seconds=data['estimated_duration'])
        
        alternatives = []
        if data.get('alternatives'):
            for alt_data in data['alternatives']:
                alt_steps = [ValidatedPlanStep.from_dict(step_data) for step_data in alt_data['steps']]
                alternatives.append(AlternativePlan(
                    name=alt_data['name'],
                    description=alt_data['description'],
                    steps=alt_steps,
                    estimated_duration=timedelta(seconds=alt_data['estimated_duration']),
                    success_probability=alt_data['success_probability'],
                    trade_offs=alt_data['trade_offs']
                ))
        
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])
        
        return cls(
            steps=steps,
            estimated_duration=estimated_duration,
            success_probability=data['success_probability'],
            alternatives=alternatives,
            stakeholders=data['stakeholders'],
            created_at=created_at
        )


@dataclass
class ValidatedResolutionResult:
    """Enhanced ResolutionResult with validation and serialization."""
    scenario: ValidatedDisruptionScenario
    reasoning_trace: ValidatedReasoningTrace
    resolution_plan: ValidatedResolutionPlan
    success: bool
    error_message: Optional[str] = None
    processing_time: Optional[timedelta] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate resolution result data."""
        if not isinstance(self.scenario, (DisruptionScenario, ValidatedDisruptionScenario)):
            raise ValidationError("Scenario must be a DisruptionScenario instance")
        
        if not isinstance(self.reasoning_trace, ValidatedReasoningTrace):
            raise ValidationError("Reasoning trace must be a ValidatedReasoningTrace instance")
        
        if not isinstance(self.resolution_plan, ValidatedResolutionPlan):
            raise ValidationError("Resolution plan must be a ValidatedResolutionPlan instance")
        
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'scenario': self.scenario.to_dict() if hasattr(self.scenario, 'to_dict') else {
                'description': self.scenario.description,
                'scenario_type': self.scenario.scenario_type.value,
                'urgency_level': self.scenario.urgency_level.value
            },
            'reasoning_trace': self.reasoning_trace.to_dict(),
            'resolution_plan': self.resolution_plan.to_dict(),
            'success': self.success,
            'error_message': self.error_message,
            'processing_time': self.processing_time.total_seconds() if self.processing_time else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save the result to a JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ValidatedResolutionResult':
        """Load result from a JSON file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidatedResolutionResult':
        """Create from dictionary."""
        # Reconstruct scenario (simplified)
        scenario_data = data['scenario']
        scenario = ValidatedDisruptionScenario(
            description=scenario_data['description'],
            entities=[],  # Would need more complex reconstruction
            scenario_type=ScenarioType(scenario_data['scenario_type']),
            urgency_level=UrgencyLevel(scenario_data['urgency_level'])
        )
        
        reasoning_trace = ValidatedReasoningTrace.from_dict(data['reasoning_trace'])
        resolution_plan = ValidatedResolutionPlan.from_dict(data['resolution_plan'])
        
        processing_time = None
        if data.get('processing_time'):
            processing_time = timedelta(seconds=data['processing_time'])
        
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])
        
        return cls(
            scenario=scenario,
            reasoning_trace=reasoning_trace,
            resolution_plan=resolution_plan,
            success=data['success'],
            error_message=data.get('error_message'),
            processing_time=processing_time,
            created_at=created_at
        )