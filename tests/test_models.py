"""
Unit tests for data models and entity extraction.
"""
import pytest
from datetime import datetime, timedelta
from src.agent.models import (
    ValidatedEntity, ValidatedDisruptionScenario, EntityExtractor,
    ValidationError, ToolAction, ToolResult, ValidatedReasoningStep,
    ValidatedPlanStep, ValidatedResolutionPlan
)
from src.agent.interfaces import EntityType, ScenarioType, UrgencyLevel


class TestValidatedEntity:
    """Test cases for ValidatedEntity class."""
    
    def test_valid_entity_creation(self):
        """Test creating a valid entity."""
        entity = ValidatedEntity(
            text="123 Main Street",
            entity_type=EntityType.ADDRESS,
            confidence=0.8,
            normalized_value="123 Main Street"
        )
        assert entity.text == "123 Main Street"
        assert entity.entity_type == EntityType.ADDRESS
        assert entity.confidence == 0.8
        assert entity.normalized_value == "123 Main Street"
    
    def test_empty_text_validation(self):
        """Test validation fails for empty text."""
        with pytest.raises(ValidationError, match="Entity text cannot be empty"):
            ValidatedEntity(
                text="",
                entity_type=EntityType.ADDRESS,
                confidence=0.8
            )
    
    def test_whitespace_only_text_validation(self):
        """Test validation fails for whitespace-only text."""
        with pytest.raises(ValidationError, match="Entity text cannot be empty"):
            ValidatedEntity(
                text="   ",
                entity_type=EntityType.ADDRESS,
                confidence=0.8
            )
    
    def test_invalid_confidence_validation(self):
        """Test validation fails for invalid confidence values."""
        with pytest.raises(ValidationError, match="Confidence must be between 0.0 and 1.0"):
            ValidatedEntity(
                text="test",
                entity_type=EntityType.ADDRESS,
                confidence=1.5
            )
        
        with pytest.raises(ValidationError, match="Confidence must be between 0.0 and 1.0"):
            ValidatedEntity(
                text="test",
                entity_type=EntityType.ADDRESS,
                confidence=-0.1
            )
    
    def test_delivery_id_validation(self):
        """Test delivery ID format validation."""
        # Valid delivery ID
        entity = ValidatedEntity(
            text="DEL123456",
            entity_type=EntityType.DELIVERY_ID,
            confidence=0.9,
            normalized_value="DEL123456"
        )
        assert entity.normalized_value == "DEL123456"
        
        # Invalid delivery ID format
        with pytest.raises(ValidationError, match="Invalid delivery ID format"):
            ValidatedEntity(
                text="invalid",
                entity_type=EntityType.DELIVERY_ID,
                confidence=0.9,
                normalized_value="invalid123"
            )
    
    def test_phone_number_validation(self):
        """Test phone number format validation."""
        # Valid phone number
        entity = ValidatedEntity(
            text="(555) 123-4567",
            entity_type=EntityType.PHONE_NUMBER,
            confidence=0.8,
            normalized_value="(555) 123-4567"
        )
        assert entity.normalized_value == "(555) 123-4567"
        
        # Invalid phone number format
        with pytest.raises(ValidationError, match="Invalid phone number format"):
            ValidatedEntity(
                text="invalid",
                entity_type=EntityType.PHONE_NUMBER,
                confidence=0.8,
                normalized_value="abc-def-ghij"
            )
    
    def test_address_validation(self):
        """Test address format validation."""
        # Valid address
        entity = ValidatedEntity(
            text="123 Main St",
            entity_type=EntityType.ADDRESS,
            confidence=0.7,
            normalized_value="123 Main St"
        )
        assert entity.normalized_value == "123 Main St"
        
        # Invalid address (too short)
        with pytest.raises(ValidationError, match="Address too short"):
            ValidatedEntity(
                text="123",
                entity_type=EntityType.ADDRESS,
                confidence=0.7,
                normalized_value="123"
            )


class TestValidatedDisruptionScenario:
    """Test cases for ValidatedDisruptionScenario class."""
    
    def test_valid_scenario_creation(self):
        """Test creating a valid disruption scenario."""
        entities = [
            ValidatedEntity(
                text="Pizza Palace",
                entity_type=EntityType.MERCHANT,
                confidence=0.8
            )
        ]
        
        scenario = ValidatedDisruptionScenario(
            description="Pizza Palace is overloaded and running behind schedule",
            entities=entities,
            scenario_type=ScenarioType.MERCHANT,
            urgency_level=UrgencyLevel.MEDIUM
        )
        
        assert scenario.description == "Pizza Palace is overloaded and running behind schedule"
        assert len(scenario.entities) == 1
        assert scenario.scenario_type == ScenarioType.MERCHANT
        assert scenario.urgency_level == UrgencyLevel.MEDIUM
    
    def test_empty_description_validation(self):
        """Test validation fails for empty description."""
        with pytest.raises(ValidationError, match="Scenario description cannot be empty"):
            ValidatedDisruptionScenario(
                description="",
                entities=[],
                scenario_type=ScenarioType.OTHER,
                urgency_level=UrgencyLevel.LOW
            )
    
    def test_short_description_validation(self):
        """Test validation fails for too short description."""
        with pytest.raises(ValidationError, match="Scenario description too short"):
            ValidatedDisruptionScenario(
                description="Short",
                entities=[],
                scenario_type=ScenarioType.OTHER,
                urgency_level=UrgencyLevel.LOW
            )
    
    def test_get_entities_by_type(self):
        """Test getting entities by type."""
        entities = [
            ValidatedEntity("Pizza Palace", EntityType.MERCHANT, 0.8),
            ValidatedEntity("123 Main St", EntityType.ADDRESS, 0.7),
            ValidatedEntity("Joe's Pizza", EntityType.MERCHANT, 0.6)
        ]
        
        scenario = ValidatedDisruptionScenario(
            description="Multiple entities test scenario",
            entities=entities,
            scenario_type=ScenarioType.MULTI_FACTOR,
            urgency_level=UrgencyLevel.MEDIUM
        )
        
        merchants = scenario.get_entities_by_type(EntityType.MERCHANT)
        addresses = scenario.get_entities_by_type(EntityType.ADDRESS)
        
        assert len(merchants) == 2
        assert len(addresses) == 1
        assert merchants[0].text == "Pizza Palace"
        assert merchants[1].text == "Joe's Pizza"
        assert addresses[0].text == "123 Main St"
    
    def test_has_entity_type(self):
        """Test checking if scenario has entities of specific type."""
        entities = [
            ValidatedEntity("Pizza Palace", EntityType.MERCHANT, 0.8)
        ]
        
        scenario = ValidatedDisruptionScenario(
            description="Test scenario with merchant",
            entities=entities,
            scenario_type=ScenarioType.MERCHANT,
            urgency_level=UrgencyLevel.MEDIUM
        )
        
        assert scenario.has_entity_type(EntityType.MERCHANT) is True
        assert scenario.has_entity_type(EntityType.ADDRESS) is False
    
    def test_get_primary_address(self):
        """Test getting primary address (highest confidence)."""
        entities = [
            ValidatedEntity("123 Main St", EntityType.ADDRESS, 0.7),
            ValidatedEntity("456 Oak Ave", EntityType.ADDRESS, 0.9),
            ValidatedEntity("Pizza Palace", EntityType.MERCHANT, 0.8)
        ]
        
        scenario = ValidatedDisruptionScenario(
            description="Test scenario with multiple addresses",
            entities=entities,
            scenario_type=ScenarioType.ADDRESS,
            urgency_level=UrgencyLevel.MEDIUM
        )
        
        primary_address = scenario.get_primary_address()
        assert primary_address is not None
        assert primary_address.text == "456 Oak Ave"
        assert primary_address.confidence == 0.9
    
    def test_get_primary_merchant(self):
        """Test getting primary merchant (highest confidence)."""
        entities = [
            ValidatedEntity("Pizza Palace", EntityType.MERCHANT, 0.6),
            ValidatedEntity("Joe's Pizza", EntityType.MERCHANT, 0.8),
            ValidatedEntity("123 Main St", EntityType.ADDRESS, 0.7)
        ]
        
        scenario = ValidatedDisruptionScenario(
            description="Test scenario with multiple merchants",
            entities=entities,
            scenario_type=ScenarioType.MERCHANT,
            urgency_level=UrgencyLevel.MEDIUM
        )
        
        primary_merchant = scenario.get_primary_merchant()
        assert primary_merchant is not None
        assert primary_merchant.text == "Joe's Pizza"
        assert primary_merchant.confidence == 0.8
    
    def test_get_delivery_ids(self):
        """Test getting all delivery IDs."""
        entities = [
            ValidatedEntity("DEL123456", EntityType.DELIVERY_ID, 0.9, "DEL123456"),
            ValidatedEntity("ORD789012", EntityType.DELIVERY_ID, 0.8, "ORD789012"),
            ValidatedEntity("Pizza Palace", EntityType.MERCHANT, 0.7)
        ]
        
        scenario = ValidatedDisruptionScenario(
            description="Test scenario with delivery IDs",
            entities=entities,
            scenario_type=ScenarioType.OTHER,
            urgency_level=UrgencyLevel.MEDIUM
        )
        
        delivery_ids = scenario.get_delivery_ids()
        assert len(delivery_ids) == 2
        assert "DEL123456" in delivery_ids
        assert "ORD789012" in delivery_ids


class TestEntityExtractor:
    """Test cases for EntityExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = EntityExtractor()
    
    def test_extract_delivery_ids(self):
        """Test extracting delivery IDs from text."""
        text = "Order DEL123456 is delayed and ORD789012 needs attention"
        entities = self.extractor.extract_entities(text)
        
        delivery_entities = [e for e in entities if e.entity_type == EntityType.DELIVERY_ID]
        assert len(delivery_entities) == 2
        
        delivery_ids = [e.normalized_value for e in delivery_entities]
        assert "DEL123456" in delivery_ids
        assert "ORD789012" in delivery_ids
    
    def test_extract_phone_numbers(self):
        """Test extracting phone numbers from text."""
        text = "Customer at (555) 123-4567 or 555-987-6543 needs help"
        entities = self.extractor.extract_entities(text)
        
        phone_entities = [e for e in entities if e.entity_type == EntityType.PHONE_NUMBER]
        assert len(phone_entities) == 2
        
        normalized_phones = [e.normalized_value for e in phone_entities]
        assert "(555) 123-4567" in normalized_phones
        assert "(555) 987-6543" in normalized_phones
    
    def test_extract_addresses(self):
        """Test extracting addresses from text."""
        text = "Delivery to 123 Main Street and pickup from 456 Oak Avenue"
        entities = self.extractor.extract_entities(text)
        
        address_entities = [e for e in entities if e.entity_type == EntityType.ADDRESS]
        assert len(address_entities) == 2
        
        addresses = [e.text for e in address_entities]
        assert "123 Main Street" in addresses
        assert "456 Oak Avenue" in addresses
    
    def test_extract_merchants(self):
        """Test extracting merchants from text."""
        text = "Pizza Palace is overloaded and McDonald's is closed"
        entities = self.extractor.extract_entities(text)
        
        merchant_entities = [e for e in entities if e.entity_type == EntityType.MERCHANT]
        assert len(merchant_entities) >= 1  # At least one merchant should be found
        
        merchant_texts = [e.text.lower() for e in merchant_entities]
        # Should find at least one of the merchants
        assert any('pizza' in text or 'mcdonalds' in text for text in merchant_texts)
    
    def test_classify_scenario_type_traffic(self):
        """Test classifying traffic scenarios."""
        text = "Road is closed due to construction causing traffic jam"
        entities = []
        
        scenario_type = self.extractor.classify_scenario_type(text, entities)
        assert scenario_type == ScenarioType.TRAFFIC
    
    def test_classify_scenario_type_merchant(self):
        """Test classifying merchant scenarios."""
        text = "Restaurant is overloaded and kitchen prep time is extended"
        entities = [ValidatedEntity("Restaurant", EntityType.MERCHANT, 0.8)]
        
        scenario_type = self.extractor.classify_scenario_type(text, entities)
        assert scenario_type == ScenarioType.MERCHANT
    
    def test_classify_scenario_type_address(self):
        """Test classifying address scenarios."""
        text = "Wrong address provided, apartment number is missing"
        entities = [ValidatedEntity("123 Main St", EntityType.ADDRESS, 0.7)]
        
        scenario_type = self.extractor.classify_scenario_type(text, entities)
        assert scenario_type == ScenarioType.ADDRESS
    
    def test_classify_scenario_type_multi_factor(self):
        """Test classifying multi-factor scenarios."""
        text = "Traffic jam and restaurant is closed, wrong address too"
        entities = [
            ValidatedEntity("Restaurant", EntityType.MERCHANT, 0.8),
            ValidatedEntity("123 Main St", EntityType.ADDRESS, 0.7)
        ]
        
        scenario_type = self.extractor.classify_scenario_type(text, entities)
        assert scenario_type == ScenarioType.MULTI_FACTOR
    
    def test_determine_urgency_level_critical(self):
        """Test determining critical urgency level."""
        text = "Emergency situation, need immediate help ASAP"
        urgency = self.extractor.determine_urgency_level(text)
        assert urgency == UrgencyLevel.CRITICAL
    
    def test_determine_urgency_level_high(self):
        """Test determining high urgency level."""
        text = "This is important and needs priority attention"
        urgency = self.extractor.determine_urgency_level(text)
        assert urgency == UrgencyLevel.HIGH
    
    def test_determine_urgency_level_low(self):
        """Test determining low urgency level."""
        text = "Handle this whenever you get a chance, no rush"
        urgency = self.extractor.determine_urgency_level(text)
        assert urgency == UrgencyLevel.LOW
    
    def test_determine_urgency_level_medium(self):
        """Test determining medium urgency level (default)."""
        text = "Standard delivery issue that needs attention"
        urgency = self.extractor.determine_urgency_level(text)
        assert urgency == UrgencyLevel.MEDIUM
    
    def test_create_scenario_from_text(self):
        """Test creating complete scenario from text."""
        text = "Pizza Palace at 123 Main Street is overloaded, delivery DEL123456 is urgent"
        scenario = self.extractor.create_scenario_from_text(text)
        
        assert isinstance(scenario, ValidatedDisruptionScenario)
        assert scenario.description == text
        assert scenario.scenario_type in [ScenarioType.MERCHANT, ScenarioType.MULTI_FACTOR]
        assert scenario.urgency_level == UrgencyLevel.CRITICAL  # "urgent" is a critical indicator
        
        # Check that entities were extracted
        assert len(scenario.entities) > 0
        entity_types = [e.entity_type for e in scenario.entities]
        assert EntityType.DELIVERY_ID in entity_types
    
    def test_extract_person_names(self):
        """Test extracting person names from text."""
        text = "Driver John Smith called about the delivery issue"
        entities = self.extractor.extract_entities(text)
        
        person_entities = [e for e in entities if e.entity_type == EntityType.PERSON]
        assert len(person_entities) >= 1
        
        # Should find "John Smith"
        person_names = [e.text for e in person_entities]
        assert any("John Smith" in name for name in person_names)



class TestToolAction:
    """Test cases for ToolAction class."""
    
    def test_valid_tool_action_creation(self):
        """Test creating a valid tool action."""
        action = ToolAction(
            tool_name="check_traffic",
            parameters={"location": "123 Main St", "radius": 5}
        )
        assert action.tool_name == "check_traffic"
        assert action.parameters["location"] == "123 Main St"
        assert action.parameters["radius"] == 5
    
    def test_empty_tool_name_validation(self):
        """Test validation fails for empty tool name."""
        with pytest.raises(ValidationError, match="Tool name cannot be empty"):
            ToolAction(tool_name="", parameters={})
    
    def test_invalid_parameters_validation(self):
        """Test validation fails for non-dict parameters."""
        with pytest.raises(ValidationError, match="Tool parameters must be a dictionary"):
            ToolAction(tool_name="test_tool", parameters="invalid")
    
    def test_serialization(self):
        """Test tool action serialization."""
        action = ToolAction("test_tool", {"param1": "value1"})
        data = action.to_dict()
        
        assert data["tool_name"] == "test_tool"
        assert data["parameters"]["param1"] == "value1"
        
        # Test deserialization
        restored = ToolAction.from_dict(data)
        assert restored.tool_name == action.tool_name
        assert restored.parameters == action.parameters


class TestToolResult:
    """Test cases for ToolResult class."""
    
    def test_valid_tool_result_creation(self):
        """Test creating a valid tool result."""
        result = ToolResult(
            tool_name="check_traffic",
            success=True,
            data={"traffic_level": "moderate"},
            execution_time=1.5
        )
        assert result.tool_name == "check_traffic"
        assert result.success is True
        assert result.data["traffic_level"] == "moderate"
        assert result.execution_time == 1.5
        assert result.timestamp is not None
    
    def test_negative_execution_time_validation(self):
        """Test validation fails for negative execution time."""
        with pytest.raises(ValidationError, match="Execution time cannot be negative"):
            ToolResult(
                tool_name="test_tool",
                success=True,
                data={},
                execution_time=-1.0
            )
    
    def test_invalid_data_validation(self):
        """Test validation fails for non-dict data."""
        with pytest.raises(ValidationError, match="Tool result data must be a dictionary"):
            ToolResult(
                tool_name="test_tool",
                success=True,
                data="invalid",
                execution_time=1.0
            )
    
    def test_serialization(self):
        """Test tool result serialization."""
        result = ToolResult(
            tool_name="test_tool",
            success=True,
            data={"key": "value"},
            execution_time=1.0,
            error_message="test error"
        )
        data = result.to_dict()
        
        assert data["tool_name"] == "test_tool"
        assert data["success"] is True
        assert data["data"]["key"] == "value"
        assert data["execution_time"] == 1.0
        assert data["error_message"] == "test error"
        assert data["timestamp"] is not None
        
        # Test deserialization
        restored = ToolResult.from_dict(data)
        assert restored.tool_name == result.tool_name
        assert restored.success == result.success
        assert restored.data == result.data
        assert restored.execution_time == result.execution_time
        assert restored.error_message == result.error_message


class TestValidatedReasoningStep:
    """Test cases for ValidatedReasoningStep class."""
    
    def test_valid_reasoning_step_creation(self):
        """Test creating a valid reasoning step."""
        action = ToolAction("check_traffic", {"location": "test"})
        step = ValidatedReasoningStep(
            step_number=1,
            thought="I need to check traffic conditions",
            action=action,
            observation="Traffic is moderate"
        )
        assert step.step_number == 1
        assert step.thought == "I need to check traffic conditions"
        assert step.action == action
        assert step.observation == "Traffic is moderate"
        assert step.timestamp is not None
    
    def test_invalid_step_number_validation(self):
        """Test validation fails for invalid step number."""
        with pytest.raises(ValidationError, match="Step number must be positive"):
            ValidatedReasoningStep(
                step_number=0,
                thought="test thought"
            )
    
    def test_empty_thought_validation(self):
        """Test validation fails for empty thought."""
        with pytest.raises(ValidationError, match="Thought cannot be empty"):
            ValidatedReasoningStep(
                step_number=1,
                thought=""
            )
    
    def test_add_tool_result(self):
        """Test adding tool results to reasoning step."""
        step = ValidatedReasoningStep(step_number=1, thought="test")
        result = ToolResult("test_tool", True, {}, 1.0)
        
        step.add_tool_result(result)
        assert len(step.tool_results) == 1
        assert step.tool_results[0] == result
    
    def test_get_successful_results(self):
        """Test getting only successful tool results."""
        step = ValidatedReasoningStep(step_number=1, thought="test")
        success_result = ToolResult("tool1", True, {}, 1.0)
        failure_result = ToolResult("tool2", False, {}, 1.0)
        
        step.add_tool_result(success_result)
        step.add_tool_result(failure_result)
        
        successful = step.get_successful_results()
        assert len(successful) == 1
        assert successful[0] == success_result
    
    def test_serialization(self):
        """Test reasoning step serialization."""
        action = ToolAction("test_tool", {"param": "value"})
        step = ValidatedReasoningStep(
            step_number=1,
            thought="test thought",
            action=action,
            observation="test observation"
        )
        
        data = step.to_dict()
        assert data["step_number"] == 1
        assert data["thought"] == "test thought"
        assert data["action"]["tool_name"] == "test_tool"
        assert data["observation"] == "test observation"
        
        # Test deserialization
        restored = ValidatedReasoningStep.from_dict(data)
        assert restored.step_number == step.step_number
        assert restored.thought == step.thought
        assert restored.action.tool_name == step.action.tool_name
        assert restored.observation == step.observation


class TestValidatedPlanStep:
    """Test cases for ValidatedPlanStep class."""
    
    def test_valid_plan_step_creation(self):
        """Test creating a valid plan step."""
        step = ValidatedPlanStep(
            sequence=1,
            action="Contact customer",
            responsible_party="Customer Service",
            estimated_time=timedelta(minutes=5),
            dependencies=[],
            success_criteria="Customer acknowledges receipt of message"
        )
        assert step.sequence == 1
        assert step.action == "Contact customer"
        assert step.responsible_party == "Customer Service"
        assert step.estimated_time == timedelta(minutes=5)
        assert step.dependencies == []
        assert step.success_criteria == "Customer acknowledges receipt of message"
        assert step.status == "pending"
    
    def test_invalid_sequence_validation(self):
        """Test validation fails for invalid sequence number."""
        with pytest.raises(ValidationError, match="Sequence number must be positive"):
            ValidatedPlanStep(
                sequence=0,
                action="test",
                responsible_party="test",
                estimated_time=timedelta(minutes=1),
                dependencies=[],
                success_criteria="test"
            )
    
    def test_empty_action_validation(self):
        """Test validation fails for empty action."""
        with pytest.raises(ValidationError, match="Action cannot be empty"):
            ValidatedPlanStep(
                sequence=1,
                action="",
                responsible_party="test",
                estimated_time=timedelta(minutes=1),
                dependencies=[],
                success_criteria="test"
            )
    
    def test_invalid_dependencies_validation(self):
        """Test validation fails for invalid dependencies."""
        with pytest.raises(ValidationError, match="Dependencies must reference earlier steps"):
            ValidatedPlanStep(
                sequence=1,
                action="test",
                responsible_party="test",
                estimated_time=timedelta(minutes=1),
                dependencies=[2],  # Can't depend on later step
                success_criteria="test"
            )
    
    def test_status_management(self):
        """Test plan step status management."""
        step = ValidatedPlanStep(
            sequence=1,
            action="test",
            responsible_party="test",
            estimated_time=timedelta(minutes=1),
            dependencies=[],
            success_criteria="test"
        )
        
        assert step.status == "pending"
        
        step.mark_in_progress()
        assert step.status == "in_progress"
        
        step.mark_completed(timedelta(minutes=2), "Completed successfully")
        assert step.status == "completed"
        assert step.actual_duration == timedelta(minutes=2)
        assert step.notes == "Completed successfully"
    
    def test_is_ready_to_execute(self):
        """Test checking if step is ready to execute."""
        step = ValidatedPlanStep(
            sequence=3,
            action="test",
            responsible_party="test",
            estimated_time=timedelta(minutes=1),
            dependencies=[1, 2],
            success_criteria="test"
        )
        
        assert step.is_ready_to_execute([1]) is False  # Missing dependency 2
        assert step.is_ready_to_execute([1, 2]) is True  # All dependencies satisfied
        assert step.is_ready_to_execute([1, 2, 3]) is True  # Extra completed steps OK


class TestValidatedResolutionPlan:
    """Test cases for ValidatedResolutionPlan class."""
    
    def test_valid_resolution_plan_creation(self):
        """Test creating a valid resolution plan."""
        steps = [
            ValidatedPlanStep(1, "Step 1", "Agent", timedelta(minutes=5), [], "Done"),
            ValidatedPlanStep(2, "Step 2", "Agent", timedelta(minutes=3), [1], "Done")
        ]
        
        plan = ValidatedResolutionPlan(
            steps=steps,
            estimated_duration=timedelta(minutes=8),
            success_probability=0.85,
            alternatives=[],
            stakeholders=["Customer", "Driver", "Restaurant"]
        )
        
        assert len(plan.steps) == 2
        assert plan.estimated_duration == timedelta(minutes=8)
        assert plan.success_probability == 0.85
        assert len(plan.stakeholders) == 3
        assert plan.created_at is not None
    
    def test_empty_steps_validation(self):
        """Test validation fails for empty steps."""
        with pytest.raises(ValidationError, match="Plan must have at least one step"):
            ValidatedResolutionPlan(
                steps=[],
                estimated_duration=timedelta(minutes=1),
                success_probability=0.5,
                alternatives=[],
                stakeholders=[]
            )
    
    def test_invalid_success_probability_validation(self):
        """Test validation fails for invalid success probability."""
        steps = [ValidatedPlanStep(1, "test", "test", timedelta(minutes=1), [], "test")]
        
        with pytest.raises(ValidationError, match="Success probability must be between 0.0 and 1.0"):
            ValidatedResolutionPlan(
                steps=steps,
                estimated_duration=timedelta(minutes=1),
                success_probability=1.5,
                alternatives=[],
                stakeholders=[]
            )
    
    def test_step_sequence_validation(self):
        """Test validation fails for incorrect step sequencing."""
        steps = [
            ValidatedPlanStep(1, "Step 1", "Agent", timedelta(minutes=1), [], "Done"),
            ValidatedPlanStep(3, "Step 3", "Agent", timedelta(minutes=1), [], "Done")  # Missing step 2
        ]
        
        with pytest.raises(ValidationError, match="Step sequence error"):
            ValidatedResolutionPlan(
                steps=steps,
                estimated_duration=timedelta(minutes=2),
                success_probability=0.5,
                alternatives=[],
                stakeholders=[]
            )
    
    def test_get_next_steps(self):
        """Test getting next executable steps."""
        steps = [
            ValidatedPlanStep(1, "Step 1", "Agent", timedelta(minutes=1), [], "Done"),
            ValidatedPlanStep(2, "Step 2", "Agent", timedelta(minutes=1), [1], "Done"),
            ValidatedPlanStep(3, "Step 3", "Agent", timedelta(minutes=1), [2], "Done")
        ]
        
        plan = ValidatedResolutionPlan(
            steps=steps,
            estimated_duration=timedelta(minutes=3),
            success_probability=0.5,
            alternatives=[],
            stakeholders=[]
        )
        
        # Mark step 1 as completed
        steps[0].mark_completed()
        
        next_steps = plan.get_next_steps()
        assert len(next_steps) == 1
        assert next_steps[0].sequence == 2
    
    def test_get_completion_percentage(self):
        """Test getting completion percentage."""
        steps = [
            ValidatedPlanStep(1, "Step 1", "Agent", timedelta(minutes=1), [], "Done"),
            ValidatedPlanStep(2, "Step 2", "Agent", timedelta(minutes=1), [], "Done")
        ]
        
        plan = ValidatedResolutionPlan(
            steps=steps,
            estimated_duration=timedelta(minutes=2),
            success_probability=0.5,
            alternatives=[],
            stakeholders=[]
        )
        
        assert plan.get_completion_percentage() == 0.0
        
        steps[0].mark_completed()
        assert plan.get_completion_percentage() == 50.0
        
        steps[1].mark_completed()
        assert plan.get_completion_percentage() == 100.0


if __name__ == "__main__":
    pytest.main([__file__])