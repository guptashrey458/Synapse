"""
Unit tests for customer communication tools.
"""
import pytest
from unittest.mock import patch
from datetime import datetime

from src.tools.communication_tools import (
    NotifyCustomerTool, CollectEvidenceTool, IssueInstantRefundTool, SendResolutionNotificationTool,
    NotificationType, CommunicationChannel, EvidenceType, NotificationResult, Evidence
)


class TestNotifyCustomerTool:
    """Test cases for NotifyCustomerTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = NotifyCustomerTool()
    
    def test_tool_initialization(self):
        """Test tool is properly initialized."""
        assert self.tool.name == "notify_customer"
        assert "proactive notifications" in self.tool.description
        assert "delivery_id" in self.tool.parameters
        assert "customer_id" in self.tool.parameters
        assert "message_type" in self.tool.parameters
    
    def test_validate_parameters_success(self):
        """Test parameter validation with valid inputs."""
        valid_params = {
            "delivery_id": "DEL123",
            "customer_id": "CUST456",
            "message_type": "delay_notification"
        }
        assert self.tool.validate_parameters(**valid_params) is True
        
        # With optional parameters
        valid_params.update({
            "message_content": "Custom message",
            "preferred_channel": "sms",
            "urgency": "high"
        })
        assert self.tool.validate_parameters(**valid_params) is True
    
    def test_validate_parameters_missing_required(self):
        """Test parameter validation with missing required fields."""
        # Missing delivery_id
        invalid_params = {"customer_id": "CUST456", "message_type": "delay_notification"}
        assert self.tool.validate_parameters(**invalid_params) is False
        
        # Missing customer_id
        invalid_params = {"delivery_id": "DEL123", "message_type": "delay_notification"}
        assert self.tool.validate_parameters(**invalid_params) is False
        
        # Missing message_type
        invalid_params = {"delivery_id": "DEL123", "customer_id": "CUST456"}
        assert self.tool.validate_parameters(**invalid_params) is False
    
    def test_execute_success(self):
        """Test successful customer notification."""
        params = {
            "delivery_id": "DEL789",
            "customer_id": "CUST123",
            "message_type": "delay_notification",
            "preferred_channel": "sms",
            "urgency": "high"
        }
        
        result = self.tool.execute(**params)
        
        assert result.success is True
        assert result.tool_name == "notify_customer"
        assert result.execution_time > 0
        assert result.error_message is None
        
        # Check result data structure
        data = result.data
        assert data["delivery_id"] == params["delivery_id"]
        assert data["customer_id"] == params["customer_id"]
        assert data["message_type"] == params["message_type"]
        assert data["channel"] == params["preferred_channel"]
        assert "notification_id" in data
        assert "sent_successfully" in data
        assert "delivery_confirmation" in data
        assert "message_content" in data
        assert "estimated_read_time" in data
    
    def test_execute_with_custom_message(self):
        """Test execution with custom message content."""
        params = {
            "delivery_id": "DEL999",
            "customer_id": "CUST999",
            "message_type": "delivery_update",
            "message_content": "Your order is on the way!"
        }
        
        result = self.tool.execute(**params)
        
        assert result.success is True
        data = result.data
        assert data["message_content"] == params["message_content"]
    
    def test_execute_parameter_validation_failure(self):
        """Test execution with invalid parameters."""
        invalid_params = {"delivery_id": "DEL123"}  # Missing required fields
        
        result = self.tool.execute(**invalid_params)
        
        assert result.success is False
        assert "Missing required parameters" in result.error_message
        assert result.data == {}
    
    def test_send_notification(self):
        """Test notification sending simulation."""
        notification_result = self.tool._send_notification(
            "DEL123", "CUST456", NotificationType.DELAY_NOTIFICATION,
            CommunicationChannel.SMS, None, "medium"
        )
        
        assert isinstance(notification_result, NotificationResult)
        assert notification_result.delivery_id == "DEL123"
        assert notification_result.customer_id == "CUST456"
        assert notification_result.message_type == NotificationType.DELAY_NOTIFICATION
        assert notification_result.channel == CommunicationChannel.SMS
        assert isinstance(notification_result.sent_successfully, bool)
        assert isinstance(notification_result.delivery_confirmation, bool)
    
    def test_generate_message_content_custom(self):
        """Test message content generation with custom message."""
        custom_message = "This is a custom message"
        content = self.tool._generate_message_content(
            NotificationType.DELAY_NOTIFICATION, custom_message
        )
        assert content == custom_message
    
    def test_generate_message_content_template(self):
        """Test message content generation with template."""
        content = self.tool._generate_message_content(
            NotificationType.DELAY_NOTIFICATION, None
        )
        assert isinstance(content, str)
        assert len(content) > 0
    
    def test_estimate_read_time(self):
        """Test read time estimation."""
        read_time = self.tool._estimate_read_time(CommunicationChannel.SMS, "high")
        assert isinstance(read_time, str)
        assert "minutes" in read_time
        
        read_time = self.tool._estimate_read_time(CommunicationChannel.EMAIL, "low")
        assert isinstance(read_time, str)
        assert "minutes" in read_time


class TestCollectEvidenceTool:
    """Test cases for CollectEvidenceTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = CollectEvidenceTool()
    
    def test_tool_initialization(self):
        """Test tool is properly initialized."""
        assert self.tool.name == "collect_evidence"
        assert "evidence" in self.tool.description
        assert "delivery_id" in self.tool.parameters
        assert "dispute_type" in self.tool.parameters
        assert "evidence_types" in self.tool.parameters
        assert "collector_id" in self.tool.parameters
    
    def test_validate_parameters_success(self):
        """Test parameter validation with valid inputs."""
        valid_params = {
            "delivery_id": "DEL123",
            "dispute_type": "damaged",
            "evidence_types": ["photo", "gps_location"],
            "collector_id": "DRV456"
        }
        assert self.tool.validate_parameters(**valid_params) is True
        
        # With optional location
        valid_params["location"] = "123 Main St"
        assert self.tool.validate_parameters(**valid_params) is True
    
    def test_validate_parameters_missing_required(self):
        """Test parameter validation with missing required fields."""
        # Missing delivery_id
        invalid_params = {
            "dispute_type": "damaged",
            "evidence_types": ["photo"],
            "collector_id": "DRV456"
        }
        assert self.tool.validate_parameters(**invalid_params) is False
        
        # Missing evidence_types
        invalid_params = {
            "delivery_id": "DEL123",
            "dispute_type": "damaged",
            "collector_id": "DRV456"
        }
        assert self.tool.validate_parameters(**invalid_params) is False
    
    def test_execute_success(self):
        """Test successful evidence collection."""
        params = {
            "delivery_id": "DEL789",
            "dispute_type": "damaged",
            "evidence_types": ["photo", "gps_location", "driver_statement"],
            "collector_id": "DRV123",
            "location": "Customer address"
        }
        
        result = self.tool.execute(**params)
        
        assert result.success is True
        assert result.tool_name == "collect_evidence"
        assert result.execution_time > 0
        assert result.error_message is None
        
        # Check result data structure
        data = result.data
        assert data["delivery_id"] == params["delivery_id"]
        assert data["dispute_type"] == params["dispute_type"]
        assert data["collector_id"] == params["collector_id"]
        assert "total_evidence_items" in data
        assert "evidence_collected" in data
        assert "collection_successful" in data
        assert "next_steps" in data
        
        # Check evidence structure
        if data["evidence_collected"]:
            evidence_item = data["evidence_collected"][0]
            assert "evidence_id" in evidence_item
            assert "type" in evidence_item
            assert "description" in evidence_item
            assert "timestamp" in evidence_item
            assert "verified" in evidence_item
            assert "metadata" in evidence_item
    
    def test_execute_parameter_validation_failure(self):
        """Test execution with invalid parameters."""
        invalid_params = {"delivery_id": "DEL123"}  # Missing required fields
        
        result = self.tool.execute(**invalid_params)
        
        assert result.success is False
        assert "Missing required parameters" in result.error_message
        assert result.data == {}
    
    def test_collect_evidence_item_photo(self):
        """Test photo evidence collection."""
        evidence = self.tool._collect_evidence_item(
            "DEL123", EvidenceType.PHOTO, "DRV456", "Test Location", "damaged"
        )
        
        assert isinstance(evidence, Evidence)
        assert evidence.evidence_type == EvidenceType.PHOTO
        assert "photo" in evidence.description.lower()
        assert "file_name" in evidence.metadata
        assert "gps_coordinates" in evidence.metadata
    
    def test_collect_evidence_item_gps(self):
        """Test GPS evidence collection."""
        evidence = self.tool._collect_evidence_item(
            "DEL123", EvidenceType.GPS_LOCATION, "DRV456", "Test Location", "wrong_address"
        )
        
        assert isinstance(evidence, Evidence)
        assert evidence.evidence_type == EvidenceType.GPS_LOCATION
        assert "gps" in evidence.description.lower()
        assert "latitude" in evidence.metadata
        assert "longitude" in evidence.metadata
    
    def test_collect_evidence_item_statement(self):
        """Test statement evidence collection."""
        evidence = self.tool._collect_evidence_item(
            "DEL123", EvidenceType.DRIVER_STATEMENT, "DRV456", "Test Location", "missing"
        )
        
        assert isinstance(evidence, Evidence)
        assert evidence.evidence_type == EvidenceType.DRIVER_STATEMENT
        assert len(evidence.description) > 0
        assert "confidence_level" in evidence.metadata
    
    def test_generate_next_steps_sufficient_evidence(self):
        """Test next steps generation with sufficient evidence."""
        evidence = [
            Evidence("E1", EvidenceType.PHOTO, "Photo", "DRV1", datetime.now(), {}, True),
            Evidence("E2", EvidenceType.GPS_LOCATION, "GPS", "DRV1", datetime.now(), {}, True)
        ]
        
        next_steps = self.tool._generate_next_steps("damaged", evidence)
        
        assert len(next_steps) > 0
        assert any("sufficient evidence" in step.lower() for step in next_steps)
    
    def test_generate_next_steps_insufficient_evidence(self):
        """Test next steps generation with insufficient evidence."""
        evidence = [
            Evidence("E1", EvidenceType.PHOTO, "Photo", "DRV1", datetime.now(), {}, False)
        ]
        
        next_steps = self.tool._generate_next_steps("damaged", evidence)
        
        assert len(next_steps) > 0
        assert any("additional evidence" in step.lower() for step in next_steps)


class TestIssueInstantRefundTool:
    """Test cases for IssueInstantRefundTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = IssueInstantRefundTool()
    
    def test_tool_initialization(self):
        """Test tool is properly initialized."""
        assert self.tool.name == "issue_instant_refund"
        assert "instant refund" in self.tool.description
        assert "delivery_id" in self.tool.parameters
        assert "customer_id" in self.tool.parameters
        assert "refund_amount" in self.tool.parameters
        assert "refund_reason" in self.tool.parameters
    
    def test_validate_parameters_success(self):
        """Test parameter validation with valid inputs."""
        valid_params = {
            "delivery_id": "DEL123",
            "customer_id": "CUST456",
            "refund_amount": 25.99,
            "refund_reason": "Order damaged"
        }
        assert self.tool.validate_parameters(**valid_params) is True
        
        # With optional refund_type
        valid_params["refund_type"] = "partial"
        assert self.tool.validate_parameters(**valid_params) is True
    
    def test_validate_parameters_missing_required(self):
        """Test parameter validation with missing required fields."""
        # Missing refund_amount
        invalid_params = {
            "delivery_id": "DEL123",
            "customer_id": "CUST456",
            "refund_reason": "Order damaged"
        }
        assert self.tool.validate_parameters(**invalid_params) is False
        
        # Missing refund_reason
        invalid_params = {
            "delivery_id": "DEL123",
            "customer_id": "CUST456",
            "refund_amount": 25.99
        }
        assert self.tool.validate_parameters(**invalid_params) is False
    
    def test_execute_success(self):
        """Test successful refund processing."""
        params = {
            "delivery_id": "DEL789",
            "customer_id": "CUST123",
            "refund_amount": 35.50,
            "refund_reason": "Order never arrived",
            "refund_type": "full"
        }
        
        result = self.tool.execute(**params)
        
        # Note: Success depends on simulation, but we check structure
        assert result.tool_name == "issue_instant_refund"
        assert result.execution_time > 0
        
        # Check result data structure
        data = result.data
        assert data["delivery_id"] == params["delivery_id"]
        assert data["customer_id"] == params["customer_id"]
        assert data["refund_amount"] == params["refund_amount"]
        assert data["refund_reason"] == params["refund_reason"]
        assert "refund_id" in data
        assert "processing_successful" in data
        assert "transaction_id" in data
        assert "estimated_arrival" in data
        assert "refund_method" in data
    
    def test_execute_parameter_validation_failure(self):
        """Test execution with invalid parameters."""
        invalid_params = {"delivery_id": "DEL123"}  # Missing required fields
        
        result = self.tool.execute(**invalid_params)
        
        assert result.success is False
        assert "Missing required parameters" in result.error_message
        assert result.data == {}
    
    def test_process_refund(self):
        """Test refund processing simulation."""
        refund_result = self.tool._process_refund(
            "DEL123", "CUST456", 25.99, "Order damaged", "full"
        )
        
        assert "refund_id" in refund_result
        assert "transaction_id" in refund_result
        assert "success" in refund_result
        assert "refund_method" in refund_result
        assert "estimated_arrival" in refund_result
        assert "notes" in refund_result
        assert isinstance(refund_result["success"], bool)


class TestSendResolutionNotificationTool:
    """Test cases for SendResolutionNotificationTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = SendResolutionNotificationTool()
    
    def test_tool_initialization(self):
        """Test tool is properly initialized."""
        assert self.tool.name == "send_resolution_notification"
        assert "resolution" in self.tool.description
        assert "delivery_id" in self.tool.parameters
        assert "customer_id" in self.tool.parameters
        assert "resolution_type" in self.tool.parameters
        assert "resolution_details" in self.tool.parameters
    
    def test_validate_parameters_success(self):
        """Test parameter validation with valid inputs."""
        valid_params = {
            "delivery_id": "DEL123",
            "customer_id": "CUST456",
            "resolution_type": "refund",
            "resolution_details": "Full refund processed"
        }
        assert self.tool.validate_parameters(**valid_params) is True
        
        # With optional follow_up_required
        valid_params["follow_up_required"] = True
        assert self.tool.validate_parameters(**valid_params) is True
    
    def test_validate_parameters_missing_required(self):
        """Test parameter validation with missing required fields."""
        # Missing resolution_type
        invalid_params = {
            "delivery_id": "DEL123",
            "customer_id": "CUST456",
            "resolution_details": "Full refund processed"
        }
        assert self.tool.validate_parameters(**invalid_params) is False
        
        # Missing resolution_details
        invalid_params = {
            "delivery_id": "DEL123",
            "customer_id": "CUST456",
            "resolution_type": "refund"
        }
        assert self.tool.validate_parameters(**invalid_params) is False
    
    def test_execute_success(self):
        """Test successful resolution notification."""
        params = {
            "delivery_id": "DEL789",
            "customer_id": "CUST123",
            "resolution_type": "refund",
            "resolution_details": "Full refund of $25.99 processed",
            "follow_up_required": True
        }
        
        result = self.tool.execute(**params)
        
        # Note: Success depends on simulation, but we check structure
        assert result.tool_name == "send_resolution_notification"
        assert result.execution_time > 0
        
        # Check result data structure
        data = result.data
        assert data["delivery_id"] == params["delivery_id"]
        assert data["customer_id"] == params["customer_id"]
        assert data["resolution_type"] == params["resolution_type"]
        assert data["resolution_details"] == params["resolution_details"]
        assert "notification_id" in data
        assert "notification_sent" in data
        assert "channels_used" in data
        assert "estimated_read_time" in data
        assert "follow_up_scheduled" in data
        assert "customer_satisfaction_survey_sent" in data
    
    def test_execute_parameter_validation_failure(self):
        """Test execution with invalid parameters."""
        invalid_params = {"delivery_id": "DEL123"}  # Missing required fields
        
        result = self.tool.execute(**invalid_params)
        
        assert result.success is False
        assert "Missing required parameters" in result.error_message
        assert result.data == {}
    
    def test_send_resolution_notification(self):
        """Test resolution notification sending simulation."""
        notification_result = self.tool._send_resolution_notification(
            "DEL123", "CUST456", "refund", "Full refund processed", True
        )
        
        assert "notification_id" in notification_result
        assert "sent" in notification_result
        assert "channels" in notification_result
        assert "estimated_read_time" in notification_result
        assert "survey_sent" in notification_result
        assert isinstance(notification_result["sent"], bool)
        assert isinstance(notification_result["channels"], list)


class TestNotificationType:
    """Test cases for NotificationType enum."""
    
    def test_notification_type_values(self):
        """Test NotificationType enum values."""
        assert NotificationType.DELAY_NOTIFICATION.value == "delay_notification"
        assert NotificationType.ETA_UPDATE.value == "eta_update"
        assert NotificationType.MERCHANT_ISSUE.value == "merchant_issue"
        assert NotificationType.DELIVERY_UPDATE.value == "delivery_update"
        assert NotificationType.RESOLUTION_OFFER.value == "resolution_offer"
        assert NotificationType.REFUND_NOTIFICATION.value == "refund_notification"


class TestCommunicationChannel:
    """Test cases for CommunicationChannel enum."""
    
    def test_communication_channel_values(self):
        """Test CommunicationChannel enum values."""
        assert CommunicationChannel.SMS.value == "sms"
        assert CommunicationChannel.EMAIL.value == "email"
        assert CommunicationChannel.PUSH_NOTIFICATION.value == "push_notification"
        assert CommunicationChannel.PHONE_CALL.value == "phone_call"
        assert CommunicationChannel.IN_APP_MESSAGE.value == "in_app_message"


class TestEvidenceType:
    """Test cases for EvidenceType enum."""
    
    def test_evidence_type_values(self):
        """Test EvidenceType enum values."""
        assert EvidenceType.PHOTO.value == "photo"
        assert EvidenceType.GPS_LOCATION.value == "gps_location"
        assert EvidenceType.TIMESTAMP.value == "timestamp"
        assert EvidenceType.DRIVER_STATEMENT.value == "driver_statement"
        assert EvidenceType.CUSTOMER_STATEMENT.value == "customer_statement"
        assert EvidenceType.MERCHANT_CONFIRMATION.value == "merchant_confirmation"


class TestNotificationResult:
    """Test cases for NotificationResult data class."""
    
    def test_notification_result_creation(self):
        """Test NotificationResult creation and attributes."""
        notification_result = NotificationResult(
            notification_id="NOTIF123",
            delivery_id="DEL456",
            customer_id="CUST789",
            channel=CommunicationChannel.SMS,
            message_type=NotificationType.DELAY_NOTIFICATION,
            sent_successfully=True,
            delivery_confirmation=True,
            customer_response="Thanks for the update"
        )
        
        assert notification_result.notification_id == "NOTIF123"
        assert notification_result.delivery_id == "DEL456"
        assert notification_result.customer_id == "CUST789"
        assert notification_result.channel == CommunicationChannel.SMS
        assert notification_result.message_type == NotificationType.DELAY_NOTIFICATION
        assert notification_result.sent_successfully is True
        assert notification_result.delivery_confirmation is True
        assert notification_result.customer_response == "Thanks for the update"
        assert notification_result.sent_timestamp is not None


class TestEvidence:
    """Test cases for Evidence data class."""
    
    def test_evidence_creation(self):
        """Test Evidence creation and attributes."""
        timestamp = datetime.now()
        evidence = Evidence(
            evidence_id="EVD123",
            evidence_type=EvidenceType.PHOTO,
            description="Photo of damaged package",
            collected_by="DRV456",
            timestamp=timestamp,
            metadata={"file_name": "evidence.jpg"},
            verified=True
        )
        
        assert evidence.evidence_id == "EVD123"
        assert evidence.evidence_type == EvidenceType.PHOTO
        assert evidence.description == "Photo of damaged package"
        assert evidence.collected_by == "DRV456"
        assert evidence.timestamp == timestamp
        assert evidence.metadata == {"file_name": "evidence.jpg"}
        assert evidence.verified is True