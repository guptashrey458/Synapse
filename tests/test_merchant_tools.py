"""
Unit tests for merchant and delivery tools.
"""
import pytest
from unittest.mock import patch
from datetime import datetime

from src.tools.merchant_tools import (
    GetMerchantStatusTool, GetNearbyMerchantsTool, GetDeliveryStatusTool,
    MerchantStatus, DeliveryStatus, MerchantInfo, DeliveryInfo
)


class TestGetMerchantStatusTool:
    """Test cases for GetMerchantStatusTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = GetMerchantStatusTool()
    
    def test_tool_initialization(self):
        """Test tool is properly initialized."""
        assert self.tool.name == "get_merchant_status"
        assert "status" in self.tool.description
        assert "merchant_id" in self.tool.parameters
        assert "merchant_name" in self.tool.parameters
    
    def test_validate_parameters_success(self):
        """Test parameter validation with valid inputs."""
        # With merchant_id
        valid_params = {"merchant_id": "MERCH123"}
        assert self.tool.validate_parameters(**valid_params) is True
        
        # With merchant_name
        valid_params = {"merchant_name": "Pizza Palace"}
        assert self.tool.validate_parameters(**valid_params) is True
        
        # With both
        valid_params = {"merchant_id": "MERCH123", "merchant_name": "Pizza Palace"}
        assert self.tool.validate_parameters(**valid_params) is True
    
    def test_validate_parameters_missing_required(self):
        """Test parameter validation with missing required fields."""
        # No merchant identifier
        invalid_params = {"include_queue_info": True}
        assert self.tool.validate_parameters(**invalid_params) is False
        
        # Empty parameters
        invalid_params = {}
        assert self.tool.validate_parameters(**invalid_params) is False
    
    def test_execute_success_with_merchant_id(self):
        """Test successful execution with merchant_id."""
        params = {
            "merchant_id": "MERCH456",
            "include_queue_info": True
        }
        
        result = self.tool.execute(**params)
        
        assert result.success is True
        assert result.tool_name == "get_merchant_status"
        assert result.execution_time > 0
        assert result.error_message is None
        
        # Check result data structure
        data = result.data
        assert data["merchant_id"] == params["merchant_id"]
        assert "name" in data
        assert "address" in data
        assert "phone" in data
        assert "status" in data
        assert "current_prep_time_minutes" in data
        assert "average_prep_time_minutes" in data
        assert "capacity_utilization" in data
        assert "queue_info" in data
        assert "recommendations" in data
        
        # Check queue info structure
        queue_info = data["queue_info"]
        assert "orders_in_queue" in queue_info
        assert "estimated_wait_time" in queue_info
        assert "queue_status" in queue_info
    
    def test_execute_success_with_merchant_name(self):
        """Test successful execution with merchant_name."""
        params = {
            "merchant_name": "Burger Joint",
            "include_queue_info": False
        }
        
        result = self.tool.execute(**params)
        
        assert result.success is True
        data = result.data
        assert "queue_info" not in data  # Should be excluded when include_queue_info is False
    
    def test_execute_parameter_validation_failure(self):
        """Test execution with invalid parameters."""
        invalid_params = {"include_queue_info": True}  # Missing merchant identifier
        
        result = self.tool.execute(**invalid_params)
        
        assert result.success is False
        assert "Missing required parameter" in result.error_message
        assert result.data == {}
    
    def test_generate_merchant_info(self):
        """Test merchant info generation."""
        merchant_info = self.tool._generate_merchant_info("TEST123", "Test Restaurant")
        
        assert isinstance(merchant_info, MerchantInfo)
        assert merchant_info.merchant_id == "TEST123"
        assert merchant_info.name == "Test Restaurant"
        assert isinstance(merchant_info.status, MerchantStatus)
        assert merchant_info.current_prep_time_minutes >= 0
        assert merchant_info.average_prep_time_minutes >= 0
        assert merchant_info.orders_in_queue >= 0
        assert 0.0 <= merchant_info.capacity_utilization <= 1.0
    
    def test_get_queue_status(self):
        """Test queue status categorization."""
        assert self.tool._get_queue_status(0) == "no_queue"
        assert self.tool._get_queue_status(2) == "light_queue"
        assert self.tool._get_queue_status(5) == "moderate_queue"
        assert self.tool._get_queue_status(10) == "heavy_queue"
        assert self.tool._get_queue_status(20) == "extremely_busy"
    
    def test_generate_merchant_recommendations_closed(self):
        """Test recommendations for closed merchant."""
        merchant_info = MerchantInfo(
            merchant_id="TEST", name="Test", address="123 St", phone="123-456-7890",
            status=MerchantStatus.CLOSED, current_prep_time_minutes=0,
            average_prep_time_minutes=15, orders_in_queue=0, capacity_utilization=0.0
        )
        
        recommendations = self.tool._generate_merchant_recommendations(merchant_info)
        
        assert len(recommendations) > 0
        assert any("closed" in rec.lower() for rec in recommendations)
        assert any("alternative" in rec.lower() for rec in recommendations)
    
    def test_generate_merchant_recommendations_overloaded(self):
        """Test recommendations for overloaded merchant."""
        merchant_info = MerchantInfo(
            merchant_id="TEST", name="Test", address="123 St", phone="123-456-7890",
            status=MerchantStatus.OVERLOADED, current_prep_time_minutes=50,
            average_prep_time_minutes=20, orders_in_queue=20, capacity_utilization=1.0
        )
        
        recommendations = self.tool._generate_merchant_recommendations(merchant_info)
        
        assert len(recommendations) > 0
        assert any("overloaded" in rec.lower() for rec in recommendations)
        assert any("alternative" in rec.lower() for rec in recommendations)
        assert any("notify customer" in rec.lower() for rec in recommendations)


class TestGetNearbyMerchantsTool:
    """Test cases for GetNearbyMerchantsTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = GetNearbyMerchantsTool()
    
    def test_tool_initialization(self):
        """Test tool is properly initialized."""
        assert self.tool.name == "get_nearby_merchants"
        assert "nearby merchants" in self.tool.description
        assert "location" in self.tool.parameters
        assert "cuisine_type" in self.tool.parameters
    
    def test_validate_parameters_success(self):
        """Test parameter validation with valid inputs."""
        valid_params = {"location": "123 Main St"}
        assert self.tool.validate_parameters(**valid_params) is True
        
        valid_params = {
            "location": "Downtown",
            "cuisine_type": "italian",
            "max_distance_miles": 3.0,
            "only_open": True
        }
        assert self.tool.validate_parameters(**valid_params) is True
    
    def test_validate_parameters_missing_required(self):
        """Test parameter validation with missing required fields."""
        # Missing location
        invalid_params = {"cuisine_type": "italian"}
        assert self.tool.validate_parameters(**invalid_params) is False
        
        # Empty location
        invalid_params = {"location": ""}
        assert self.tool.validate_parameters(**invalid_params) is False
    
    def test_execute_success(self):
        """Test successful nearby merchant search."""
        params = {
            "location": "City Center",
            "cuisine_type": "italian",
            "max_distance_miles": 5.0,
            "only_open": True
        }
        
        result = self.tool.execute(**params)
        
        assert result.success is True
        assert result.tool_name == "get_nearby_merchants"
        assert result.execution_time > 0
        assert result.error_message is None
        
        # Check result data structure
        data = result.data
        assert data["search_location"] == params["location"]
        assert data["search_radius_miles"] == params["max_distance_miles"]
        assert data["cuisine_filter"] == params["cuisine_type"]
        assert data["only_open_filter"] == params["only_open"]
        assert "total_found" in data
        assert "merchants" in data
        assert isinstance(data["merchants"], list)
        
        # Check merchant structure
        if data["merchants"]:
            merchant = data["merchants"][0]
            assert "merchant_id" in merchant
            assert "name" in merchant
            assert "cuisine_type" in merchant
            assert "address" in merchant
            assert "distance_miles" in merchant
            assert "estimated_prep_time_minutes" in merchant
            assert "status" in merchant
            assert "rating" in merchant
            assert "accepts_new_orders" in merchant
    
    def test_execute_parameter_validation_failure(self):
        """Test execution with invalid parameters."""
        invalid_params = {"cuisine_type": "italian"}  # Missing location
        
        result = self.tool.execute(**invalid_params)
        
        assert result.success is False
        assert "Missing required parameter" in result.error_message
        assert result.data == {}
    
    def test_generate_nearby_merchants(self):
        """Test nearby merchant generation."""
        merchants = self.tool._generate_nearby_merchants(
            "Test Location", "italian", 5.0, True
        )
        
        assert isinstance(merchants, list)
        assert len(merchants) >= 3  # Should generate at least 3 merchants
        
        for merchant in merchants:
            assert "merchant_id" in merchant
            assert "name" in merchant
            assert "cuisine_type" in merchant
            assert "distance_miles" in merchant
            assert merchant["distance_miles"] <= 5.0  # Within max distance
            assert merchant["accepts_new_orders"] is True  # only_open=True
        
        # Check if sorted by distance
        distances = [m["distance_miles"] for m in merchants]
        assert distances == sorted(distances)


class TestGetDeliveryStatusTool:
    """Test cases for GetDeliveryStatusTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = GetDeliveryStatusTool()
    
    def test_tool_initialization(self):
        """Test tool is properly initialized."""
        assert self.tool.name == "get_delivery_status"
        assert "status" in self.tool.description
        assert "delivery_id" in self.tool.parameters
        assert "include_history" in self.tool.parameters
    
    def test_validate_parameters_success(self):
        """Test parameter validation with valid inputs."""
        valid_params = {"delivery_id": "DEL123"}
        assert self.tool.validate_parameters(**valid_params) is True
        
        valid_params = {
            "delivery_id": "DEL456",
            "include_history": True
        }
        assert self.tool.validate_parameters(**valid_params) is True
    
    def test_validate_parameters_missing_required(self):
        """Test parameter validation with missing required fields."""
        # Missing delivery_id
        invalid_params = {"include_history": True}
        assert self.tool.validate_parameters(**invalid_params) is False
        
        # Empty delivery_id
        invalid_params = {"delivery_id": ""}
        assert self.tool.validate_parameters(**invalid_params) is False
    
    def test_execute_success(self):
        """Test successful delivery status check."""
        params = {
            "delivery_id": "DEL789",
            "include_history": True
        }
        
        result = self.tool.execute(**params)
        
        assert result.success is True
        assert result.tool_name == "get_delivery_status"
        assert result.execution_time > 0
        assert result.error_message is None
        
        # Check result data structure
        data = result.data
        assert data["delivery_id"] == params["delivery_id"]
        assert "merchant_id" in data
        assert "customer_address" in data
        assert "current_status" in data
        assert "estimated_prep_time_minutes" in data
        assert "estimated_delivery_time" in data
        assert "status_history" in data
        assert "next_actions" in data
        
        # Check status history structure
        if data["status_history"]:
            history_item = data["status_history"][0]
            assert "status" in history_item
            assert "timestamp" in history_item
            assert "notes" in history_item
    
    def test_execute_without_history(self):
        """Test execution without status history."""
        params = {
            "delivery_id": "DEL999",
            "include_history": False
        }
        
        result = self.tool.execute(**params)
        
        assert result.success is True
        data = result.data
        assert "status_history" not in data
    
    def test_execute_parameter_validation_failure(self):
        """Test execution with invalid parameters."""
        invalid_params = {"include_history": True}  # Missing delivery_id
        
        result = self.tool.execute(**invalid_params)
        
        assert result.success is False
        assert "Missing required parameter" in result.error_message
        assert result.data == {}
    
    def test_generate_delivery_info(self):
        """Test delivery info generation."""
        delivery_info = self.tool._generate_delivery_info("TEST123")
        
        assert isinstance(delivery_info, DeliveryInfo)
        assert delivery_info.delivery_id == "TEST123"
        assert isinstance(delivery_info.status, DeliveryStatus)
        assert delivery_info.estimated_prep_time >= 0
        assert delivery_info.estimated_delivery_time is not None
    
    def test_generate_status_history(self):
        """Test status history generation."""
        delivery_info = DeliveryInfo(
            delivery_id="TEST", merchant_id="MERCH123", customer_address="123 St",
            status=DeliveryStatus.PREPARING, estimated_prep_time=20,
            estimated_delivery_time=datetime.now()
        )
        
        history = self.tool._generate_status_history(delivery_info)
        
        assert isinstance(history, list)
        assert len(history) > 0
        
        for item in history:
            assert "status" in item
            assert "timestamp" in item
            assert "notes" in item
    
    def test_get_status_notes(self):
        """Test status notes generation."""
        notes = self.tool._get_status_notes(DeliveryStatus.CONFIRMED)
        assert isinstance(notes, str)
        assert len(notes) > 0
        
        notes = self.tool._get_status_notes(DeliveryStatus.OUT_FOR_DELIVERY)
        assert "en route" in notes.lower() or "picked up" in notes.lower()
    
    def test_generate_delivery_actions(self):
        """Test delivery actions generation."""
        delivery_info = DeliveryInfo(
            delivery_id="TEST", merchant_id="MERCH123", customer_address="123 St",
            status=DeliveryStatus.PREPARING, estimated_prep_time=20,
            estimated_delivery_time=datetime.now()
        )
        
        actions = self.tool._generate_delivery_actions(delivery_info)
        
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert all(isinstance(action, str) for action in actions)


class TestMerchantStatus:
    """Test cases for MerchantStatus enum."""
    
    def test_merchant_status_values(self):
        """Test MerchantStatus enum values."""
        assert MerchantStatus.OPEN.value == "open"
        assert MerchantStatus.BUSY.value == "busy"
        assert MerchantStatus.OVERLOADED.value == "overloaded"
        assert MerchantStatus.CLOSED.value == "closed"
        assert MerchantStatus.TEMPORARILY_CLOSED.value == "temporarily_closed"


class TestDeliveryStatus:
    """Test cases for DeliveryStatus enum."""
    
    def test_delivery_status_values(self):
        """Test DeliveryStatus enum values."""
        assert DeliveryStatus.PENDING.value == "pending"
        assert DeliveryStatus.CONFIRMED.value == "confirmed"
        assert DeliveryStatus.PREPARING.value == "preparing"
        assert DeliveryStatus.READY_FOR_PICKUP.value == "ready_for_pickup"
        assert DeliveryStatus.OUT_FOR_DELIVERY.value == "out_for_delivery"
        assert DeliveryStatus.DELIVERED.value == "delivered"
        assert DeliveryStatus.CANCELLED.value == "cancelled"
        assert DeliveryStatus.DELAYED.value == "delayed"


class TestMerchantInfo:
    """Test cases for MerchantInfo data class."""
    
    def test_merchant_info_creation(self):
        """Test MerchantInfo creation and attributes."""
        merchant_info = MerchantInfo(
            merchant_id="TEST123",
            name="Test Restaurant",
            address="123 Main St",
            phone="555-1234",
            status=MerchantStatus.OPEN,
            current_prep_time_minutes=20,
            average_prep_time_minutes=15,
            orders_in_queue=5,
            capacity_utilization=0.7
        )
        
        assert merchant_info.merchant_id == "TEST123"
        assert merchant_info.name == "Test Restaurant"
        assert merchant_info.status == MerchantStatus.OPEN
        assert merchant_info.current_prep_time_minutes == 20
        assert merchant_info.capacity_utilization == 0.7
        assert merchant_info.estimated_ready_time is None


class TestDeliveryInfo:
    """Test cases for DeliveryInfo data class."""
    
    def test_delivery_info_creation(self):
        """Test DeliveryInfo creation and attributes."""
        delivery_time = datetime.now()
        delivery_info = DeliveryInfo(
            delivery_id="DEL123",
            merchant_id="MERCH456",
            customer_address="456 Oak Ave",
            status=DeliveryStatus.PREPARING,
            estimated_prep_time=25,
            estimated_delivery_time=delivery_time,
            driver_id="DRV789",
            special_instructions="Ring doorbell"
        )
        
        assert delivery_info.delivery_id == "DEL123"
        assert delivery_info.merchant_id == "MERCH456"
        assert delivery_info.status == DeliveryStatus.PREPARING
        assert delivery_info.estimated_prep_time == 25
        assert delivery_info.estimated_delivery_time == delivery_time
        assert delivery_info.driver_id == "DRV789"
        assert delivery_info.special_instructions == "Ring doorbell"