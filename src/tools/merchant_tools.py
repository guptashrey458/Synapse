"""
Merchant and delivery tools for logistics coordination.
"""
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .interfaces import Tool, ToolResult


class MerchantStatus(Enum):
    """Merchant operational status."""
    OPEN = "open"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    CLOSED = "closed"
    TEMPORARILY_CLOSED = "temporarily_closed"


class DeliveryStatus(Enum):
    """Delivery tracking status."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PREPARING = "preparing"
    READY_FOR_PICKUP = "ready_for_pickup"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    DELAYED = "delayed"


@dataclass
class MerchantInfo:
    """Information about a merchant."""
    merchant_id: str
    name: str
    address: str
    phone: str
    status: MerchantStatus
    current_prep_time_minutes: int
    average_prep_time_minutes: int
    orders_in_queue: int
    capacity_utilization: float  # 0.0 to 1.0
    estimated_ready_time: Optional[datetime] = None


@dataclass
class DeliveryInfo:
    """Information about a delivery."""
    delivery_id: str
    merchant_id: str
    customer_address: str
    status: DeliveryStatus
    estimated_prep_time: int
    estimated_delivery_time: datetime
    driver_id: Optional[str] = None
    special_instructions: Optional[str] = None


class GetMerchantStatusTool(Tool):
    """Tool for checking merchant status and kitchen prep times."""
    
    def __init__(self):
        super().__init__(
            name="get_merchant_status",
            description="Get current status and prep time information for a merchant",
            parameters={
                "merchant_id": {"type": "string", "description": "Merchant identifier"},
                "merchant_name": {"type": "string", "description": "Merchant name (alternative to ID)", "required": False},
                "include_queue_info": {"type": "boolean", "description": "Include order queue details", "required": False}
            }
        )
        
        # Predefined merchant scenarios for realistic simulation
        self._merchant_scenarios = {
            "normal": {
                "status": MerchantStatus.OPEN,
                "prep_time_range": (10, 20),
                "queue_range": (1, 5),
                "capacity_range": (0.3, 0.7),
                "probability": 0.5
            },
            "busy": {
                "status": MerchantStatus.BUSY,
                "prep_time_range": (20, 35),
                "queue_range": (5, 12),
                "capacity_range": (0.7, 0.9),
                "probability": 0.3
            },
            "overloaded": {
                "status": MerchantStatus.OVERLOADED,
                "prep_time_range": (35, 60),
                "queue_range": (12, 25),
                "capacity_range": (0.9, 1.0),
                "probability": 0.15
            },
            "closed": {
                "status": MerchantStatus.CLOSED,
                "prep_time_range": (0, 0),
                "queue_range": (0, 0),
                "capacity_range": (0.0, 0.0),
                "probability": 0.05
            }
        }
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate required parameters for merchant status check."""
        return "merchant_id" in kwargs or "merchant_name" in kwargs
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute merchant status check with realistic simulation."""
        start_time = time.time()
        
        if not self.validate_parameters(**kwargs):
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message="Missing required parameter: merchant_id or merchant_name"
            )
        
        try:
            # Simulate API call delay
            time.sleep(random.uniform(0.1, 0.4))
            
            merchant_id = kwargs.get("merchant_id", f"MERCH_{random.randint(1000, 9999)}")
            merchant_name = kwargs.get("merchant_name", f"Restaurant {merchant_id[-3:]}")
            include_queue_info = kwargs.get("include_queue_info", True)
            
            # Generate merchant info based on scenarios
            merchant_info = self._generate_merchant_info(merchant_id, merchant_name)
            
            result_data = {
                "merchant_id": merchant_info.merchant_id,
                "name": merchant_info.name,
                "address": merchant_info.address,
                "phone": merchant_info.phone,
                "status": merchant_info.status.value,
                "current_prep_time_minutes": merchant_info.current_prep_time_minutes,
                "average_prep_time_minutes": merchant_info.average_prep_time_minutes,
                "capacity_utilization": round(merchant_info.capacity_utilization, 2),
                "estimated_ready_time": merchant_info.estimated_ready_time.isoformat() if merchant_info.estimated_ready_time else None,
                "last_updated": datetime.now().isoformat()
            }
            
            if include_queue_info:
                result_data["queue_info"] = {
                    "orders_in_queue": merchant_info.orders_in_queue,
                    "estimated_wait_time": merchant_info.current_prep_time_minutes,
                    "queue_status": self._get_queue_status(merchant_info.orders_in_queue)
                }
            
            result_data["recommendations"] = self._generate_merchant_recommendations(merchant_info)
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=result_data,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message=f"Merchant status check failed: {str(e)}"
            )
    
    def _generate_merchant_info(self, merchant_id: str, merchant_name: str) -> MerchantInfo:
        """Generate realistic merchant information."""
        # Select scenario based on probability
        scenario_key = random.choices(
            list(self._merchant_scenarios.keys()),
            weights=[s["probability"] for s in self._merchant_scenarios.values()]
        )[0]
        
        scenario = self._merchant_scenarios[scenario_key]
        
        # Generate values within scenario ranges
        prep_time = random.randint(*scenario["prep_time_range"])
        queue_size = random.randint(*scenario["queue_range"])
        capacity = random.uniform(*scenario["capacity_range"])
        
        # Calculate estimated ready time
        estimated_ready = None
        if scenario["status"] != MerchantStatus.CLOSED:
            estimated_ready = datetime.now() + timedelta(minutes=prep_time)
        
        return MerchantInfo(
            merchant_id=merchant_id,
            name=merchant_name,
            address=f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Elm'])} St",
            phone=f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
            status=scenario["status"],
            current_prep_time_minutes=prep_time,
            average_prep_time_minutes=random.randint(15, 25),
            orders_in_queue=queue_size,
            capacity_utilization=capacity,
            estimated_ready_time=estimated_ready
        )
    
    def _get_queue_status(self, queue_size: int) -> str:
        """Get human-readable queue status."""
        if queue_size == 0:
            return "no_queue"
        elif queue_size <= 3:
            return "light_queue"
        elif queue_size <= 8:
            return "moderate_queue"
        elif queue_size <= 15:
            return "heavy_queue"
        else:
            return "extremely_busy"
    
    def _generate_merchant_recommendations(self, merchant_info: MerchantInfo) -> List[str]:
        """Generate actionable recommendations based on merchant status."""
        recommendations = []
        
        if merchant_info.status == MerchantStatus.CLOSED:
            recommendations.extend([
                "Merchant is currently closed - consider alternative merchants",
                "Check merchant operating hours for reopening time",
                "Notify customer about merchant closure and offer alternatives"
            ])
        elif merchant_info.status == MerchantStatus.OVERLOADED:
            recommendations.extend([
                f"Merchant is overloaded with {merchant_info.orders_in_queue} orders in queue",
                f"Extended prep time of {merchant_info.current_prep_time_minutes} minutes expected",
                "Consider suggesting alternative merchants to customer",
                "Proactively notify customer about significant delays"
            ])
        elif merchant_info.status == MerchantStatus.BUSY:
            recommendations.extend([
                f"Merchant is busy - prep time extended to {merchant_info.current_prep_time_minutes} minutes",
                "Consider notifying customer about potential delays",
                "Monitor merchant status for improvements"
            ])
        else:
            recommendations.append(f"Merchant operating normally - {merchant_info.current_prep_time_minutes} minute prep time")
        
        return recommendations


class GetNearbyMerchantsTool(Tool):
    """Tool for finding nearby merchants as alternatives."""
    
    def __init__(self):
        super().__init__(
            name="get_nearby_merchants",
            description="Find nearby merchants as alternatives for delivery issues",
            parameters={
                "location": {"type": "string", "description": "Address or location to search around"},
                "cuisine_type": {"type": "string", "description": "Type of cuisine to match", "required": False},
                "max_distance_miles": {"type": "number", "description": "Maximum search radius in miles", "required": False},
                "only_open": {"type": "boolean", "description": "Only return open merchants", "required": False}
            }
        )
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate required parameters for nearby merchant search."""
        return "location" in kwargs and bool(kwargs["location"])
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute nearby merchant search."""
        start_time = time.time()
        
        if not self.validate_parameters(**kwargs):
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message="Missing required parameter: location"
            )
        
        try:
            # Simulate API call delay
            time.sleep(random.uniform(0.2, 0.6))
            
            location = kwargs["location"]
            cuisine_type = kwargs.get("cuisine_type", "any")
            max_distance = kwargs.get("max_distance_miles", 5.0)
            only_open = kwargs.get("only_open", True)
            
            # Generate nearby merchants
            nearby_merchants = self._generate_nearby_merchants(
                location, cuisine_type, max_distance, only_open
            )
            
            result_data = {
                "search_location": location,
                "search_radius_miles": max_distance,
                "cuisine_filter": cuisine_type,
                "only_open_filter": only_open,
                "total_found": len(nearby_merchants),
                "merchants": nearby_merchants,
                "search_timestamp": datetime.now().isoformat()
            }
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=result_data,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message=f"Nearby merchant search failed: {str(e)}"
            )
    
    def _generate_nearby_merchants(self, location: str, cuisine_type: str, 
                                 max_distance: float, only_open: bool) -> List[Dict[str, Any]]:
        """Generate list of nearby merchants."""
        merchants = []
        num_merchants = random.randint(3, 8)
        
        cuisine_types = ["italian", "chinese", "mexican", "american", "thai", "indian", "japanese"]
        if cuisine_type != "any" and cuisine_type.lower() in cuisine_types:
            primary_cuisine = cuisine_type.lower()
        else:
            primary_cuisine = random.choice(cuisine_types)
        
        for i in range(num_merchants):
            # Generate merchant details
            merchant_id = f"MERCH_{random.randint(1000, 9999)}"
            distance = round(random.uniform(0.1, max_distance), 1)
            
            # Determine status
            if only_open:
                status = random.choice([MerchantStatus.OPEN, MerchantStatus.BUSY])
            else:
                status = random.choice(list(MerchantStatus))
            
            # Select cuisine (favor the requested type)
            if i < 2 and cuisine_type != "any":  # First 2 results match requested cuisine
                selected_cuisine = primary_cuisine
            else:
                selected_cuisine = random.choice(cuisine_types)
            
            prep_time = random.randint(10, 45) if status != MerchantStatus.CLOSED else 0
            rating = round(random.uniform(3.5, 4.8), 1)
            
            merchant = {
                "merchant_id": merchant_id,
                "name": f"{selected_cuisine.title()} {random.choice(['Kitchen', 'Bistro', 'Restaurant', 'Cafe'])} {i+1}",
                "cuisine_type": selected_cuisine,
                "address": f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Elm', 'Broadway'])} St",
                "distance_miles": distance,
                "estimated_prep_time_minutes": prep_time,
                "status": status.value,
                "rating": rating,
                "phone": f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
                "accepts_new_orders": status in [MerchantStatus.OPEN, MerchantStatus.BUSY],
                "delivery_fee": round(random.uniform(1.99, 4.99), 2)
            }
            
            merchants.append(merchant)
        
        # Sort by distance
        merchants.sort(key=lambda m: m["distance_miles"])
        
        return merchants


class GetDeliveryStatusTool(Tool):
    """Tool for tracking delivery status and updates."""
    
    def __init__(self):
        super().__init__(
            name="get_delivery_status",
            description="Get current status and tracking information for a delivery",
            parameters={
                "delivery_id": {"type": "string", "description": "Delivery identifier"},
                "include_history": {"type": "boolean", "description": "Include status change history", "required": False}
            }
        )
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate required parameters for delivery status check."""
        return "delivery_id" in kwargs and bool(kwargs["delivery_id"])
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute delivery status check."""
        start_time = time.time()
        
        if not self.validate_parameters(**kwargs):
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message="Missing required parameter: delivery_id"
            )
        
        try:
            # Simulate API call delay
            time.sleep(random.uniform(0.1, 0.3))
            
            delivery_id = kwargs["delivery_id"]
            include_history = kwargs.get("include_history", False)
            
            # Generate delivery info
            delivery_info = self._generate_delivery_info(delivery_id)
            
            result_data = {
                "delivery_id": delivery_info.delivery_id,
                "merchant_id": delivery_info.merchant_id,
                "customer_address": delivery_info.customer_address,
                "current_status": delivery_info.status.value,
                "estimated_prep_time_minutes": delivery_info.estimated_prep_time,
                "estimated_delivery_time": delivery_info.estimated_delivery_time.isoformat(),
                "driver_id": delivery_info.driver_id,
                "special_instructions": delivery_info.special_instructions,
                "last_updated": datetime.now().isoformat()
            }
            
            if include_history:
                result_data["status_history"] = self._generate_status_history(delivery_info)
            
            result_data["next_actions"] = self._generate_delivery_actions(delivery_info)
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=result_data,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message=f"Delivery status check failed: {str(e)}"
            )
    
    def _generate_delivery_info(self, delivery_id: str) -> DeliveryInfo:
        """Generate realistic delivery information."""
        # Select random status with realistic probabilities
        status_weights = {
            DeliveryStatus.PENDING: 0.1,
            DeliveryStatus.CONFIRMED: 0.15,
            DeliveryStatus.PREPARING: 0.3,
            DeliveryStatus.READY_FOR_PICKUP: 0.1,
            DeliveryStatus.OUT_FOR_DELIVERY: 0.25,
            DeliveryStatus.DELIVERED: 0.05,
            DeliveryStatus.CANCELLED: 0.02,
            DeliveryStatus.DELAYED: 0.03
        }
        
        status = random.choices(
            list(status_weights.keys()),
            weights=list(status_weights.values())
        )[0]
        
        # Generate timing based on status
        if status in [DeliveryStatus.PENDING, DeliveryStatus.CONFIRMED]:
            prep_time = random.randint(15, 45)
            delivery_time = datetime.now() + timedelta(minutes=prep_time + random.randint(20, 40))
        elif status == DeliveryStatus.PREPARING:
            prep_time = random.randint(5, 25)
            delivery_time = datetime.now() + timedelta(minutes=prep_time + random.randint(15, 30))
        elif status == DeliveryStatus.READY_FOR_PICKUP:
            prep_time = 0
            delivery_time = datetime.now() + timedelta(minutes=random.randint(10, 25))
        elif status == DeliveryStatus.OUT_FOR_DELIVERY:
            prep_time = 0
            delivery_time = datetime.now() + timedelta(minutes=random.randint(5, 20))
        else:
            prep_time = 0
            delivery_time = datetime.now()
        
        driver_id = f"DRV_{random.randint(100, 999)}" if status in [
            DeliveryStatus.OUT_FOR_DELIVERY, DeliveryStatus.DELIVERED
        ] else None
        
        return DeliveryInfo(
            delivery_id=delivery_id,
            merchant_id=f"MERCH_{random.randint(1000, 9999)}",
            customer_address=f"{random.randint(100, 9999)} {random.choice(['Oak', 'Pine', 'Elm', 'Main'])} Ave",
            status=status,
            estimated_prep_time=prep_time,
            estimated_delivery_time=delivery_time,
            driver_id=driver_id,
            special_instructions=random.choice([
                None, "Leave at door", "Ring doorbell", "Call upon arrival", "Apartment 2B"
            ])
        )
    
    def _generate_status_history(self, delivery_info: DeliveryInfo) -> List[Dict[str, Any]]:
        """Generate realistic status change history."""
        history = []
        current_time = datetime.now()
        
        # Generate backwards from current status
        status_progression = [
            DeliveryStatus.PENDING,
            DeliveryStatus.CONFIRMED,
            DeliveryStatus.PREPARING,
            DeliveryStatus.READY_FOR_PICKUP,
            DeliveryStatus.OUT_FOR_DELIVERY,
            DeliveryStatus.DELIVERED
        ]
        
        current_index = status_progression.index(delivery_info.status) if delivery_info.status in status_progression else 2
        
        for i in range(current_index + 1):
            status = status_progression[i]
            timestamp = current_time - timedelta(minutes=random.randint(5, 30) * (current_index - i))
            
            history.append({
                "status": status.value,
                "timestamp": timestamp.isoformat(),
                "notes": self._get_status_notes(status)
            })
        
        return history
    
    def _get_status_notes(self, status: DeliveryStatus) -> str:
        """Get descriptive notes for status changes."""
        notes_map = {
            DeliveryStatus.PENDING: "Order received and awaiting confirmation",
            DeliveryStatus.CONFIRMED: "Order confirmed by merchant",
            DeliveryStatus.PREPARING: "Kitchen has started preparing your order",
            DeliveryStatus.READY_FOR_PICKUP: "Order is ready for driver pickup",
            DeliveryStatus.OUT_FOR_DELIVERY: "Driver has picked up order and is en route",
            DeliveryStatus.DELIVERED: "Order successfully delivered",
            DeliveryStatus.CANCELLED: "Order was cancelled",
            DeliveryStatus.DELAYED: "Order is experiencing delays"
        }
        return notes_map.get(status, "Status updated")
    
    def _generate_delivery_actions(self, delivery_info: DeliveryInfo) -> List[str]:
        """Generate next actions based on delivery status."""
        actions = []
        
        if delivery_info.status == DeliveryStatus.PENDING:
            actions.extend([
                "Wait for merchant confirmation",
                "Monitor for confirmation timeout"
            ])
        elif delivery_info.status == DeliveryStatus.CONFIRMED:
            actions.extend([
                "Monitor preparation progress",
                "Assign driver when ready"
            ])
        elif delivery_info.status == DeliveryStatus.PREPARING:
            actions.extend([
                "Track preparation time",
                "Prepare driver assignment"
            ])
        elif delivery_info.status == DeliveryStatus.READY_FOR_PICKUP:
            actions.extend([
                "Notify assigned driver",
                "Monitor pickup time"
            ])
        elif delivery_info.status == DeliveryStatus.OUT_FOR_DELIVERY:
            actions.extend([
                "Track driver location",
                "Monitor delivery progress"
            ])
        elif delivery_info.status == DeliveryStatus.DELAYED:
            actions.extend([
                "Investigate delay cause",
                "Notify customer with updated ETA",
                "Consider alternative solutions"
            ])
        
        return actions