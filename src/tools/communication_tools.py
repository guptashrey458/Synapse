"""
Customer communication tools for delivery coordination.
"""
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .interfaces import Tool, ToolResult


class NotificationType(Enum):
    """Types of customer notifications."""
    DELAY_NOTIFICATION = "delay_notification"
    ETA_UPDATE = "eta_update"
    MERCHANT_ISSUE = "merchant_issue"
    DELIVERY_UPDATE = "delivery_update"
    RESOLUTION_OFFER = "resolution_offer"
    REFUND_NOTIFICATION = "refund_notification"


class CommunicationChannel(Enum):
    """Communication channels for customer contact."""
    SMS = "sms"
    EMAIL = "email"
    PUSH_NOTIFICATION = "push_notification"
    PHONE_CALL = "phone_call"
    IN_APP_MESSAGE = "in_app_message"


class EvidenceType(Enum):
    """Types of evidence for dispute resolution."""
    PHOTO = "photo"
    GPS_LOCATION = "gps_location"
    TIMESTAMP = "timestamp"
    DRIVER_STATEMENT = "driver_statement"
    CUSTOMER_STATEMENT = "customer_statement"
    MERCHANT_CONFIRMATION = "merchant_confirmation"


@dataclass
class NotificationResult:
    """Result of a customer notification."""
    notification_id: str
    delivery_id: str
    customer_id: str
    channel: CommunicationChannel
    message_type: NotificationType
    sent_successfully: bool
    delivery_confirmation: bool
    customer_response: Optional[str] = None
    sent_timestamp: datetime = None
    
    def __post_init__(self):
        if self.sent_timestamp is None:
            self.sent_timestamp = datetime.now()


@dataclass
class Evidence:
    """Evidence collected for dispute resolution."""
    evidence_id: str
    evidence_type: EvidenceType
    description: str
    collected_by: str
    timestamp: datetime
    metadata: Dict[str, Any]
    verified: bool = False


class NotifyCustomerTool(Tool):
    """Tool for proactive customer communication."""
    
    def __init__(self):
        super().__init__(
            name="notify_customer",
            description="Send proactive notifications to customers about delivery issues",
            parameters={
                "delivery_id": {"type": "string", "description": "Delivery identifier"},
                "customer_id": {"type": "string", "description": "Customer identifier"},
                "message_type": {"type": "string", "description": "Type of notification to send"},
                "message_content": {"type": "string", "description": "Custom message content", "required": False},
                "preferred_channel": {"type": "string", "description": "Preferred communication channel", "required": False},
                "urgency": {"type": "string", "description": "Message urgency: low, medium, high", "required": False}
            }
        )
        
        # Predefined message templates
        self._message_templates = {
            NotificationType.DELAY_NOTIFICATION: {
                "subject": "Delivery Update - Slight Delay Expected",
                "template": "Hi {customer_name}, your order #{delivery_id} is experiencing a {delay_minutes}-minute delay due to {reason}. New estimated arrival: {new_eta}. We apologize for the inconvenience."
            },
            NotificationType.ETA_UPDATE: {
                "subject": "Updated Delivery Time",
                "template": "Your order #{delivery_id} has an updated delivery time of {new_eta}. Track your order for real-time updates."
            },
            NotificationType.MERCHANT_ISSUE: {
                "subject": "Restaurant Update for Your Order",
                "template": "Hi {customer_name}, {merchant_name} is experiencing {issue_type}. We're working on a solution and will update you shortly. Estimated resolution: {resolution_time}."
            },
            NotificationType.DELIVERY_UPDATE: {
                "subject": "Your Order is On the Way!",
                "template": "Great news! Your order #{delivery_id} is now out for delivery. Your driver {driver_name} will arrive in approximately {eta_minutes} minutes."
            },
            NotificationType.RESOLUTION_OFFER: {
                "subject": "We'd Like to Make This Right",
                "template": "We're sorry about the issue with order #{delivery_id}. We'd like to offer you {resolution_offer} as an apology. Reply to accept or contact us for alternatives."
            }
        }
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate required parameters for customer notification."""
        required = ["delivery_id", "customer_id", "message_type"]
        return all(param in kwargs and bool(kwargs[param]) for param in required)
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute customer notification with realistic simulation."""
        start_time = time.time()
        
        if not self.validate_parameters(**kwargs):
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message="Missing required parameters: delivery_id, customer_id, message_type"
            )
        
        try:
            # Simulate API call delay
            time.sleep(random.uniform(0.1, 0.4))
            
            delivery_id = kwargs["delivery_id"]
            customer_id = kwargs["customer_id"]
            message_type_str = kwargs["message_type"]
            custom_message = kwargs.get("message_content")
            preferred_channel = kwargs.get("preferred_channel", "sms")
            urgency = kwargs.get("urgency", "medium")
            
            # Parse message type
            try:
                message_type = NotificationType(message_type_str)
            except ValueError:
                message_type = NotificationType.DELIVERY_UPDATE
            
            # Parse communication channel
            try:
                channel = CommunicationChannel(preferred_channel)
            except ValueError:
                channel = CommunicationChannel.SMS
            
            # Generate notification result
            notification_result = self._send_notification(
                delivery_id, customer_id, message_type, channel, custom_message, urgency
            )
            
            result_data = {
                "notification_id": notification_result.notification_id,
                "delivery_id": notification_result.delivery_id,
                "customer_id": notification_result.customer_id,
                "channel": notification_result.channel.value,
                "message_type": notification_result.message_type.value,
                "sent_successfully": notification_result.sent_successfully,
                "delivery_confirmation": notification_result.delivery_confirmation,
                "sent_timestamp": notification_result.sent_timestamp.isoformat(),
                "customer_response": notification_result.customer_response,
                "message_content": self._generate_message_content(message_type, custom_message),
                "urgency_level": urgency,
                "estimated_read_time": self._estimate_read_time(channel, urgency)
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
                error_message=f"Customer notification failed: {str(e)}"
            )
    
    def _send_notification(self, delivery_id: str, customer_id: str, 
                          message_type: NotificationType, channel: CommunicationChannel,
                          custom_message: Optional[str], urgency: str) -> NotificationResult:
        """Simulate sending notification to customer."""
        notification_id = f"NOTIF_{random.randint(10000, 99999)}"
        
        # Simulate delivery success rate based on channel
        channel_success_rates = {
            CommunicationChannel.SMS: 0.95,
            CommunicationChannel.EMAIL: 0.85,
            CommunicationChannel.PUSH_NOTIFICATION: 0.90,
            CommunicationChannel.PHONE_CALL: 0.75,
            CommunicationChannel.IN_APP_MESSAGE: 0.88
        }
        
        success_rate = channel_success_rates.get(channel, 0.85)
        sent_successfully = random.random() < success_rate
        
        # Simulate delivery confirmation
        delivery_confirmation = sent_successfully and random.random() < 0.8
        
        # Simulate customer response for certain message types
        customer_response = None
        if sent_successfully and message_type == NotificationType.RESOLUTION_OFFER:
            responses = ["Accept", "Decline", "Contact me", None]
            customer_response = random.choice(responses)
        elif sent_successfully and urgency == "high":
            responses = ["Thanks for the update", "When will it arrive?", "Cancel my order", None]
            customer_response = random.choice(responses)
        
        return NotificationResult(
            notification_id=notification_id,
            delivery_id=delivery_id,
            customer_id=customer_id,
            channel=channel,
            message_type=message_type,
            sent_successfully=sent_successfully,
            delivery_confirmation=delivery_confirmation,
            customer_response=customer_response
        )
    
    def _generate_message_content(self, message_type: NotificationType, 
                                custom_message: Optional[str]) -> str:
        """Generate message content based on type and custom input."""
        if custom_message:
            return custom_message
        
        template_info = self._message_templates.get(message_type)
        if not template_info:
            return "Thank you for your order. We'll keep you updated on your delivery status."
        
        # Fill in template with sample data
        sample_data = {
            "customer_name": "Customer",
            "delivery_id": f"DEL{random.randint(1000, 9999)}",
            "delay_minutes": random.randint(10, 45),
            "reason": random.choice(["heavy traffic", "restaurant delay", "weather conditions"]),
            "new_eta": (datetime.now() + timedelta(minutes=random.randint(20, 60))).strftime("%I:%M %p"),
            "merchant_name": "Restaurant",
            "issue_type": random.choice(["high order volume", "kitchen delays", "temporary closure"]),
            "resolution_time": f"{random.randint(15, 45)} minutes",
            "driver_name": f"Driver {random.randint(1, 999)}",
            "eta_minutes": random.randint(10, 30),
            "resolution_offer": random.choice(["a full refund", "20% off your next order", "free delivery"])
        }
        
        try:
            return template_info["template"].format(**sample_data)
        except KeyError:
            return template_info["template"]
    
    def _estimate_read_time(self, channel: CommunicationChannel, urgency: str) -> str:
        """Estimate when customer will likely read the message."""
        base_minutes = {
            CommunicationChannel.SMS: 5,
            CommunicationChannel.PUSH_NOTIFICATION: 10,
            CommunicationChannel.IN_APP_MESSAGE: 15,
            CommunicationChannel.EMAIL: 30,
            CommunicationChannel.PHONE_CALL: 1
        }
        
        urgency_multiplier = {
            "high": 0.5,
            "medium": 1.0,
            "low": 2.0
        }
        
        base = base_minutes.get(channel, 15)
        multiplier = urgency_multiplier.get(urgency, 1.0)
        estimated_minutes = int(base * multiplier)
        
        return f"{estimated_minutes} minutes"


class CollectEvidenceTool(Tool):
    """Tool for collecting evidence for dispute resolution."""
    
    def __init__(self):
        super().__init__(
            name="collect_evidence",
            description="Collect evidence for delivery dispute resolution",
            parameters={
                "delivery_id": {"type": "string", "description": "Delivery identifier"},
                "dispute_type": {"type": "string", "description": "Type of dispute: damaged, missing, wrong_address, etc."},
                "evidence_types": {"type": "array", "description": "Types of evidence to collect"},
                "collector_id": {"type": "string", "description": "ID of person collecting evidence (driver, agent, etc.)"},
                "location": {"type": "string", "description": "Location where evidence is collected", "required": False}
            }
        )
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate required parameters for evidence collection."""
        required = ["delivery_id", "dispute_type", "evidence_types", "collector_id"]
        return all(param in kwargs and bool(kwargs[param]) for param in required)
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute evidence collection with realistic simulation."""
        start_time = time.time()
        
        if not self.validate_parameters(**kwargs):
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message="Missing required parameters: delivery_id, dispute_type, evidence_types, collector_id"
            )
        
        try:
            # Simulate evidence collection time
            time.sleep(random.uniform(0.2, 0.8))
            
            delivery_id = kwargs["delivery_id"]
            dispute_type = kwargs["dispute_type"]
            evidence_types = kwargs["evidence_types"]
            collector_id = kwargs["collector_id"]
            location = kwargs.get("location", "Delivery location")
            
            # Collect evidence for each requested type
            collected_evidence = []
            for evidence_type_str in evidence_types:
                try:
                    evidence_type = EvidenceType(evidence_type_str)
                    evidence = self._collect_evidence_item(
                        delivery_id, evidence_type, collector_id, location, dispute_type
                    )
                    collected_evidence.append(evidence)
                except ValueError:
                    # Skip invalid evidence types
                    continue
            
            result_data = {
                "delivery_id": delivery_id,
                "dispute_type": dispute_type,
                "collection_location": location,
                "collector_id": collector_id,
                "total_evidence_items": len(collected_evidence),
                "evidence_collected": [
                    {
                        "evidence_id": evidence.evidence_id,
                        "type": evidence.evidence_type.value,
                        "description": evidence.description,
                        "timestamp": evidence.timestamp.isoformat(),
                        "verified": evidence.verified,
                        "metadata": evidence.metadata
                    }
                    for evidence in collected_evidence
                ],
                "collection_timestamp": datetime.now().isoformat(),
                "collection_successful": len(collected_evidence) > 0,
                "next_steps": self._generate_next_steps(dispute_type, collected_evidence)
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
                error_message=f"Evidence collection failed: {str(e)}"
            )
    
    def _collect_evidence_item(self, delivery_id: str, evidence_type: EvidenceType,
                              collector_id: str, location: str, dispute_type: str) -> Evidence:
        """Collect a single piece of evidence."""
        evidence_id = f"EVD_{random.randint(10000, 99999)}"
        
        # Generate evidence based on type
        if evidence_type == EvidenceType.PHOTO:
            description = f"Photo evidence of {dispute_type} at delivery location"
            metadata = {
                "file_name": f"evidence_{evidence_id}.jpg",
                "file_size": f"{random.randint(500, 2000)}KB",
                "resolution": "1920x1080",
                "gps_coordinates": f"{random.uniform(37.0, 38.0):.6f}, {random.uniform(-122.5, -121.5):.6f}"
            }
        elif evidence_type == EvidenceType.GPS_LOCATION:
            description = f"GPS location data for delivery attempt"
            metadata = {
                "latitude": random.uniform(37.0, 38.0),
                "longitude": random.uniform(-122.5, -121.5),
                "accuracy_meters": random.randint(3, 15),
                "timestamp_utc": datetime.now().isoformat()
            }
        elif evidence_type == EvidenceType.TIMESTAMP:
            description = f"Timestamp evidence for delivery events"
            metadata = {
                "arrival_time": (datetime.now() - timedelta(minutes=random.randint(5, 30))).isoformat(),
                "departure_time": datetime.now().isoformat(),
                "duration_minutes": random.randint(2, 10)
            }
        elif evidence_type == EvidenceType.DRIVER_STATEMENT:
            statements = [
                "Customer was not available at delivery address",
                "Package was left at door as requested",
                "Delivered to person who answered the door",
                "Unable to access building - no response to buzzer",
                "Package appeared damaged upon pickup from restaurant"
            ]
            description = random.choice(statements)
            metadata = {
                "statement_length": len(description),
                "confidence_level": random.choice(["high", "medium", "low"])
            }
        elif evidence_type == EvidenceType.CUSTOMER_STATEMENT:
            statements = [
                "Never received the delivery",
                "Package was damaged when received",
                "Wrong items were delivered",
                "Delivery was very late",
                "Driver was unprofessional"
            ]
            description = random.choice(statements)
            metadata = {
                "statement_length": len(description),
                "customer_rating": random.randint(1, 5)
            }
        elif evidence_type == EvidenceType.MERCHANT_CONFIRMATION:
            description = f"Merchant confirmation of order preparation and handoff"
            metadata = {
                "prep_time_minutes": random.randint(15, 45),
                "handoff_time": (datetime.now() - timedelta(minutes=random.randint(10, 60))).isoformat(),
                "merchant_notes": random.choice(["Order prepared as requested", "Special instructions followed", "No issues noted"])
            }
        else:
            description = f"Evidence of type {evidence_type.value}"
            metadata = {}
        
        # Simulate verification status
        verified = random.random() < 0.8  # 80% of evidence gets verified
        
        return Evidence(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            description=description,
            collected_by=collector_id,
            timestamp=datetime.now(),
            metadata=metadata,
            verified=verified
        )
    
    def _generate_next_steps(self, dispute_type: str, evidence: List[Evidence]) -> List[str]:
        """Generate next steps based on collected evidence."""
        next_steps = []
        
        if not evidence:
            next_steps.append("No evidence collected - manual investigation required")
            return next_steps
        
        verified_count = sum(1 for e in evidence if e.verified)
        
        if verified_count >= 2:
            next_steps.append("Sufficient evidence collected for automated resolution")
            
            if dispute_type in ["damaged", "missing"]:
                next_steps.append("Process refund or replacement based on evidence")
            elif dispute_type == "wrong_address":
                next_steps.append("Verify correct address and arrange redelivery")
            else:
                next_steps.append("Review evidence and determine appropriate resolution")
        else:
            next_steps.append("Additional evidence may be needed for resolution")
            next_steps.append("Consider customer service escalation")
        
        # Add evidence-specific steps
        evidence_types = [e.evidence_type for e in evidence]
        if EvidenceType.PHOTO in evidence_types:
            next_steps.append("Review photo evidence for damage assessment")
        if EvidenceType.GPS_LOCATION in evidence_types:
            next_steps.append("Verify delivery location accuracy")
        
        return next_steps


class IssueInstantRefundTool(Tool):
    """Tool for issuing instant refunds to customers."""
    
    def __init__(self):
        super().__init__(
            name="issue_instant_refund",
            description="Issue instant refund to customer for delivery issues",
            parameters={
                "delivery_id": {"type": "string", "description": "Delivery identifier"},
                "customer_id": {"type": "string", "description": "Customer identifier"},
                "refund_amount": {"type": "number", "description": "Refund amount in dollars"},
                "refund_reason": {"type": "string", "description": "Reason for refund"},
                "refund_type": {"type": "string", "description": "Type: full, partial, delivery_fee, etc.", "required": False}
            }
        )
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate required parameters for refund processing."""
        required = ["delivery_id", "customer_id", "refund_amount", "refund_reason"]
        return all(param in kwargs and bool(kwargs[param]) for param in required)
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute instant refund with realistic simulation."""
        start_time = time.time()
        
        if not self.validate_parameters(**kwargs):
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message="Missing required parameters: delivery_id, customer_id, refund_amount, refund_reason"
            )
        
        try:
            # Simulate refund processing time
            time.sleep(random.uniform(0.3, 1.0))
            
            delivery_id = kwargs["delivery_id"]
            customer_id = kwargs["customer_id"]
            refund_amount = float(kwargs["refund_amount"])
            refund_reason = kwargs["refund_reason"]
            refund_type = kwargs.get("refund_type", "full")
            
            # Simulate refund processing
            refund_result = self._process_refund(
                delivery_id, customer_id, refund_amount, refund_reason, refund_type
            )
            
            result_data = {
                "refund_id": refund_result["refund_id"],
                "delivery_id": delivery_id,
                "customer_id": customer_id,
                "refund_amount": refund_amount,
                "refund_reason": refund_reason,
                "refund_type": refund_type,
                "processing_successful": refund_result["success"],
                "transaction_id": refund_result["transaction_id"],
                "estimated_arrival": refund_result["estimated_arrival"],
                "refund_method": refund_result["refund_method"],
                "processing_timestamp": datetime.now().isoformat(),
                "customer_notification_sent": True,
                "internal_notes": refund_result["notes"]
            }
            
            return ToolResult(
                tool_name=self.name,
                success=refund_result["success"],
                data=result_data,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message=f"Refund processing failed: {str(e)}"
            )
    
    def _process_refund(self, delivery_id: str, customer_id: str, amount: float,
                       reason: str, refund_type: str) -> Dict[str, Any]:
        """Simulate refund processing."""
        refund_id = f"REF_{random.randint(100000, 999999)}"
        transaction_id = f"TXN_{random.randint(1000000, 9999999)}"
        
        # Simulate processing success rate (very high for instant refunds)
        success = random.random() < 0.98
        
        if success:
            # Determine refund method and timing
            refund_methods = ["original_payment_method", "store_credit", "digital_wallet"]
            refund_method = random.choice(refund_methods)
            
            if refund_method == "store_credit":
                estimated_arrival = "Immediate"
            elif refund_method == "digital_wallet":
                estimated_arrival = "1-2 business days"
            else:
                estimated_arrival = "3-5 business days"
            
            notes = f"Refund processed successfully for {reason}"
        else:
            refund_method = "failed"
            estimated_arrival = "N/A"
            notes = "Refund processing failed - manual review required"
        
        return {
            "refund_id": refund_id,
            "transaction_id": transaction_id,
            "success": success,
            "refund_method": refund_method,
            "estimated_arrival": estimated_arrival,
            "notes": notes
        }


class SendResolutionNotificationTool(Tool):
    """Tool for sending resolution notifications to customers."""
    
    def __init__(self):
        super().__init__(
            name="send_resolution_notification",
            description="Send notification to customer about issue resolution",
            parameters={
                "delivery_id": {"type": "string", "description": "Delivery identifier"},
                "customer_id": {"type": "string", "description": "Customer identifier"},
                "resolution_type": {"type": "string", "description": "Type of resolution provided"},
                "resolution_details": {"type": "string", "description": "Details of the resolution"},
                "follow_up_required": {"type": "boolean", "description": "Whether follow-up is needed", "required": False}
            }
        )
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate required parameters for resolution notification."""
        required = ["delivery_id", "customer_id", "resolution_type", "resolution_details"]
        return all(param in kwargs and bool(kwargs[param]) for param in required)
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute resolution notification with realistic simulation."""
        start_time = time.time()
        
        if not self.validate_parameters(**kwargs):
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message="Missing required parameters: delivery_id, customer_id, resolution_type, resolution_details"
            )
        
        try:
            # Simulate notification sending time
            time.sleep(random.uniform(0.1, 0.3))
            
            delivery_id = kwargs["delivery_id"]
            customer_id = kwargs["customer_id"]
            resolution_type = kwargs["resolution_type"]
            resolution_details = kwargs["resolution_details"]
            follow_up_required = kwargs.get("follow_up_required", False)
            
            # Generate resolution notification
            notification_result = self._send_resolution_notification(
                delivery_id, customer_id, resolution_type, resolution_details, follow_up_required
            )
            
            result_data = {
                "notification_id": notification_result["notification_id"],
                "delivery_id": delivery_id,
                "customer_id": customer_id,
                "resolution_type": resolution_type,
                "resolution_details": resolution_details,
                "notification_sent": notification_result["sent"],
                "channels_used": notification_result["channels"],
                "estimated_read_time": notification_result["estimated_read_time"],
                "follow_up_scheduled": follow_up_required,
                "customer_satisfaction_survey_sent": notification_result["survey_sent"],
                "sent_timestamp": datetime.now().isoformat()
            }
            
            return ToolResult(
                tool_name=self.name,
                success=notification_result["sent"],
                data=result_data,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message=f"Resolution notification failed: {str(e)}"
            )
    
    def _send_resolution_notification(self, delivery_id: str, customer_id: str,
                                    resolution_type: str, resolution_details: str,
                                    follow_up_required: bool) -> Dict[str, Any]:
        """Simulate sending resolution notification."""
        notification_id = f"RESNOTIF_{random.randint(10000, 99999)}"
        
        # Simulate multi-channel notification
        channels = ["sms", "email"]
        if random.random() < 0.7:  # 70% chance of push notification
            channels.append("push_notification")
        
        # Simulate sending success
        sent = random.random() < 0.95  # 95% success rate
        
        # Simulate customer satisfaction survey
        survey_sent = sent and random.random() < 0.8  # 80% chance if notification sent
        
        # Estimate read time based on resolution type
        if resolution_type in ["refund", "credit"]:
            estimated_read_time = "2-5 minutes"  # High priority
        else:
            estimated_read_time = "10-30 minutes"  # Normal priority
        
        return {
            "notification_id": notification_id,
            "sent": sent,
            "channels": channels,
            "estimated_read_time": estimated_read_time,
            "survey_sent": survey_sent
        }